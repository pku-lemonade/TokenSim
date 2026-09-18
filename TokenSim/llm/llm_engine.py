from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional
import logging
import math
import time

import simpy

from TokenSim.af import (
    AFCacheConfig,
    AFComputeLatencyProvider,
    AFExecutor,
    AFIteration,
    AFResourceRegistry,
    AFTransferLatencyProvider,
    AFRankBinding,
    AFRuntimePlan,
    compile_af_runtime_plan,
    phase_for_requests,
)
from TokenSim.block.replica_kv import ReplicaKVBlockManager
from TokenSim.config.config import (
    CacheConfig,
    ClusterConfig,
    KVTransferConfig,
    ParallelConfig,
    ParallelRankInfo,
    WorkerConfig,
    _GB,
)
from TokenSim.config.psla_config import PSLAConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.kv_transfer import (
    ConnectorStats,
    KVConnectorFactory,
    KVConnectorMetadata,
    P2PConnector,
)
from TokenSim.hardware import HardwareContext
from TokenSim.hardware.placement import build_topology_placement
from TokenSim.hardware.topology import TopologyPlacement
from TokenSim.latency import build_latency_backend
from TokenSim.latency.operator_table import collective_model_for
from TokenSim.llm.llm_request import Request, RequestStatus
from TokenSim.llm.llm_scheduler import (
    LLMDynamicScheduler,
    LLMPagedAttnScheduler,
    LLMStaticScheduler,
    SchedulePhase,
)
from TokenSim.placement import (
    BalancedLoadWorkerPool,
    DataParallelWorkerPool,
    LeastGpuMemoryWorkerPool,
    RoundRobinWorkerPool,
)
from TokenSim.parallel import ParallelCommunicator
from TokenSim.moe import MoEModelConfig, ExpertPlacement, build_expert_placement
from TokenSim.config.model_config import ModelSpec


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Task(Enum):
    ADD = auto()
    STEP = auto()
    STOP = auto()


@dataclass
class Message:
    task: Task
    sender_id: int = 0
    requests: list[Request] | None = None
    metadata: Any | None = None

    def __lt__(self, other: Message):
        return self.task.value < other.task.value


class WorkerStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    STOPPED = auto()


class AFComputeWorker:
    """Physical AF-only worker that never owns requests or KV cache."""

    def __init__(
        self,
        env: simpy.Environment,
        id: int,
        config: WorkerConfig,
        *,
        runtime_bindings: tuple[AFRankBinding, ...] = (),
        runtime_state=None,
    ) -> None:
        self.env = env
        self.id = id
        self.role = None
        self.name = config.name
        self.capabilities = config.capabilities
        self.hardware = config.hardware
        self.network = config.network
        self.nettype = config.nettype
        self.rank_info = config.rank_info or ParallelRankInfo()
        self.physical_worker_ref = config.physical_worker_ref
        self.global_rank = self.rank_info.global_rank
        self.tp_rank = self.rank_info.tp_rank
        self.pp_rank = self.rank_info.pp_rank
        self.dp_rank = self.rank_info.dp_rank
        self.rank_in_dp_group = self.rank_info.rank_in_dp_group
        self.kv_cache_group_id = self.rank_info.kv_cache_group_id
        self.status = WorkerStatus.WAITING
        self.preempted_cnt = 0
        self.af_runtime_bindings = runtime_bindings
        self.af_runtime_state = runtime_state
        self.owned_expert_ids = sorted(
            {
                expert_id
                for binding in runtime_bindings
                for expert_id in binding.expert_ids
            }
        )

    def workload(self) -> float:
        return 0.0

    def cpu_workload(self) -> float:
        return 0.0


class RequestCompletionDebugPrinter:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.wall_start: float | None = None

    def start(self, wall_start: float | None = None) -> None:
        self.wall_start = time.perf_counter() if wall_start is None else wall_start

    def record(self, event: str, request: Request, timestamp: float) -> None:
        if not self.enabled or self.wall_start is None:
            return
        wall_elapsed = time.perf_counter() - self.wall_start
        print(
            f"[request={request.id}] debug event={event} "
            f"wall_elapsed_s={wall_elapsed:.6f} "
            f"request_id={request.id} input_len={request.prefill_len} "
            f"output_len={request.generation_idx} timestamp={timestamp:.6f}"
        )


class Worker:
    def __init__(self, env: simpy.Environment, id: int) -> None:
        self.env = env
        self.id = id
        self.msg_queue = simpy.PriorityStore(self.env)

    def send_task(
        self,
        worker: Worker,
        task: Task,
        requests: Optional[list[Request]] = None,
        metadata=None,
    ):
        worker.msg_queue.put(Message(task, self.id, requests, metadata))

    def check(self, requests: list[Request]):
        logger.debug(
            "worker %s check at time=%s requests=%s",
            self.id,
            self.env.now,
            _request_ids(requests),
        )


def _request_ids(requests: list[Request] | None) -> list[int]:
    if not requests:
        return []
    return [req.id for req in requests]


def _merge_connector_metadata(
    left: KVConnectorMetadata,
    right: KVConnectorMetadata | None,
) -> KVConnectorMetadata:
    if right is None:
        return left
    return KVConnectorMetadata(
        connector_name=right.connector_name if right.has_work else left.connector_name,
        loads=[*left.loads, *right.loads],
        saves=[*left.saves, *right.saves],
        preempted_request_ids=[
            *left.preempted_request_ids,
            *right.preempted_request_ids,
        ],
    )


class LLMWorker(Worker):
    def __init__(
        self,
        env: simpy.Environment,
        id: int,
        engine: LLMEngine,
        psla_config: PSLAConfig,
        worker_config: WorkerConfig,
        block_size: int,
        batching: str,
        kv_transfer_config: KVTransferConfig,
        hardware: HardwareContext,
        model_spec: ModelSpec,
        placement: TopologyPlacement,
        max_parallem_sum: int,
        max_occupy_ratio: float = 1,
        parallel_config: ParallelConfig | None = None,
        moe_config: MoEModelConfig | None = None,
        expert_placement: ExpertPlacement | None = None,
        latency_backend_type: str = "operator_table",
        latency_fallback: str = "table_first",
        operator_backend: str | None = None,
        decode_context_bucket: int = 128,
        random_seed: int = 0,
        runtime_role: str | None = None,
        cache_config_override: AFCacheConfig | None = None,
        block_manager: ReplicaKVBlockManager | None = None,
        runtime_bindings: tuple[AFRankBinding, ...] = (),
        runtime_state=None,
    ):
        super().__init__(env, id)
        self.hardware_context = hardware
        self.engine = engine

        self.role = runtime_role or worker_config.role
        self.name = worker_config.name
        self.capabilities = worker_config.capabilities
        self.hardware = worker_config.hardware
        self.device = hardware.device(worker_config.hardware)
        self.network = worker_config.network
        self.nettype = worker_config.nettype
        self.model = psla_config.model
        self.model_spec = model_spec
        self.parallel_config = parallel_config or ParallelConfig.default()
        self.moe_config = moe_config or model_spec.moe
        self.expert_placement = expert_placement
        self.af_runtime_bindings = runtime_bindings
        self.af_runtime_state = runtime_state
        leader_binding = next(
            (binding for binding in runtime_bindings if binding.is_scheduler_leader),
            None,
        )
        self.rank_info = (
            ParallelRankInfo(
                global_rank=leader_binding.domain_rank,
                rank_in_dp_group=leader_binding.tp_rank,
                tp_rank=leader_binding.tp_rank,
                pp_rank=leader_binding.pp_rank,
                dp_rank=leader_binding.dp_rank,
                kv_cache_group_id=(
                    f"{leader_binding.domain}:dp{leader_binding.dp_rank}"
                ),
            )
            if leader_binding is not None
            else worker_config.rank_info or ParallelRankInfo()
        )
        self.physical_worker_ref = worker_config.physical_worker_ref
        self.global_rank = self.rank_info.global_rank
        self.tp_rank = self.rank_info.tp_rank
        self.pp_rank = self.rank_info.pp_rank
        self.dp_rank = self.rank_info.dp_rank
        self.rank_in_dp_group = self.rank_info.rank_in_dp_group
        self.kv_cache_group_id = self.rank_info.kv_cache_group_id
        self.placement = placement
        self.device_index = placement.device_index(self.id)

        self.block_size = block_size
        if cache_config_override is not None:
            # AF attention replicas size their KV ledger from the runtime plan.
            self.cache_config = cache_config_override
        else:
            self.cache_config = CacheConfig(
                self.block_size,
                self.device,
                self.model_spec,
                parallel_config=self.parallel_config,
                rank_info=self.rank_info,
                expert_placement=self.expert_placement,
            )
        self.preempted_cnt = 0
        self.pending_connector_metadata: KVConnectorMetadata | None = None

        backend_name = worker_config.operator_backend or operator_backend
        self.operator_package = (
            hardware.operator_package(self.device.device_id, backend_name)
            if latency_backend_type != "analytical"
            else None
        )
        self.collective_model = collective_model_for(
            self.device, self.placement, hardware.links, self.operator_package
        )
        self.parallel_communicator = ParallelCommunicator(
            placement=self.placement,
            collective_model=self.collective_model,
            workers=self.engine.workers,
            worker_id=self.id,
            rank_info=self.rank_info,
            parallel_config=self.parallel_config,
            dtype=self.model_spec.dtype,
        )

        self.latency_backend = build_latency_backend(
            backend_type=latency_backend_type,
            device=self.device,
            model=self.model_spec,
            package=self.operator_package,
            parallel_config=self.parallel_config,
            rank_info=self.rank_info,
            communicator=self.parallel_communicator,
            expert_placement=self.expert_placement,
            fallback=latency_fallback,
            decode_context_bucket=decode_context_bucket,
            random_seed=random_seed + id,
        )
        runtime_experts = {
            expert_id
            for binding in runtime_bindings
            for expert_id in binding.expert_ids
        }
        self.owned_expert_ids = (
            sorted(runtime_experts)
            if runtime_bindings
            else (
                self.expert_placement.experts_for_rank(self.rank_info)
                if self.expert_placement is not None
                else []
            )
        )

        self.kv_transfer_config = kv_transfer_config.with_worker_defaults(
            self.role,
            kv_rank=id,
        )
        self.connector = KVConnectorFactory.create_connector(
            self.kv_transfer_config,
            self.cache_config,
        )

        if batching == "paged-attn":
            self.scheduler = LLMPagedAttnScheduler(
                self.id,
                cache_config=self.cache_config,
                max_parallem_sum=max_parallem_sum,
                max_occupy_ratio=max_occupy_ratio if self.role != "prefill" else 1,
                connector=self.connector,
                block_manager=block_manager,
            )
        elif batching == "static":
            self.scheduler = LLMStaticScheduler(
                max_parallem_sum=max_parallem_sum,
                connector=self.connector,
            )
        else:
            self.scheduler = LLMDynamicScheduler(
                max_parallem_sum=max_parallem_sum,
                connector=self.connector,
            )

        self.status = WorkerStatus.WAITING
        self.action = self.env.process(self.run())

    def workload(self):
        return self.scheduler.workload()

    def cpu_workload(self):
        return self.scheduler.cpu_workload()

    def step(self, requests: list[Request], latency: float):
        for request in requests:
            was_prefill = request.is_prefill
            request.step(self.env, latency, len(requests))
            if was_prefill:
                self.engine.record_request_completion("prefill", request)
            if request.is_done:
                self.engine.record_request_completion("decode", request)

    def estimate_transfer_latency(self, remote_id: int, num_blocks: int) -> float:
        """Seconds to move ``num_blocks`` KV blocks to another worker."""
        if num_blocks == 0 or self.id == remote_id:
            return 0.0
        bytes_ = int(num_blocks * self.cache_config.block_size * self.cache_config.size_per_token)
        return self.parallel_communicator.estimate_point_to_point(bytes_, remote_id)

    def run(self):
        while True:
            if self.status == WorkerStatus.RUNNING and not self.msg_queue.items:
                self.send_task(self, Task.STEP)
            msg = yield self.msg_queue.get()

            try:
                match msg.task:
                    case Task.ADD:
                        if msg.metadata is not None:
                            self.pending_connector_metadata = msg.metadata
                        self.scheduler.add_requests(msg.requests or [])
                        self.status = WorkerStatus.RUNNING

                    case Task.STEP:
                        if self.status != WorkerStatus.RUNNING:
                            raise SimulationStateError(
                                f"worker {self.id} received STEP while {self.status}"
                            )
                        self.connector.set_simulation_time(self.env.now)
                        schedule_output = self.scheduler.schedule_output()
                        connector_metadata = _merge_connector_metadata(
                            schedule_output.connector_metadata,
                            self.pending_connector_metadata,
                        )
                        self.pending_connector_metadata = None
                        self.connector.handle_preemptions(connector_metadata)
                        self.connector.bind_connector_metadata(connector_metadata)

                        load_latency = self.connector.start_load_kv()
                        if load_latency:
                            yield self.env.timeout(load_latency)

                        if schedule_output.preempted:
                            self.preempted_cnt += len(schedule_output.preempted)

                        running = schedule_output.running
                        if running:
                            use_af = (
                                schedule_output.phase != SchedulePhase.RECOMPUTE
                                and self.engine.uses_af_for(running)
                            )
                            if use_af:
                                iteration = self.engine.af_executor.submit(
                                    self.id,
                                    running,
                                )
                                yield self.env.process(
                                    self.engine.af_executor.run_iteration(iteration)
                                )
                                latency = iteration.makespan or 0.0
                                self.engine.record_af_iteration(iteration)
                            else:
                                latency = self.dynamic_batch(running)
                                if latency:
                                    yield from self.engine.run_complete_compute(
                                        self.id,
                                        latency,
                                    )
                            if schedule_output.phase == SchedulePhase.RECOMPUTE:
                                self.scheduler.finish_recompute(running, latency)
                            else:
                                self.step(running, latency)
                                running = self.scheduler.update(running)
                            self.connector.set_simulation_time(self.env.now)
                            save_latency = self.connector.wait_for_save()
                            if save_latency:
                                yield self.env.timeout(save_latency)
                            worker_meta = self.connector.build_connector_worker_meta()
                            self.scheduler.update_connector_output(worker_meta)

                            if running and self.role == "prefill":
                                self.engine.dispatch_prefill_to_decode(
                                    self, list(running)
                                )
                                self.scheduler.running = []
                                running = []

                        if not running and not self.scheduler.waiting:
                            self.status = WorkerStatus.WAITING

                    case Task.STOP:
                        self.status = WorkerStatus.STOPPED
                        break
            except Exception:
                logger.exception(
                    "worker %s failed while handling %s for requests=%s metadata=%s",
                    self.id,
                    msg.task,
                    _request_ids(msg.requests),
                    msg.metadata,
                )
                self.send_task(self.engine, Task.STOP)

    def dynamic_batch(self, requests: list[Request]) -> float:
        return self.latency_backend.estimate_step_latency(requests)


class LLMEngine(Worker):
    def __init__(
        self,
        env: simpy.Environment,
        block_size: int,
        batching: str,
        kv_transfer_config: KVTransferConfig,
        psla_config: PSLAConfig,
        cluster_config: ClusterConfig,
        hardware: HardwareContext,
        prefill_worker_pool_type: str,
        decode_worker_pool_type: str,
        max_parallem_sum: int,
        max_occupy_ratio: float = 1,
        parallel_config: ParallelConfig | None = None,
        latency_backend_type: str = "operator_table",
        latency_fallback: str = "table_first",
        operator_backend: str | None = None,
        decode_context_bucket: int = 128,
        random_seed: int = 0,
        debug_print: bool = False,
        af_compute_provider: AFComputeLatencyProvider | None = None,
        af_transfer_provider: AFTransferLatencyProvider | None = None,
        af_dtype: str | None = None,
    ):
        super().__init__(env, -1)

        prefill_worker_pool_type = prefill_worker_pool_type.lower()
        decode_worker_pool_type = decode_worker_pool_type.lower()

        pool_type_map = {
            "round_robin": RoundRobinWorkerPool,
            "least_gpu_memory": LeastGpuMemoryWorkerPool,
            "balanced_load": BalancedLoadWorkerPool,
        }
        self.hardware = hardware
        self.parallel_config = parallel_config or ParallelConfig.default()
        self.debug_printer = RequestCompletionDebugPrinter(debug_print)
        self.model_spec = hardware.model(psla_config.model).with_moe_override(
            psla_config.moe_config
        )
        self.moe_config = self.model_spec.moe
        # AF activations / KV default to the model's weight dtype so that the
        # AF stages query the same table rows as the step-level backend.
        af_dtype = af_dtype or self.model_spec.dtype
        self.expert_placement = build_expert_placement(
            self.moe_config,
            self.parallel_config,
            total_layers=self.model_spec.num_layers,
        )
        self.latency_backend_type = latency_backend_type
        self.latency_fallback = latency_fallback

        self.cluster_config = cluster_config
        worker_configs = cluster_config.workers(self.parallel_config)
        self.placement = build_topology_placement(cluster_config, worker_configs, hardware)
        self.af_runtime_plan: AFRuntimePlan | None = (
            compile_af_runtime_plan(
                cluster_config=cluster_config,
                worker_configs=worker_configs,
                model=self.model_spec,
                hardware=hardware,
                block_size=block_size,
                dtype=af_dtype,
            )
            if cluster_config.af.enabled
            else None
        )
        self.af_replica_block_managers: dict[tuple[str, int], ReplicaKVBlockManager] = (
            {}
        )
        self.af_resources = (
            AFResourceRegistry(env, cluster_config, worker_configs)
            if cluster_config.af.enabled
            else None
        )
        self.af_executor: AFExecutor | None = None
        self.af_iteration_summaries: list[dict[str, Any]] = []
        self.workers: list[LLMWorker | AFComputeWorker] = []
        self.request_workers: list[LLMWorker] = []
        for worker_id, worker_config in enumerate(worker_configs):
            physical_identity = (
                worker_config.physical_worker_ref.identity
                if worker_config.physical_worker_ref is not None
                else None
            )
            runtime_bindings = (
                self.af_runtime_plan.bindings_for_physical(physical_identity)
                if self.af_runtime_plan is not None and physical_identity is not None
                else ()
            )
            runtime_state = (
                self.af_runtime_plan.physical_state(physical_identity)
                if runtime_bindings and self.af_runtime_plan is not None
                else None
            )
            if self._is_request_owning_config(worker_config):
                has_af_scheduler_binding = any(
                    binding.is_scheduler_leader for binding in runtime_bindings
                )
                scheduler_replicas = (
                    self.af_runtime_plan.scheduler_replicas_for_physical(
                        physical_identity
                    )
                    if self.af_runtime_plan is not None
                    and physical_identity is not None
                    and has_af_scheduler_binding
                    else ()
                )
                replica = scheduler_replicas[0] if scheduler_replicas else None
                if replica is not None:
                    block_manager = next(
                        (
                            self.af_replica_block_managers[item.replica_key]
                            for item in scheduler_replicas
                            if item.replica_key in self.af_replica_block_managers
                        ),
                        None,
                    )
                    if block_manager is None:
                        block_manager = ReplicaKVBlockManager.from_plan(
                            self.af_runtime_plan,
                            replica,
                            aliases=scheduler_replicas,
                        )
                    for item in scheduler_replicas:
                        self.af_replica_block_managers[item.replica_key] = block_manager
                    cache_config_override = AFCacheConfig.from_plan(
                        self.af_runtime_plan, replica
                    )
                    worker_parallel_config = self.af_runtime_plan.domain_runtime(
                        replica.domain
                    ).parallel_config()
                else:
                    block_manager = None
                    cache_config_override = None
                    worker_parallel_config = self.parallel_config
                worker = LLMWorker(
                    env=env,
                    id=worker_id,
                    engine=self,
                    psla_config=psla_config,
                    worker_config=worker_config,
                    block_size=block_size,
                    batching=batching,
                    kv_transfer_config=kv_transfer_config,
                    hardware=hardware,
                    model_spec=self.model_spec,
                    placement=self.placement,
                    max_parallem_sum=max_parallem_sum,
                    max_occupy_ratio=max_occupy_ratio,
                    parallel_config=worker_parallel_config,
                    moe_config=self.moe_config,
                    expert_placement=self.expert_placement,
                    latency_backend_type=latency_backend_type,
                    latency_fallback=latency_fallback,
                    operator_backend=operator_backend,
                    decode_context_bucket=decode_context_bucket,
                    random_seed=random_seed,
                    runtime_role=self._runtime_role(worker_config),
                    cache_config_override=cache_config_override,
                    block_manager=block_manager,
                    runtime_bindings=runtime_bindings,
                    runtime_state=runtime_state,
                )
                self.request_workers.append(worker)
            else:
                worker = AFComputeWorker(
                    env,
                    worker_id,
                    worker_config,
                    runtime_bindings=runtime_bindings,
                    runtime_state=runtime_state,
                )
            self.workers.append(worker)
        for worker in self.request_workers:
            worker.parallel_communicator.workers = self.workers

        if cluster_config.af.enabled:
            af_compute_provider, af_transfer_provider = self._resolve_af_providers(
                af_compute_provider,
                af_transfer_provider,
                worker_configs,
                af_dtype=af_dtype,
                operator_backend=operator_backend,
                latency_fallback=latency_fallback,
                latency_backend_type=latency_backend_type,
            )
            self.af_compute_provider = af_compute_provider
            self.af_transfer_provider = af_transfer_provider
            self.af_executor = AFExecutor(
                env=env,
                cluster_config=cluster_config,
                worker_configs=worker_configs,
                compute_provider=af_compute_provider,
                transfer_provider=af_transfer_provider,
                model_id=psla_config.model,
                dtype=af_dtype,
                total_layers=self.model_spec.num_layers,
                hidden_size=self.model_spec.hidden_size,
                resources=self.af_resources,
                moe_config=self.moe_config,
                random_seed=random_seed,
                runtime_plan=self.af_runtime_plan,
            )

        self.prefill_workers = self._build_worker_pool(
            [
                worker
                for worker in self.request_workers
                if self._worker_supports_phase(worker, "prefill")
                if self._is_schedulable_worker(worker)
            ],
            pool_type_map[prefill_worker_pool_type],
        )
        self.decode_workers = self._build_worker_pool(
            [
                worker
                for worker in self.request_workers
                if self._worker_supports_phase(worker, "decode")
                if self._is_schedulable_worker(worker)
            ],
            pool_type_map[decode_worker_pool_type],
        )

        self.connector_stats = ConnectorStats()
        self.action = self.env.process(self.run())

    def _resolve_af_providers(
        self,
        compute_provider: AFComputeLatencyProvider | None,
        transfer_provider: AFTransferLatencyProvider | None,
        worker_configs: list[WorkerConfig],
        *,
        af_dtype: str,
        operator_backend: str | None,
        latency_fallback: str,
        latency_backend_type: str,
    ) -> tuple[AFComputeLatencyProvider | None, AFTransferLatencyProvider | None]:
        """Build operator-table providers unless the caller injected both."""
        af_config = self.cluster_config.af
        if af_config.latency_provider != "operator_table" or (
            compute_provider is not None and transfer_provider is not None
        ):
            return compute_provider, transfer_provider
        from TokenSim.af.operator_provider import build_operator_table_af_providers

        built_compute, built_transfer = build_operator_table_af_providers(
            hardware=self.hardware,
            model=self.model_spec,
            cluster_config=self.cluster_config,
            worker_configs=worker_configs,
            placement=self.placement,
            dtype=af_dtype,
            operator_backend=af_config.operator_backend or operator_backend,
            fallback=af_config.latency_fallback or latency_fallback,
            latency_backend_type=latency_backend_type,
        )
        return compute_provider or built_compute, transfer_provider or built_transfer

    def _is_request_owning_config(self, worker_config: WorkerConfig) -> bool:
        return any(
            self._config_supports_phase(worker_config, phase)
            for phase in ("prefill", "decode")
        )

    def _config_supports_phase(
        self,
        worker_config: WorkerConfig,
        phase: str,
    ) -> bool:
        if self.cluster_config.af.enabled and phase in self.cluster_config.af.phases:
            physical = worker_config.physical_worker_ref
            return physical is not None and physical.identity in (
                self.cluster_config.af_request_owner_identities(phase)
            )
        capability = "prefill" if phase == "prefill" else "decode_full"
        return capability in worker_config.capabilities

    def _runtime_role(self, worker_config: WorkerConfig) -> str:
        supports_prefill = self._config_supports_phase(worker_config, "prefill")
        supports_decode = self._config_supports_phase(worker_config, "decode")
        if supports_prefill and supports_decode:
            return "hybrid"
        if supports_prefill:
            return "prefill"
        if supports_decode:
            return "decode"
        raise ConfigurationError(
            f"request-owning worker group {worker_config.name!r} is not bound to "
            "Prefill or Decode"
        )

    def _worker_supports_phase(self, worker: LLMWorker, phase: str) -> bool:
        if self.cluster_config.af.enabled and phase in self.cluster_config.af.phases:
            physical = worker.physical_worker_ref
            return physical is not None and physical.identity in (
                self.cluster_config.af_request_owner_identities(phase)
            )
        capability = "prefill" if phase == "prefill" else "decode_full"
        return capability in worker.capabilities

    def uses_af_for(self, requests: list[Request]) -> bool:
        if self.af_executor is None:
            return False
        return phase_for_requests(requests) in self.cluster_config.af.phases

    def run_complete_compute(self, worker_id: int, latency: float):
        if self.af_resources is None:
            yield self.env.timeout(latency)
            return
        resource = self.af_resources.compute_slots[worker_id]
        with resource.request() as request:
            yield request
            yield self.env.timeout(latency)

    def record_af_iteration(self, iteration: AFIteration) -> None:
        if self.af_executor is None:
            return
        self.af_iteration_summaries.append(self.af_executor.trace.summarize(iteration))

    def af_runtime_summary(self) -> dict[str, Any] | None:
        if self.af_runtime_plan is None:
            return None
        return self.af_runtime_plan.summary(self.af_replica_block_managers)

    def start_debug_clock(self, wall_start: float | None = None) -> None:
        self.debug_printer.start(wall_start)

    def record_request_completion(self, event: str, request: Request) -> None:
        self.debug_printer.record(event, request, self.env.now)

    def validate_request_capacity(self, requests: list[Request]) -> None:
        role_pools = (
            (
                "prefill",
                self.prefill_workers.workers,
                lambda req: req.num_logical_token_blocks,
            ),
            (
                "decode",
                self.decode_workers.workers,
                lambda req: (req.prefill_len + req.decode_len + req.block_size - 1)
                // req.block_size,
            ),
        )
        for role, workers, required_blocks in role_pools:
            capacities = []
            for worker in workers:
                block_manager = getattr(worker.scheduler, "block_manager", None)
                if block_manager is None:
                    continue
                capacities.append(
                    block_manager.num_total_gpu_blocks - block_manager.watermark_blocks
                )
            if not capacities:
                continue
            available_blocks = min(capacities)
            for req in requests:
                needed = required_blocks(req)
                if needed > available_blocks:
                    raise ConfigurationError(
                        f"request {req.id} cannot fit on every eligible {role} worker: "
                        f"required_blocks={needed}, available_blocks={available_blocks}, "
                        f"prefill_len={req.prefill_len}, decode_len={req.decode_len}, "
                        f"block_size={req.block_size}"
                    )

    def _build_worker_pool(self, workers, pool_cls):
        dp_ranks = {getattr(worker, "dp_rank", 0) for worker in workers}
        if len(dp_ranks) > 1:
            return DataParallelWorkerPool(workers, pool_cls)
        return pool_cls(workers)

    def _is_schedulable_worker(self, worker: LLMWorker) -> bool:
        if self.af_runtime_plan is not None and worker.af_runtime_bindings:
            return any(
                binding.is_scheduler_leader for binding in worker.af_runtime_bindings
            )
        if self.parallel_config.world_size == 1:
            return True
        return worker.tp_rank == 0 and worker.pp_rank == 0

    def run(self):
        while True:
            msg = yield self.msg_queue.get()
            match msg.task:
                case Task.ADD:
                    self._send_to_prefill_worker(msg.requests)
                case Task.STOP:
                    self._broadcast_stop()
                    break

    def add_requests(self, requests: list[Request]):
        self.send_task(self, Task.ADD, requests)

    def add_requests_burst(self, requests: list[Request]):
        self._send_to_prefill_worker(requests)

    def _send_to_prefill_worker(self, requests: list[Request] | None) -> None:
        worker = self.prefill_workers.select_prefill_worker(
            requests[0] if requests else None
        )
        self.send_task(worker, Task.ADD, requests)

    def _broadcast_stop(self) -> None:
        for worker in self.request_workers:
            self.send_task(worker, Task.STOP)

    def dispatch_prefill_to_decode(
        self,
        sender: LLMWorker,
        requests: list[Request],
    ) -> None:
        if not requests:
            return
        num_requests = len(requests)
        num_per_worker = math.ceil(num_requests / len(self.decode_workers))
        for i in range(0, num_requests, num_per_worker):
            batch = requests[i : i + num_per_worker]
            worker = self.decode_workers.select_transfer_target(batch)
            request_block_counts = (
                sender.scheduler.block_manager.get_request_block_counts(batch)
            )
            num_blocks = sum(request_block_counts.values())
            connector = sender.connector
            uses_default_p2p = not hasattr(connector, "build_transfer_plan")
            if uses_default_p2p:
                connector = P2PConnector(sender.kv_transfer_config, sender.cache_config)
            latency = (
                sender.estimate_transfer_latency(worker.id, num_blocks)
                if uses_default_p2p or connector.name == "P2PConnector"
                else None
            )
            plan = connector.build_transfer_plan(
                requests=batch,
                source_worker_id=sender.id,
                target_worker_id=worker.id,
                kind="local" if sender is worker else "load",
                latency=latency,
                request_block_counts=request_block_counts,
            )
            sender_metadata = KVConnectorMetadata(
                connector_name=connector.name,
                saves=[plan],
            )
            receiver_metadata = KVConnectorMetadata(
                connector_name=connector.name,
                loads=[] if sender is worker else [plan],
            )
            sender.connector.bind_connector_metadata(sender_metadata)
            sender.connector.wait_for_save()
            for req in batch:
                sender.scheduler.block_manager.release_request_blocks(req)
                sender.connector.on_request_released(req)
                req.status = RequestStatus.WAITING_FOR_KV
            self.send_task(worker, Task.ADD, batch, receiver_metadata)
