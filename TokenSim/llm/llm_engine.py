from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional
import logging
import math
import time

import simpy

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
from TokenSim.latency import build_latency_backend
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

from TransformerRoofline import TransformerRoofline


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
        roofline: TransformerRoofline,
        max_parallem_sum: int,
        max_occupy_ratio: float = 1,
        parallel_config: ParallelConfig | None = None,
        moe_config: MoEModelConfig | None = None,
        expert_placement: ExpertPlacement | None = None,
        latency_backend_type: str = "roofline",
        random_seed: int = 0,
        wrapped_llmcompass_vars: tuple[Any, Any, Any] | None = None,
    ):
        super().__init__(env, id)
        self.roofline = roofline
        self.engine = engine

        self.role = worker_config.role
        self.hardware = worker_config.hardware
        self.network = worker_config.network
        self.nettype = worker_config.nettype
        self.model = psla_config.model
        self.parallel_config = parallel_config or ParallelConfig.default()
        self.moe_config = moe_config or psla_config.moe_config
        self.expert_placement = expert_placement
        self.rank_info = worker_config.rank_info or ParallelRankInfo()
        self.global_rank = self.rank_info.global_rank
        self.tp_rank = self.rank_info.tp_rank
        self.pp_rank = self.rank_info.pp_rank
        self.dp_rank = self.rank_info.dp_rank
        self.rank_in_dp_group = self.rank_info.rank_in_dp_group
        self.kv_cache_group_id = self.rank_info.kv_cache_group_id

        self.block_size = block_size
        try:
            self.cache_config = CacheConfig(
                self.block_size,
                self.hardware,
                self.model,
                self.roofline,
                parallel_config=self.parallel_config,
                rank_info=self.rank_info,
                moe_config=self.moe_config,
                expert_placement=self.expert_placement,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            self.cache_config = CacheConfig(
                self.block_size,
                self.hardware,
                self.model,
                self.roofline,
            )
        self.preempted_cnt = 0
        self.pending_connector_metadata: KVConnectorMetadata | None = None
        self.parallel_communicator = ParallelCommunicator(
            roofline=self.roofline,
            workers=self.engine.workers,
            worker_id=self.id,
            rank_info=self.rank_info,
            parallel_config=self.parallel_config,
            hardware=self.hardware,
        )

        self.latency_backend = build_latency_backend(
            backend_type=latency_backend_type,
            roofline=self.roofline,
            model=self.model,
            hardware=self.hardware,
            parallel_config=self.parallel_config,
            rank_info=self.rank_info,
            communicator=self.parallel_communicator,
            moe_config=self.moe_config,
            expert_placement=self.expert_placement,
            random_seed=random_seed + id,
            wrapped_llmcompass_vars=wrapped_llmcompass_vars,
        )
        self.owned_expert_ids = (
            self.expert_placement.experts_for_rank(self.rank_info)
            if self.expert_placement is not None
            else []
        )

        self.kv_transfer_config = kv_transfer_config.with_worker_defaults(
            worker_config.role,
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
        if num_blocks == 0:
            return 0.0
        hardwares = self.roofline.hardwares
        links = self.roofline.links

        if self.id == remote_id:
            return 0.0

        remote_worker = self.engine.workers[remote_id]
        if self.network == remote_worker.network:
            latency = max(
                links[hardwares[self.hardware].Nvlink].Latency,
                links[hardwares[remote_worker.hardware].Nvlink].Latency,
            )
            bandwidth = min(
                links[hardwares[self.hardware].Nvlink].UniBW,
                links[hardwares[remote_worker.hardware].Nvlink].UniBW,
            )
        else:
            latency = max(
                links[self.nettype].Latency, links[remote_worker.nettype].Latency
            )
            bandwidth = min(
                links[self.nettype].UniBW, links[remote_worker.nettype].UniBW
            )

        return (
            latency
            + num_blocks
            * self.cache_config.block_size
            * self.cache_config.size_per_token
            / _GB
            / bandwidth
        )

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
                            latency = self.dynamic_batch(running)
                            if latency:
                                yield self.env.timeout(latency)
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
        roofline: TransformerRoofline,
        prefill_worker_pool_type: str,
        decode_worker_pool_type: str,
        max_parallem_sum: int,
        max_occupy_ratio: float = 1,
        parallel_config: ParallelConfig | None = None,
        latency_backend_type: str = "roofline",
        wrapped_llmcompass_vars: tuple[Any, Any, Any] | None = None,
        random_seed: int = 0,
        debug_print: bool = False,
    ):
        super().__init__(env, -1)

        prefill_worker_pool_type = prefill_worker_pool_type.lower()
        decode_worker_pool_type = decode_worker_pool_type.lower()

        pool_type_map = {
            "round_robin": RoundRobinWorkerPool,
            "least_gpu_memory": LeastGpuMemoryWorkerPool,
            "balanced_load": BalancedLoadWorkerPool,
        }
        self.parallel_config = parallel_config or ParallelConfig.default()
        self.debug_printer = RequestCompletionDebugPrinter(debug_print)
        self.moe_config = psla_config.moe_config
        self.expert_placement = build_expert_placement(
            self.moe_config,
            self.parallel_config,
            total_layers=getattr(roofline.models[psla_config.model], "Nlayer", None),
        )

        worker_configs = cluster_config.workers(self.parallel_config)
        self.workers: list[LLMWorker] = []
        self.workers = [
            LLMWorker(
                env=env,
                id=id,
                engine=self,
                psla_config=psla_config,
                worker_config=worker_config,
                block_size=block_size,
                batching=batching,
                kv_transfer_config=kv_transfer_config,
                roofline=roofline,
                max_parallem_sum=max_parallem_sum,
                max_occupy_ratio=max_occupy_ratio,
                parallel_config=self.parallel_config,
                moe_config=self.moe_config,
                expert_placement=self.expert_placement,
                latency_backend_type=latency_backend_type,
                random_seed=random_seed,
                wrapped_llmcompass_vars=wrapped_llmcompass_vars,
            )
            for id, worker_config in enumerate(worker_configs)
        ]
        for worker in self.workers:
            worker.parallel_communicator.workers = self.workers

        self.prefill_workers = self._build_worker_pool(
            [
                worker
                for worker in self.workers
                if worker.role == "hybrid" or worker.role == "prefill"
                if self._is_schedulable_worker(worker)
            ],
            pool_type_map[prefill_worker_pool_type],
        )
        self.decode_workers = self._build_worker_pool(
            [
                worker
                for worker in self.workers
                if worker.role == "hybrid" or worker.role == "decode"
                if self._is_schedulable_worker(worker)
            ],
            pool_type_map[decode_worker_pool_type],
        )

        self.connector_stats = ConnectorStats()
        self.action = self.env.process(self.run())

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
        if self.parallel_config.data_parallel_size > 1:
            return DataParallelWorkerPool(workers, pool_cls)
        return pool_cls(workers)

    def _is_schedulable_worker(self, worker: LLMWorker) -> bool:
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
        for worker in self.workers:
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
