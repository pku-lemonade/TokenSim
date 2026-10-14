"""Simulation engine, worker pools and request dispatch."""

from __future__ import annotations

from TokenSim.kv_transfer.connectors.base import flatten_connectors
from TokenSim.config.simulation import SimulationConfig
import logging
import math
import simpy
from TokenSim.config.config import ClusterConfig, KVTransferConfig, ParallelConfig
from TokenSim.config.workload_config import WorkloadConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.kv_transfer import ConnectorStats, KVConnectorMetadata, P2PConnector
from TokenSim.hardware import HardwareContext
from TokenSim.hardware.placement import build_topology_placement
from TokenSim.llm.llm_request import Request, RequestStatus
from TokenSim.placement import (
    BalancedLoadWorkerPool,
    DataParallelWorkerPool,
    LeastGpuMemoryWorkerPool,
    RoundRobinWorkerPool,
)
from TokenSim.moe import build_expert_placement
from TokenSim.config.model_config import ModelSpec
from .messages import Task, Worker, RequestCompletionDebugPrinter
from .worker import LLMWorker
from .af import AFEngineMixin

logger = logging.getLogger(__name__)


class LLMEngine(AFEngineMixin, Worker):
    def __init__(
        self,
        env: simpy.Environment,
        kv_transfer_config: KVTransferConfig,
        model_spec: ModelSpec,
        cluster_config: ClusterConfig,
        hardware: HardwareContext,
        parallel_config: ParallelConfig | None = None,
        config: SimulationConfig | None = None,
        workload_config: WorkloadConfig | None = None,
        af_compute_provider=None,
        af_transfer_provider=None,
        af_dtype: str | None = None,
    ):
        config = config or SimulationConfig()
        self.config = config
        self.workload_config = workload_config or WorkloadConfig()
        from TokenSim.timing import TimingRecorder

        self.timing_recorder = TimingRecorder()
        super().__init__(env, -1)

        pool_type_map = {
            "round_robin": RoundRobinWorkerPool,
            "least_gpu_memory": LeastGpuMemoryWorkerPool,
            "balanced_load": BalancedLoadWorkerPool,
        }
        self.hardware = hardware
        self.parallel_config = parallel_config or ParallelConfig.default()
        self.debug_printer = RequestCompletionDebugPrinter(config.debug_print)
        self.model_spec = model_spec
        self.moe_config = self.model_spec.moe
        self.expert_placement = build_expert_placement(
            self.moe_config,
            self.parallel_config,
            total_layers=self.model_spec.num_layers,
        )
        self.latency_backend_type = config.latency_backend_type
        self.latency_fallback = config.latency_fallback
        self.max_num_batched_tokens = config.max_num_batched_tokens

        worker_configs = cluster_config.workers(self.parallel_config)
        self.placement = build_topology_placement(
            cluster_config, worker_configs, hardware
        )
        self.cluster_config = cluster_config
        self._initialize_workers(
            kv_transfer_config,
            worker_configs,
            af_compute_provider,
            af_transfer_provider,
            af_dtype,
        )

        self.prefill_workers = self._build_worker_pool(
            [
                worker
                for worker in self.request_workers
                if self._worker_supports_phase(worker, "prefill")
                if self._is_schedulable_worker(worker)
            ],
            pool_type_map[config.prefill_worker_pool_type],
        )
        self.decode_workers = self._build_worker_pool(
            [
                worker
                for worker in self.request_workers
                if self._worker_supports_phase(worker, "decode")
                if self._is_schedulable_worker(worker)
            ],
            pool_type_map[config.decode_worker_pool_type],
        )

        self.connector_stats = ConnectorStats()
        # First exception raised by a worker step (e.g. MissingOperatorDataError
        # under table_only); the simulation stops and the caller re-raises it.
        self.failure: tuple[int, Exception] | None = None
        self.failure_event = self.env.event()
        self.router = None
        if config.router_policy == "consistent_hash":
            from TokenSim.placement.consistent_hash import ConsistentHashRouter

            self.router = ConsistentHashRouter(
                self.prefill_workers.workers, backend_url=config.router_backend_url
            )
        elif config.router_policy == "fixed_mapping":
            from TokenSim.placement.consistent_hash import FixedWorkerRouter

            self.router = FixedWorkerRouter(
                self.prefill_workers.workers,
                config.router_mapping or {},
                backend_url=config.router_backend_url,
            )
        elif config.router_policy != "none":
            raise ConfigurationError(
                f"unsupported router policy: {config.router_policy}"
            )
        self.dp_coordinator = None
        if config.dp_lockstep and self.parallel_config.data_parallel_size > 1:
            if self.parallel_config.pipeline_parallel_size != 1:
                raise ConfigurationError(
                    "DP lockstep pipeline stages are not yet supported"
                )
            from TokenSim.llm.dp_coordinator import DPStepCoordinator

            self.dp_coordinator = DPStepCoordinator(
                env,
                self,
                self.prefill_workers.workers,
                sync_interval=config.dp_sync_interval,
            )
        self.action = self.env.process(self.run())

    def record_worker_failure(self, worker_id: int, error: Exception) -> None:
        if self.failure is None:
            self.failure = (worker_id, error)
            if not self.failure_event.triggered:
                self.failure_event.succeed(self.failure)

    def raise_if_failed(self) -> None:
        """Re-raise the worker exception that stopped the simulation, if any."""
        if self.failure is not None:
            worker_id, error = self.failure
            raise SimulationStateError(f"worker {worker_id} failed: {error}") from error

    def start_debug_clock(self, wall_start: float | None = None) -> None:
        self.debug_printer.start(wall_start)

    def record_request_completion(self, event: str, request: Request) -> None:
        self.debug_printer.record(event, request, self.env.now)

    def reset_profile_stats(self) -> None:
        """Reset counters after warmup while retaining caches and routing affinity."""
        from TokenSim.kv_transfer.connectors.metadata import ConnectorStats
        from TokenSim.moe.stats import MoEStats
        from TokenSim.mooncake.metrics import MooncakeStats
        from TokenSim.parallel import ParallelStats

        self.timing_recorder.reset()
        self.connector_stats = ConnectorStats()
        if self.router is not None:
            self.router.reset_stats()
        if self.dp_coordinator is not None:
            self.dp_coordinator.reset_stats()
        seen_mooncake_stats: set[int] = set()
        for worker in self.request_workers:
            worker.preempted_cnt = 0
            stats = getattr(worker.scheduler, "stats", None)
            if stats is not None:
                for key in stats:
                    stats[key] = 0
            worker.parallel_communicator.stats = ParallelStats()
            latency_backend = getattr(worker, "latency_backend", None)
            if hasattr(latency_backend, "reset_profile_stats"):
                latency_backend.reset_profile_stats()
            if hasattr(latency_backend, "moe_stats"):
                previous = latency_backend.moe_stats
                latency_backend.moe_stats = MoEStats(
                    expert_placement=previous.expert_placement,
                    effective_moe_config=previous.effective_moe_config,
                )
            for connector in flatten_connectors(worker.connector):
                old_stats = getattr(connector, "stats", None)
                if old_stats is not None:
                    connector.stats = ConnectorStats(
                        producer_count=old_stats.producer_count,
                        consumer_count=old_stats.consumer_count,
                        both_count=old_stats.both_count,
                    )
                mooncake_stats = getattr(connector, "mooncake_stats", None)
                if mooncake_stats is None or id(mooncake_stats) in seen_mooncake_stats:
                    continue
                seen_mooncake_stats.add(id(mooncake_stats))
                preserved_tiers = list(mooncake_stats.offload_tiers)
                preserved_profiles = list(mooncake_stats.offload_profiles)
                fresh_stats = MooncakeStats(
                    offload_tiers=preserved_tiers,
                    offload_profiles=preserved_profiles,
                    pending_async_jobs=mooncake_stats.pending_async_jobs,
                    storage_metrics=mooncake_stats.storage_metrics.reset(),
                )
                for field_name in fresh_stats.__dataclass_fields__:
                    setattr(
                        mooncake_stats, field_name, getattr(fresh_stats, field_name)
                    )
        for pool_name in ("prefill_workers", "decode_workers"):
            pool = getattr(self, pool_name, None)
            if pool is None:
                continue
            pool.placement_decision_count = 0
            if hasattr(pool, "dp_placement_counts"):
                pool.dp_placement_counts = {
                    rank: 0 for rank in pool.dp_placement_counts
                }

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
                layouts = [
                    getattr(
                        getattr(worker, "cache_config", None), "hybrid_layout", None
                    )
                    for worker in workers
                ]
                needed = max(
                    (
                        layout.max_request_slots(req.prefill_len + req.decode_len)
                        if layout is not None
                        else required_blocks(req)
                    )
                    for layout in layouts
                )

                if needed > available_blocks:
                    raise ConfigurationError(
                        f"request {req.id} cannot fit on every eligible {role} worker: "
                        f"required_blocks={needed}, available_blocks={available_blocks}, "
                        f"prefill_len={req.prefill_len}, decode_len={req.decode_len}, "
                        f"block_size={req.block_size}"
                    )

    def _build_worker_pool(self, workers, pool_cls):
        if len({getattr(worker, "dp_rank", 0) for worker in workers}) > 1:
            return DataParallelWorkerPool(
                workers,
                pool_cls,
                sticky_sessions=getattr(
                    getattr(self, "config", None), "sticky_sessions", False
                ),
            )
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
        requests = [req for req in requests or [] if not req.cancel_requested]
        if not requests:
            return
        if self.router is not None:
            groups = {}
            for request in requests:
                worker = self.router.select(request)
                self.prefill_workers.placement_decision_count += 1
                if hasattr(self.prefill_workers, "dp_placement_counts"):
                    self.prefill_workers._record(worker)
                groups.setdefault(worker.id, (worker, []))[1].append(request)
            for worker, batch in groups.values():
                self.send_task(worker, Task.ADD, batch)
            return
        worker = self.prefill_workers.select_prefill_worker(
            requests[0] if requests else None
        )
        self.send_task(worker, Task.ADD, requests)

    def cancel_requests(self, requests: list[Request]) -> list[simpy.Event]:
        events = []
        for req in requests:
            already_requested = req.cancel_requested
            event = req.request_cancel(self.env)
            events.append(event)
            if req.is_done or already_requested:
                continue
            if not req.cancel_pending_worker_ids:
                req.acknowledge_cancel(self.env)
            for worker_id in sorted(req.cancel_pending_worker_ids):
                self.send_task(self.workers[worker_id], Task.CANCEL, [req])
        return events

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
            manager = getattr(sender.scheduler, "block_manager", None)
            request_block_counts = (
                manager.get_request_block_counts(batch)
                if manager is not None
                else {
                    req.id: math.ceil(req.num_computed_tokens / sender.block_size)
                    for req in batch
                }
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
                if manager is not None:
                    manager.release_request_blocks(req)
                sender.connector.on_request_released(req)
                req.status = RequestStatus.WAITING_FOR_KV
            self.send_task(worker, Task.ADD, batch, receiver_metadata)
