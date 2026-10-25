"""Worker construction, scheduling, execution and KV transfer lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING

import logging
import simpy
from TokenSim.config.config import (
    CacheConfig,
    KVTransferConfig,
    ParallelConfig,
    ParallelRankInfo,
    WorkerConfig,
)
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.kv_transfer import KVConnectorFactory, KVConnectorMetadata
from TokenSim.hardware import HardwareContext
from TokenSim.hardware.topology import TopologyPlacement
from TokenSim.latency import build_latency_backend, effective_fallback
from TokenSim.latency.operator_table import collective_model_for
from TokenSim.operator_data.coverage import MissingShapeReport
from TokenSim.llm.llm_request import Request
from TokenSim.llm.llm_scheduler import (
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    LLMDynamicScheduler,
    LLMPagedAttnScheduler,
    LLMStaticScheduler,
)
from TokenSim.parallel import ParallelCommunicator
from TokenSim.moe import MoEModelConfig, ExpertPlacement
from TokenSim.config.model_config import ModelSpec
from .af_worker import bind_af_worker, run_worker_compute
from .messages import (
    Task,
    WorkerStatus,
    Worker,
    _request_ids,
    _merge_connector_metadata,
)

if TYPE_CHECKING:
    from TokenSim.llm.engine.core import LLMEngine

logger = logging.getLogger(__name__)


class LLMWorker(Worker):
    def __init__(
        self,
        env: simpy.Environment,
        id: int,
        engine: LLMEngine,
        worker_config: WorkerConfig,
        kv_transfer_config: KVTransferConfig,
        hardware: HardwareContext,
        model_spec: ModelSpec,
        placement: TopologyPlacement,
        parallel_config: ParallelConfig | None = None,
        moe_config: MoEModelConfig | None = None,
        expert_placement: ExpertPlacement | None = None,
        kv_cache_capacity_tokens_per_dp_rank: int | None = None,
        runtime_role: str | None = None,
        cache_config_override=None,
        block_manager=None,
        runtime_bindings=(),
        runtime_state=None,
    ):
        config = engine.config
        super().__init__(env, id)
        self.hardware_context = hardware
        self.engine = engine
        if config.prefill_schedule_interval < 1:
            raise ConfigurationError("prefill_schedule_interval must be positive")
        self.prefill_schedule_interval = config.prefill_schedule_interval

        self.role = runtime_role or worker_config.role
        self.hardware = worker_config.hardware
        self.device = hardware.device(worker_config.hardware)
        self.network = worker_config.network
        self.nettype = worker_config.nettype
        self.model = model_spec.model_id
        self.model_spec = model_spec
        self.parallel_config = parallel_config or ParallelConfig.default()
        self.moe_config = moe_config or model_spec.moe
        self.expert_placement = expert_placement
        bind_af_worker(self, worker_config, runtime_bindings, runtime_state)
        self.global_rank = self.rank_info.global_rank
        self.tp_rank = self.rank_info.tp_rank
        self.pp_rank = self.rank_info.pp_rank
        self.dp_rank = self.rank_info.dp_rank
        self.rank_in_dp_group = self.rank_info.rank_in_dp_group
        self.kv_cache_group_id = self.rank_info.kv_cache_group_id
        self.placement = placement
        self.device_index = placement.device_index(self.id)

        self.block_size = config.block_size
        from TokenSim.config.speculative_execution import resolve_mtp_tokens

        self.mtp_num_speculative_tokens = resolve_mtp_tokens(
            self.model_spec,
            config.mtp_acceptance_length,
            config.mtp_num_speculative_tokens,
        )
        if (
            self.model_spec.components is not None
            and config.max_num_batched_tokens is None
        ):
            raise ConfigurationError(
                "hybrid KV allocation requires a finite prefill token budget"
            )
        self.cache_config = cache_config_override or CacheConfig(
            self.block_size,
            self.device,
            self.model_spec,
            parallel_config=self.parallel_config,
            rank_info=self.rank_info,
            expert_placement=self.expert_placement,
            usable_memory_fraction=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens
            or DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=config.max_num_seqs,
            kv_cache_capacity_tokens_per_dp_rank=kv_cache_capacity_tokens_per_dp_rank,
            moe_execution=(
                config.moe_execution
                if config.moe_execution is not None
                else {"backend": "generic"}
            ),
            attention_backend=config.attention_backend,
            speculative_tokens=self.mtp_num_speculative_tokens,
        )
        self.preempted_cnt = 0
        self.pending_connector_metadata: KVConnectorMetadata | None = None

        backend_name = worker_config.operator_backend or config.operator_backend
        fallback = effective_fallback(
            config.latency_backend_type, config.latency_fallback
        )
        self.operator_package = (
            hardware.operator_package(self.device.device_id, backend_name)
            if fallback != "analytical_only"
            else None
        )
        # One checklist per worker: compute and communication misses together.
        self.missing_report = MissingShapeReport()
        self.collective_model = collective_model_for(
            self.device,
            self.placement,
            hardware.links,
            self.operator_package,
            fallback=fallback,
            missing_report=self.missing_report,
        )
        self.parallel_communicator = ParallelCommunicator(
            placement=self.placement,
            collective_model=self.collective_model,
            workers=self.engine.workers,
            worker_id=self.id,
            rank_info=self.rank_info,
            parallel_config=self.parallel_config,
            # Collectives move activations, not weights: bf16 all-reduce and
            # DeepEP rows, even for fp8/nvfp4 checkpoints.
            dtype=self.model_spec.activation_dtype,
        )

        self.latency_backend = build_latency_backend(
            backend_type=config.latency_backend_type,
            device=self.device,
            model=self.model_spec,
            package=self.operator_package,
            parallel_config=self.parallel_config,
            rank_info=self.rank_info,
            communicator=self.parallel_communicator,
            expert_placement=self.expert_placement,
            fallback=fallback,
            decode_context_bucket=config.decode_context_bucket,
            random_seed=config.random_seed + id,
            missing_report=self.missing_report,
            moe_execution=config.moe_execution,
            attention_backend=config.attention_backend,
            mhc_backend=config.mhc_backend,
            expected_decode_query_len=1
            + (
                self.cache_config.hybrid_layout.speculative_tokens
                if self.cache_config.hybrid_layout
                else 0
            ),
            max_num_seqs=config.max_num_seqs,
        )
        self.owned_expert_ids = (
            sorted(
                {
                    expert
                    for binding in runtime_bindings
                    for expert in binding.expert_ids
                }
            )
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
        if self.cache_config.hybrid_layout is not None:
            if config.batching != "paged-attn":
                raise ConfigurationError(
                    "native compressed attention hybrid cache requires paged-attn scheduling"
                )
        self.connector = KVConnectorFactory.create_connector(
            self.kv_transfer_config,
            self.cache_config,
        )

        if config.batching == "paged-attn":
            self.scheduler = LLMPagedAttnScheduler(
                self.id,
                cache_config=self.cache_config,
                max_num_seqs=config.max_num_seqs,
                max_occupy_ratio=(
                    config.max_occupy_ratio if self.role != "prefill" else 1
                ),
                connector=self.connector,
                block_manager=block_manager,
                max_num_batched_tokens=config.max_num_batched_tokens,
                long_prefill_token_threshold=config.long_prefill_token_threshold,
                adaptive_long_prefill_threshold=config.adaptive_long_prefill_threshold,
                mtp_acceptance_length=config.mtp_acceptance_length,
                mtp_enabled=self.mtp_num_speculative_tokens > 0,
                mtp_seed=config.mtp_seed + id,
                mtp_acceptance=config.mtp_acceptance,
            )
        elif config.batching == "static":
            self.scheduler = LLMStaticScheduler(
                max_num_seqs=config.max_num_seqs,
                connector=self.connector,
            )
        else:
            self.scheduler = LLMDynamicScheduler(
                max_num_seqs=config.max_num_seqs,
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
            if request.cancel_requested:
                continue
            was_prefill = request.is_prefill
            if (
                getattr(self.scheduler, "mtp_enabled", False)
                and not request.is_context_build
            ):
                self.scheduler.record_mtp_execution(request)
            if was_prefill and request.first_prefill_source is None:
                request.record_prefill_admission(
                    request.gpu_cached_prefill_tokens,
                    request.external_cached_prefill_tokens,
                )
            request.advance(
                self.env, latency, len(requests), self.engine.timing_recorder
            )
            self.scheduler.on_tokens_computed(request)
            if was_prefill and not request.is_prefill:
                self.engine.record_request_completion("prefill", request)
            if request.is_done:
                self.engine.record_request_completion("decode", request)

    def estimate_transfer_latency(self, remote_id: int, num_blocks: int) -> float:
        """Seconds to move ``num_blocks`` KV blocks to another worker."""
        if num_blocks == 0 or self.id == remote_id:
            return 0.0
        bytes_ = int(
            num_blocks * self.cache_config.block_size * self.cache_config.size_per_token
        )
        return self.parallel_communicator.estimate_point_to_point(bytes_, remote_id)

    def run(self):
        coordinator = getattr(self.engine, "dp_coordinator", None)
        if coordinator is not None and self.id in coordinator.worker_ids:
            yield from coordinator.consume_messages(self)
            return
        while True:
            if self.status == WorkerStatus.RUNNING and not self.msg_queue.items:
                self.send_task(self, Task.STEP)
            msg = yield self.msg_queue.get()

            try:
                match msg.task:
                    case Task.ADD:
                        if msg.metadata is not None:
                            self.pending_connector_metadata = _merge_connector_metadata(
                                msg.metadata, self.pending_connector_metadata
                            )
                        self.scheduler.add_requests(msg.requests or [])
                        self.status = WorkerStatus.RUNNING

                    case Task.STEP:
                        if self.status != WorkerStatus.RUNNING:
                            if (
                                self.status == WorkerStatus.WAITING
                                and not self.scheduler.running
                                and not self.scheduler.waiting
                            ):
                                # A cancelled request may leave a previously
                                # queued STEP behind; it cannot restart work.
                                continue
                            raise SimulationStateError(
                                f"worker {self.id} received STEP while {self.status}"
                            )
                        running, load_latency = self.prepare_step()
                        if not running and self.connector.async_offload:
                            # Transfers advance independently; a worker with
                            # only loading/capacity-blocked requests must yield.
                            self.status = WorkerStatus.WAITING
                            continue
                        if load_latency:
                            trace = getattr(self.engine, "trace_session", None)
                            if trace is not None:
                                trace.record_connector_load(self, load_latency)
                            yield self.env.timeout(load_latency)
                        running = [req for req in running if not req.cancel_requested]
                        if running:
                            latency = yield from run_worker_compute(self, running)
                            save_latency = self.complete_step(running, latency)
                            if save_latency:
                                yield self.env.timeout(save_latency)
                            self.finalize_connector_step()
                            running = self.scheduler.running

                            if self.role == "prefill":
                                # Hand over requests whose prompt is complete;
                                # partially prefilled ones keep their chunks here.
                                completed = [
                                    req for req in running if not req.is_context_build
                                ]
                                if completed:
                                    self.engine.dispatch_prefill_to_decode(
                                        self, completed
                                    )
                                    self.scheduler.running = [
                                        req
                                        for req in self.scheduler.running
                                        if req.is_context_build
                                    ]
                                    running = self.scheduler.running

                        else:
                            # Blocking loads have completed even when every
                            # scheduled request was cancelled before compute.
                            self.finalize_connector_step()

                        if (
                            not running
                            and not self.scheduler.waiting
                            and not getattr(self.scheduler, "loaded_waiting", [])
                        ):
                            self.status = WorkerStatus.WAITING

                    case Task.STOP:
                        self.status = WorkerStatus.STOPPED
                        break
                    case Task.CANCEL:
                        for req in msg.requests or []:
                            if self.scheduler.cancel_request(req):
                                req.acknowledge_cancel(self.env, self.id)
                            else:
                                self.env.process(self._wait_for_cancel_release(req))
                        if (
                            not self.scheduler.running
                            and not self.scheduler.waiting
                            and not getattr(self.scheduler, "loaded_waiting", [])
                        ):
                            self.status = WorkerStatus.WAITING
            except Exception as exc:
                logger.exception(
                    "worker %s failed while handling %s for requests=%s metadata=%s",
                    self.id,
                    msg.task,
                    _request_ids(msg.requests),
                    msg.metadata,
                )
                self.engine.record_worker_failure(self.id, exc)
                self.send_task(self.engine, Task.STOP)

    def dynamic_batch(self, requests: list[Request]) -> float:
        return self.latency_backend.estimate_step_latency(requests)

    def prepare_step(self):
        self.connector.set_simulation_time(self.env.now)
        if isinstance(self.scheduler, LLMPagedAttnScheduler):
            throttle = (
                self.parallel_config.data_parallel_size > 1
                and self.prefill_schedule_interval > 1
                and getattr(self, "dp_step_counter", 0) % self.prefill_schedule_interval
                != 0
            )
            output = self.scheduler.schedule_output(throttle_prefills=throttle)
        else:
            output = self.scheduler.schedule_output()
        metadata = _merge_connector_metadata(
            output.connector_metadata, self.pending_connector_metadata
        )
        self.pending_connector_metadata = None
        self.connector.handle_preemptions(metadata)
        self.connector.bind_connector_metadata(metadata)
        load_latency = self.connector.start_load_kv()
        self.preempted_cnt += len(output.preempted)
        return [r for r in output.running if not r.cancel_requested], load_latency

    def complete_step(self, running, latency):
        self.step(running, latency)
        self.scheduler.update(running)
        self.connector.set_simulation_time(self.env.now)
        save_latency = self.connector.wait_for_save()
        return save_latency

    def finalize_connector_step(self):
        metadata = self.connector.build_connector_worker_meta()
        self.scheduler.update_connector_output(metadata)

    def _wait_for_cancel_release(self, req: Request):
        manager = getattr(self.scheduler, "block_manager", None)
        while manager is not None and manager.block_table.get_blocks(req.id):
            self.connector.set_simulation_time(self.env.now)
            self.scheduler.update_connector_output(
                self.connector.build_connector_worker_meta()
            )
            if manager.block_table.get_blocks(req.id):
                yield self.env.timeout(0.001)
        req.acknowledge_cancel(self.env, self.id)
