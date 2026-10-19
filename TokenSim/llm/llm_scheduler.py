from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
import logging
import math


from TokenSim.llm.llm_comm import KVConnectorMetadata, KVConnectorWorkerMetadata
from TokenSim.llm.llm_request import Request, RequestStatus
from TokenSim.block.block_manager import BlockManager
from TokenSim.config.config import CacheConfig, KVTransferConfig
from TokenSim.config.speculative_execution import validate_mtp_token_budget
from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer import NoopConnector
from TokenSim.llm.paged_scheduling import PagedSchedulingMixin
from TokenSim.llm.paged_tokens import PagedTokenBudgetMixin

logger = logging.getLogger(__name__)

# Token budget of one paged-attention step (vLLM V1 ``max_num_batched_tokens``
# for online serving; SGLang's ``chunked_prefill_size`` on large GPUs uses the
# same value). Prompts longer than the budget are prefilled over several steps.
DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192


class SchedulePhase(Enum):
    IDLE = auto()
    PREFILL = auto()
    DECODE = auto()
    RECOMPUTE = auto()
    # Chunked prefill: decode tokens and prefill/recompute chunks in one step.
    MIXED = auto()


@dataclass
class ScheduleOutput:
    running: list[Request]
    preempted: list[Request]
    # Requests admitted (or re-admitted after preemption) in this step. KV
    # connectors plan their loads and saves for these.
    scheduled: list[Request]
    connector_metadata: KVConnectorMetadata
    phase: SchedulePhase = SchedulePhase.IDLE


def step_phase(requests: list[Request]) -> SchedulePhase:
    """Phase label of a step: one of the pure phases or MIXED."""
    kinds = set()
    for req in requests:
        if req.needs_recompute:
            kinds.add(SchedulePhase.RECOMPUTE)
        elif req.is_prefill:
            kinds.add(SchedulePhase.PREFILL)
        else:
            kinds.add(SchedulePhase.DECODE)
    if not kinds:
        return SchedulePhase.IDLE
    return kinds.pop() if len(kinds) == 1 else SchedulePhase.MIXED


class LLMScheduler(ABC):
    def __init__(self, connector=None):
        self.running: list[Request] = []
        self.waiting: list[Request] = []
        self.connector = connector or NoopConnector(KVTransferConfig.default())

    def add_requests(self, requests: list[Request]):
        requests = [
            req for req in requests if not getattr(req, "cancel_requested", False)
        ]
        for req in requests:
            self.connector.on_new_request(req)
        self.waiting.extend(requests)

    @abstractmethod
    def schedule(self) -> tuple[list[Request], list[Request]]:
        raise NotImplementedError

    def schedule_output(self) -> ScheduleOutput:
        running, preempted = self.schedule()
        output = ScheduleOutput(
            running=running,
            preempted=preempted or [],
            scheduled=running if not preempted else [],
            connector_metadata=KVConnectorMetadata(),
            phase=step_phase(running),
        )
        output.connector_metadata = self.connector.build_connector_meta(output)
        return output

    def update(self, requests: list[Request]):
        # update block manager
        for req in requests:
            if req.is_done:
                self.free(req)
        # update scheduler running
        self.running = [req for req in self.running if not req.is_done]
        return self.running

    def update_connector_output(self, worker_output: KVConnectorWorkerMetadata) -> None:
        self.connector.update_connector_output(worker_output)

    def on_tokens_computed(self, req: Request) -> None:
        """Called after a step advanced ``req``; grouped caches publish pages here."""

    def free(self, req: Request):
        req.status = (
            RequestStatus.FINISHED_LENGTH_CAPPED
            if req.max_model_len is not None and req.context_len >= req.max_model_len
            else RequestStatus.FINISHED_STOPPED
        )

    def cancel_request(self, req: Request) -> bool:
        """Release at a worker boundary, after its current kernel/load finishes."""
        self.running = [r for r in self.running if r.id != req.id]
        self.waiting = [r for r in self.waiting if r.id != req.id]
        delayed, _ = self.connector.request_finished(req, [])
        self.connector.on_request_released(req)
        return not delayed

    def workload(self):
        return 0


class LLMDynamicScheduler(LLMScheduler):
    def __init__(self, max_num_seqs=64, connector=None):
        super().__init__(connector=connector)
        if max_num_seqs <= 0:
            raise ConfigurationError("max_num_seqs must be positive")
        self.max_num_seqs = max_num_seqs

    def schedule(self):
        available = max(0, self.max_num_seqs - len(self.running))
        self.running.extend(self.waiting[:available])
        del self.waiting[:available]
        return self.running, []

    def cpu_workload(self):
        return 0


class LLMStaticScheduler(LLMDynamicScheduler):
    def schedule(self):
        if not self.running:
            self.running.extend(self.waiting[: self.max_num_seqs])
            del self.waiting[: self.max_num_seqs]
        return self.running, []


class LLMPagedAttnScheduler(PagedSchedulingMixin, PagedTokenBudgetMixin, LLMScheduler):
    """Continuous batching with paged KV blocks and chunked prefill.

    Ported from vLLM's V1 scheduler: every step has a token budget
    (``max_num_batched_tokens``). Running requests are served first in arrival
    order, decodes taking one token each and in-progress prefills or recomputes
    taking the next chunk of their context; the remaining budget admits
    waiting requests, whose first chunk may also be partial. ``None`` disables
    the budget, so every prompt is prefilled in one step (the legacy
    behaviour).
    """

    def __init__(
        self,
        id: int,
        cache_config: CacheConfig,
        max_num_seqs=None,
        max_occupy_ratio: float = 1,
        connector=None,
        block_manager: BlockManager | None = None,
        max_num_batched_tokens: int | None = DEFAULT_MAX_NUM_BATCHED_TOKENS,
        long_prefill_token_threshold: int = 0,
        adaptive_long_prefill_threshold: bool = False,
        mtp_acceptance_length: float = 1.0,
        mtp_enabled: bool | None = None,
        mtp_seed: int = 0,
        max_parallem_sum: int | None = None,
    ):
        super().__init__(connector=connector)

        self.id = id
        self.cache_config = cache_config
        self.max_num_seqs = (
            max_parallem_sum if max_parallem_sum is not None else max_num_seqs
        )
        if max_num_batched_tokens is not None and max_num_batched_tokens < 1:
            raise ConfigurationError("max_num_batched_tokens must be at least 1")
        self.max_num_batched_tokens = max_num_batched_tokens
        if long_prefill_token_threshold < 0:
            raise ConfigurationError("long_prefill_token_threshold must be nonnegative")
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.adaptive_long_prefill_threshold = adaptive_long_prefill_threshold
        if not math.isfinite(mtp_acceptance_length) or mtp_acceptance_length < 1:
            raise ConfigurationError("mtp_acceptance_length must be finite and >= 1")
        self.mtp_acceptance_length = float(mtp_acceptance_length)
        self.mtp_seed = int(mtp_seed)
        # Grouped caches carry the speculative lookahead and the model's
        # context limit; uniform caches have neither.
        layout = getattr(self.cache_config, "hybrid_layout", None)
        self.speculative_tokens = layout.speculative_tokens if layout else 0
        self.max_model_len = layout.max_model_len if layout else None
        self.mtp_enabled = (
            bool(self.speculative_tokens and self.mtp_acceptance_length > 1)
            if mtp_enabled is None
            else bool(mtp_enabled)
        )
        if self.mtp_enabled and not self.speculative_tokens:
            raise ConfigurationError("enabled MTP scheduler requires positive cache K")
        validate_mtp_token_budget(
            self.max_num_batched_tokens,
            self.speculative_tokens if self.mtp_enabled else 0,
        )
        self.prefill_capacity_bound = False
        self.defer_prefills = False
        self.effective_long_prefill_threshold = 0
        self.stats = {
            "steps": 0,
            "deferred_running_prefills": 0,
            "deferred_waiting_prefills": 0,
            "capacity_bound_steps": 0,
            "scheduled_context_tokens": 0,
            "scheduled_context_query_tokens": 0,
            "scheduled_placeholder_queries": 0,
            "padded_context_admissions": 0,
            "scheduled_logits_rows": 0,
            "scheduled_decode_tokens": 0,
            "scheduled_decode_query_tokens": 0,
            "max_num_seqs_gate": 0,
            "full_input_gate": 0,
            "async_load_reservation_gate": 0,
            "max_inflight_prefill_reserved_slots": 0,
            "loaded_kv_capacity_gate": 0,
            "loaded_kv_resumptions": 0,
        }
        # Requests admitted by the most recent schedule() call.
        self.last_admitted: list[Request] = []

        if block_manager is not None:
            self.block_manager = block_manager
        elif layout is not None:
            from TokenSim.block.hybrid.manager import HybridKVBlockManager

            self.block_manager = HybridKVBlockManager(
                layout,
                self.cache_config.num_gpu_blocks,
                self.cache_config.model,
            )
        else:
            self.block_manager = BlockManager(
                block_size=self.cache_config.block_size,
                num_gpu_blocks=self.cache_config.num_gpu_blocks,
                num_cpu_blocks=self.cache_config.num_cpu_blocks,
                model=self.cache_config.model,
            )

        self.max_occupy_ratio = max_occupy_ratio
        self.loading = {}
        self.loaded_waiting = []
        self.inflight_prefills = {}
        set_releaser = getattr(self.connector, "set_block_releaser", None)
        if set_releaser is not None:
            set_releaser(self._release_delayed_blocks)

    def add_requests(self, requests):
        limit = self.max_model_len
        if limit is not None:
            for req in requests:
                if req.prefill_len >= limit:
                    raise ConfigurationError(
                        "prompt must leave room for an output token within max_model_len"
                    )
                req.max_model_len = limit
        super().add_requests(requests)

    def _admit(self, req: Request, *, external=None) -> None:
        """Allocate KV blocks for the whole context and record what is already cached."""
        local_plan = self.block_manager.kv_cache_manager.plan_reuse(req)
        if self.connector.async_offload:
            self.block_manager.allocate(req)
            req.gpu_cached_context_tokens = local_plan.hit_tokens
            req.gpu_cached_prefill_tokens = min(req.prefill_len, local_plan.hit_tokens)
            req.external_cached_prefill_tokens = 0
            req.external_cached_context_tokens = 0
            if external and external[1]:
                self.connector.begin_load(req, *external)
                req.record_context_admission(
                    local_plan.hit_tokens, req.external_cached_context_tokens
                )
                self.inflight_prefills[req.id] = req
                return
            if req.needs_recompute:
                req.recompute_tokens = max(0, req.context_len - local_plan.hit_tokens)
            req.start_context_build(local_plan.hit_tokens)
            req.record_context_admission(local_plan.hit_tokens, 0)
            req.status = RequestStatus.RUNNING
            return
        # Reuse the already-built prefix chain in external-cache connectors.
        req.input_cache_keys = list(local_plan.input_keys)
        num_external_tokens = self.connector.get_num_new_matched_tokens(
            req,
            local_plan.hit_tokens,
        )
        if req.needs_recompute:
            req.recompute_tokens = max(
                0,
                req.context_len - local_plan.hit_tokens - num_external_tokens,
            )
        self.block_manager.allocate(req)
        self.connector.update_state_after_alloc(
            req,
            self.block_manager.block_table.get_blocks(req.id),
            num_external_tokens,
        )
        if req.needs_recompute:
            to_compute = req.recompute_tokens
        elif req.is_prefill:
            to_compute = req.prefill_compute_len
        else:
            # KV arrived through a connector (prefill/decode disaggregation):
            # only the newest token is computed here.
            to_compute = 1
        req.start_context_build(req.context_len - to_compute)
        req.status = RequestStatus.RUNNING
        if self.block_manager.grouped_cache:
            req.record_context_admission(local_plan.hit_tokens, num_external_tokens)

    def _spec_decode_step_blocks(self):
        if not self.speculative_tokens:
            return 0
        # Current source uses num_spec_tokens == num_lookahead_tokens for MTP.
        width = 1 + 2 * self.speculative_tokens
        return (
            width + self.block_manager.block_size - 1
        ) // self.block_manager.block_size

    def _inflight_prefill_reserved_blocks(self):
        return sum(
            self.block_manager.remaining_admission_slots(req)
            + self._spec_decode_step_blocks()
            for req in self.inflight_prefills.values()
        )

    def _can_admit_external(self, req, local_plan, external):
        reserved = self._inflight_prefill_reserved_blocks()
        self.stats["max_inflight_prefill_reserved_slots"] = max(
            self.stats["max_inflight_prefill_reserved_slots"], reserved
        )
        args = dict(
            reserved_blocks=reserved + self._spec_decode_step_blocks(),
            has_scheduled_requests=bool(self.running),
        )
        allowed = self.block_manager.can_allocate_external(
            req, local_plan, *external, **args
        )
        if not allowed and self.block_manager.can_allocate_external(
            req, local_plan, *external, has_scheduled_requests=bool(self.running)
        ):
            self.stats["async_load_reservation_gate"] += 1
        return allowed

    def schedule_output(self, throttle_prefills=False) -> ScheduleOutput:
        scheduled, preempted = self.schedule(throttle_prefills)
        output = ScheduleOutput(
            running=scheduled,
            preempted=preempted,
            scheduled=list(self.last_admitted),
            connector_metadata=KVConnectorMetadata(
                preempted_request_ids=[req.id for req in preempted],
            ),
            phase=step_phase(scheduled),
        )
        output.connector_metadata = self.connector.build_connector_meta(output)
        if preempted and not output.connector_metadata.preempted_request_ids:
            output.connector_metadata.preempted_request_ids = [
                req.id for req in preempted
            ]
        return output

    def on_tokens_computed(self, req: Request) -> None:
        if self.block_manager.grouped_cache:
            self.block_manager.cache_computed(req)

    def update(self, requests: list[Request]):
        for req in requests:
            if not req.is_context_build:
                self.inflight_prefills.pop(req.id, None)
            if req.generation_idx > 0:
                self.block_manager.commit_input_cache(req)
        return super().update(requests)

    def update_connector_output(self, worker_output: KVConnectorWorkerMetadata) -> None:
        super().update_connector_output(worker_output)
        if worker_output.finished_sending:
            self.running = [
                req
                for req in self.running
                if not (
                    req.id in worker_output.finished_sending
                    and (
                        req.is_done
                        or req.status == RequestStatus.WAITING_FOR_CONNECTOR_FREE
                    )
                )
            ]

    def _is_occupy_below_usage(self):
        (_, used_blks, all_blks) = self.block_manager.get_gpu_status()
        return used_blks / all_blks < self.max_occupy_ratio

    def _preempt(self, req: Request) -> None:
        if req.status != RequestStatus.RUNNING:
            logger.debug("preempting request %s with status %s", req.id, req.status)
        self.inflight_prefills.pop(req.id, None)
        self.block_manager.release_request_blocks(req)
        req.prepare_recompute()
        req.status = RequestStatus.WAITING
        self.waiting.insert(0, req)

    def free(self, req: Request):
        self.inflight_prefills.pop(req.id, None)
        block_ids = [
            block.block_number
            for block in self.block_manager.block_table.get_blocks(req.id)
        ]
        delay_free, _ = self.connector.request_finished(req, block_ids)
        if delay_free:
            req.status = RequestStatus.WAITING_FOR_CONNECTOR_FREE
            return
        self.block_manager.free(req)
        req.status = (
            RequestStatus.FINISHED_LENGTH_CAPPED
            if req.max_model_len is not None and req.context_len >= req.max_model_len
            else RequestStatus.FINISHED_STOPPED
        )

    def cancel_request(self, req: Request) -> bool:
        self.running = [r for r in self.running if r.id != req.id]
        self.waiting = [r for r in self.waiting if r.id != req.id]
        self.loading.pop(req.id, None)
        self.inflight_prefills.pop(req.id, None)
        self.loaded_waiting = [r for r in self.loaded_waiting if r.id != req.id]
        blocks = self.block_manager.block_table.get_blocks(req.id)
        delayed, _ = self.connector.request_finished(
            req, [b.block_number for b in blocks]
        )
        if not delayed:
            # Cancellation must not publish uncomputed prompt/output cache
            # keys. Previously committed complete blocks keep their metadata.
            if blocks or self.block_manager.grouped_cache:
                self.block_manager.release_request_blocks(req)
            self.connector.on_request_released(req)
        return not self.block_manager.block_table.get_blocks(req.id)

    def workload(self):
        """Return GPU KV-cache utilization as a percentage."""
        free_blocks, used_blocks, total_blocks = self.block_manager.get_gpu_status()
        return (used_blocks / total_blocks) * 100 if total_blocks > 0 else 0

    def cpu_workload(self):
        """Return allocated CPU KV-cache capacity in GiB."""
        _GB = 1 << 30
        if self.connector.async_offload:
            return self.connector.store.used_bytes / _GB
        return (
            self.block_manager.cpu_allocator.get_num_allocated_blocks()
            * self.cache_config.block_size
            * self.cache_config.size_per_token
            / _GB
        )

    def _release_delayed_blocks(self, req: Request, block_ids: list[int]) -> None:
        self.block_manager.release_request_blocks(req)
        req.status = RequestStatus.FINISHED_STOPPED
