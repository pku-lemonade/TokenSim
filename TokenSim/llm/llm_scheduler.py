from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import math


from TokenSim.llm.llm_comm import KVConnectorMetadata, KVConnectorWorkerMetadata
from TokenSim.llm.llm_request import Request, RequestStatus
from TokenSim.block.block_manager import BlockManager
from TokenSim.config.config import CacheConfig, KVTransferConfig
from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer import NoopConnector

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


@dataclass
class _Step:
    """Mutable state of one paged-attention scheduling step."""

    budget: int | None
    scheduled: list[Request] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)
    # A running request scheduled a pending prefill chunk this step.
    prefill_scheduled: bool = False
    # Running requests scheduled before admissions began.
    scheduled_running: tuple[Request, ...] = ()


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


class LLMPagedAttnScheduler(LLMScheduler):
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
        max_num_batched_tokens: int | None = DEFAULT_MAX_NUM_BATCHED_TOKENS,
        long_prefill_token_threshold: int = 0,
        adaptive_long_prefill_threshold: bool = False,
        mtp_acceptance_length: float = 1.0,
        mtp_enabled: bool | None = None,
        mtp_seed: int = 0,
    ):
        super().__init__(connector=connector)

        self.id = id
        self.cache_config = cache_config
        self.max_num_seqs = max_num_seqs
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
        layout = self.cache_config.hybrid_layout
        self.speculative_tokens = layout.speculative_tokens if layout else 0
        self.max_model_len = layout.max_model_len if layout else None
        self.mtp_enabled = (
            bool(self.speculative_tokens and self.mtp_acceptance_length > 1)
            if mtp_enabled is None
            else bool(mtp_enabled)
        )
        if self.mtp_enabled and not self.speculative_tokens:
            raise ConfigurationError("enabled MTP scheduler requires positive cache K")
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

        if layout is not None:
            from TokenSim.block.hybrid_manager import HybridKVBlockManager

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

    def schedule(self, throttle_prefills=False) -> tuple[list[Request], list[Request]]:
        """Build one step.

        Returns the requests computing tokens in this step (each with
        ``scheduled_tokens`` set) and the requests preempted while making room
        for decode slots. Running requests that got no budget stay in
        ``self.running`` untouched; admissions of this step are recorded in
        ``self.last_admitted``.
        """
        step = _Step(self.max_num_batched_tokens)
        self.last_admitted = []
        self.stats["steps"] += 1
        self._begin_step(throttle_prefills, step.budget)
        self._schedule_running(step)
        # vLLM V1 skips admissions in a step that preempted running requests:
        # admitting new work right after evicting would thrash the KV cache.
        step.scheduled_running = tuple(step.scheduled)
        if not step.preempted:
            self._schedule_waiting(step)
        if not step.preempted and not self.defer_prefills:
            self.prefill_capacity_bound = bool(self.waiting or self.loaded_waiting)
        if self.prefill_capacity_bound:
            self.stats["capacity_bound_steps"] += 1
        self._record_scheduled(step.scheduled)
        return step.scheduled, step.preempted

    def _begin_step(self, throttle_prefills, budget):
        """Prefill throttling and the effective long-prefill chunk limit."""
        self.defer_prefills = bool(
            throttle_prefills
            and not self.prefill_capacity_bound
            and any(not r.prefill_chunk_pending for r in self.running)
        )
        eligible = len(self.running) + len(self.waiting) + len(self.loading)
        self.effective_long_prefill_threshold = (
            self.long_prefill_token_threshold if eligible > 1 else 0
        )
        if (
            self.effective_long_prefill_threshold
            and self.adaptive_long_prefill_threshold
            and budget
        ):
            self.effective_long_prefill_threshold = max(
                self.effective_long_prefill_threshold, budget // eligible
            )

    def _schedule_running(self, step):
        """Running requests, oldest first: decode tokens and context chunks."""
        queue = sorted(self.running, key=lambda req: req.arrival_time)
        self.running = []
        while queue:
            req = queue.pop(0)
            if self.defer_prefills and req.prefill_chunk_pending:
                self.stats["deferred_running_prefills"] += 1
                self.running.append(req)
                continue
            tokens = self._step_tokens(req, step.budget)
            if tokens == 0:
                # Out of budget: the request keeps its place for the next step.
                self.running.append(req)
                continue
            query_tokens = self._query_tokens(req, tokens)
            if not req.is_context_build and not self._reserve_decode_slot(
                req, queue, step.preempted, tokens, query_tokens
            ):
                continue
            if req.is_context_build and self.block_manager.grouped_cache:
                if not self._reserve_context_slots(req, tokens, queue, step.preempted):
                    continue
            req.scheduled_tokens = tokens
            req.scheduled_query_tokens = query_tokens
            step.prefill_scheduled |= req.prefill_chunk_pending
            req.prefill_chunk_pending = (
                req.num_computed_tokens + query_tokens < req.context_len
            )
            if step.budget is not None:
                step.budget -= self._budget_cost(req, tokens)
            step.scheduled.append(req)
            self.running.append(req)

    def _schedule_waiting(self, step):
        """Admissions while budget and KV blocks last (head-of-line order).

        Requests whose external KV finished loading resume before new ones.
        """
        while (self.loaded_waiting or self.waiting) and (
            step.budget is None or step.budget > 0
        ):
            resuming = bool(self.loaded_waiting)
            if not resuming and not self._is_occupy_below_usage():
                break
            if self.max_num_seqs is not None and len(self.running) >= self.max_num_seqs:
                self.stats["max_num_seqs_gate"] += 1
                break
            admitted = (
                self._resume_loaded(self.loaded_waiting[0], step)
                if resuming
                else self._admit_waiting(self.waiting[0], step)
            )
            if not admitted:
                break

    def _resume_loaded(self, req, step):
        """Schedule a request whose external KV is resident; False stops admission.

        A KV-holding waiting request retains its imported pages when query
        allocation cannot fit. Never free and reload that prefix through the
        running-victim path merely to admit it.
        """
        if self.defer_prefills and req.remaining_context_tokens > 1:
            self.stats["deferred_waiting_prefills"] += 1
            return False
        tokens = self._step_tokens(req, step.budget)
        query_tokens = self._waiting_query_tokens(
            req,
            tokens,
            self._query_tokens(req, tokens),
            step.budget,
            step.scheduled_running,
            prefill_scheduled=step.prefill_scheduled,
        )
        if query_tokens == 0:
            return False
        manager = self.block_manager
        lookahead = self._decode_lookahead(req)
        watermark = manager.watermark_blocks if self.running else 0
        if manager.remaining_admission_slots(
            req
        ) + watermark > manager.get_num_free_blocks() or not manager.can_ensure_slots(
            req, query_tokens, lookahead=lookahead
        ):
            self.stats["loaded_kv_capacity_gate"] += 1
            return False
        manager.ensure_slots(req, query_tokens, lookahead=lookahead)
        self.loaded_waiting.pop(0)
        self.loading.pop(req.id, None)
        req.status = RequestStatus.RUNNING
        self._assign_waiting(req, tokens, query_tokens, step)
        self.stats["loaded_kv_resumptions"] += 1
        return True

    def _admit_waiting(self, req, step):
        """Admit the head of the waiting queue; False stops admission this step."""
        grouped = self.block_manager.grouped_cache
        external = local_plan = None
        if self.connector.async_offload:
            local_plan = self.block_manager.plan_reuse(req)
            external = self.connector.probe(req, local_plan)
        if self.defer_prefills:
            plan = self.block_manager.kv_cache_manager.plan_reuse(req)
            # A full hit that needs only its last token may still run
            # on a deferred step (scheduler.py waiting gate).
            hit_tokens = external[0] if external else plan.hit_tokens
            if req.context_len - hit_tokens > 1:
                self.stats["deferred_waiting_prefills"] += 1
                return False
        loads_external = bool(external and external[1])
        # Check padded compute budget before acquiring GPU prefix pages.
        if grouped and not loads_external:
            prefix = (
                local_plan.hit_tokens
                if local_plan is not None
                else self.block_manager.plan_reuse(req).hit_tokens
            )
            if (
                req.context_len - prefix == 1
                and self._waiting_query_tokens(
                    req,
                    1,
                    1,
                    step.budget,
                    step.scheduled_running,
                    computed=prefix,
                    prefill_scheduled=step.prefill_scheduled,
                )
                == 0
            ):
                return False
        if loads_external:
            can_allocate = self._can_admit_external(req, local_plan, external)
        elif grouped:
            can_allocate = self.block_manager.can_allocate(
                req, has_scheduled_requests=bool(self.running)
            )
        else:
            can_allocate = self.block_manager.can_allocate(req)
        if not can_allocate:
            self.stats["full_input_gate"] += 1
            return False

        self.waiting.pop(0)
        state = req.admission_state()
        self._admit(req, external=external)
        if req.status == RequestStatus.WAITING_FOR_KV:
            self.stats.setdefault("async_load_admissions", 0)
            self.stats["async_load_admissions"] += 1
            return True
        tokens = self._step_tokens(req, step.budget)
        query_tokens = self._waiting_query_tokens(
            req,
            tokens,
            self._query_tokens(req, tokens),
            step.budget,
            step.scheduled_running,
            prefill_scheduled=step.prefill_scheduled,
        )
        if query_tokens == 0:
            raise ConfigurationError("waiting query budget changed after admission")
        if grouped:
            lookahead = self._decode_lookahead(req)
            if not self.block_manager.can_ensure_slots(
                req, query_tokens, lookahead=lookahead
            ):
                # vLLM stops admitting when allocate_slots fails. The
                # admission has not run a kernel: release its tentative
                # references, undo its accounting and keep its place.
                self.block_manager.release_request_blocks(req)
                self.connector.discard_load(req)
                self.inflight_prefills.pop(req.id, None)
                req.restore_admission_state(state)
                self.waiting.insert(0, req)
                self.stats["full_input_gate"] += 1
                return False
            self.block_manager.ensure_slots(req, query_tokens, lookahead=lookahead)
            if req.num_computed_tokens + tokens < req.context_len:
                self.inflight_prefills[req.id] = req
        self._assign_waiting(req, tokens, query_tokens, step)
        self.last_admitted.append(req)
        return True

    def _assign_waiting(self, req, tokens, query_tokens, step):
        req.scheduled_tokens = tokens
        req.scheduled_query_tokens = query_tokens
        req.scheduled_placeholder_tokens = (
            max(0, query_tokens - tokens) if req.is_context_build else 0
        )
        req.prefill_chunk_pending = (
            req.num_computed_tokens + query_tokens < req.context_len
        )
        if step.budget is not None:
            step.budget -= query_tokens
        step.scheduled.append(req)
        self.running.append(req)

    def _record_scheduled(self, scheduled):
        stats = self.stats
        for req in scheduled:
            if req.is_context_build:
                stats["scheduled_context_tokens"] += req.scheduled_tokens
                stats["scheduled_context_query_tokens"] += req.query_tokens
            else:
                stats["scheduled_decode_tokens"] += req.scheduled_tokens
                stats["scheduled_decode_query_tokens"] += req.query_tokens
            stats["scheduled_placeholder_queries"] += req.scheduled_placeholder_tokens
            stats["padded_context_admissions"] += int(
                req.scheduled_placeholder_tokens > 0
            )
            stats["scheduled_logits_rows"] += req.logits_tokens

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

    def _step_tokens(self, req: Request, budget: int | None) -> int:
        """Tokens ``req`` computes this step: its remaining context, capped by the budget."""
        remaining = max(1, req.remaining_context_tokens)
        if not req.is_context_build:
            # ``scheduled_tokens`` is accepted output progress.  The target
            # forward may verify up to 1+K positions, accounted separately in
            # the token budget and latency query width.
            remaining = max(1, req.decode_len - req.generation_idx)
            limit = self.max_model_len
            if limit is not None:
                remaining = min(remaining, max(0, limit - req.context_len))
            if remaining == 0:
                return 0
            accepted = min(
                self._decode_output_tokens(req, remaining), self._query_room(req)
            )
            cost = self._query_tokens(req, accepted)
            return accepted if budget is None or budget >= cost else 0
        if self.effective_long_prefill_threshold:
            remaining = min(remaining, self.effective_long_prefill_threshold)
        remaining = min(remaining, self._query_room(req))
        return remaining if budget is None else min(remaining, budget)

    def _decode_output_tokens(self, req: Request, remaining: int) -> int:
        if not self.mtp_enabled:
            return min(1, remaining)
        low = max(1, int(math.floor(self.mtp_acceptance_length)))
        high = max(low, int(math.ceil(self.mtp_acceptance_length)))
        high = min(high, 1 + self.speculative_tokens)
        if high == low:
            return min(low, remaining)
        state = (
            (int(req.id) * 1103515245)
            + (int(req.generation_idx) * 12345)
            # Mix the simulation seed before reducing to a uint32.  Adding
            # seed 1 or 2 directly changes the fraction by only 1e-10, which
            # can leave two supposedly independent runs bit-identical.  The
            # multiplicative constant is the standard golden-ratio hash step
            # and keeps request/generation identity deterministic.
            + (int(self.mtp_seed) * 0x9E3779B9)
        ) & 0xFFFFFFFF
        fraction = state / 4294967296.0
        accepted = high if fraction < self.mtp_acceptance_length - low else low
        return min(accepted, remaining)

    def _query_tokens(self, req: Request, accepted: int) -> int:
        if req.is_context_build or not self.mtp_enabled:
            return min(accepted, self._query_room(req))
        # Target verification still processes the complete speculative input
        # even when only one output token remains. Output progress is clipped
        # separately in _decode_output_tokens; partial queries would also
        # defeat the serving runner's uniform decode graph.
        return min(1 + self.speculative_tokens, self._query_room(req))

    def _query_room(self, req):
        limit = self.max_model_len
        return (
            max(0, limit - req.num_computed_tokens - 1)
            if limit is not None
            else float("inf")
        )

    def _waiting_query_tokens(
        self,
        req,
        real_tokens,
        queries,
        budget,
        scheduled,
        *,
        computed=None,
        prefill_scheduled=False,
    ):
        if not self.mtp_enabled or not req.is_context_build or real_tokens != 1:
            return queries
        prefix = req.num_computed_tokens if computed is None else computed
        width = 1 + self.speculative_tokens
        if (
            req.context_len - prefix != 1
            or prefill_scheduled
            or not (scheduled or prefix > 0)
        ):
            return queries
        limit = self.max_model_len
        if limit is not None and prefix + width + 1 > limit:
            return queries
        return width if budget is None or budget >= width else 0

    def _budget_cost(self, req: Request, accepted: int) -> int:
        return (
            self._query_tokens(req, accepted) if not req.is_context_build else accepted
        )

    def _decode_lookahead(self, req: Request) -> int:
        # The initial draft pass also runs over prefill chunks. Only async
        # KV loading skips a forward (and therefore its lookahead allocation).
        return self.speculative_tokens if self.mtp_enabled else 0

    def _reserve_context_slots(self, req, tokens, queue, preempted):
        # The same FCFS youngest-victim rule applies to prefill/recompute
        # allocation, not only decode growth. Never publish canceled or
        # uncomputed blocks while releasing a victim.
        lookahead = self._decode_lookahead(req)
        while not self.block_manager.can_ensure_slots(req, tokens, lookahead=lookahead):
            victim = queue.pop() if queue else req
            self._preempt(victim)
            preempted.append(victim)
            if victim is req:
                return False
        self.block_manager.ensure_slots(req, tokens, lookahead=lookahead)
        return True

    def _reserve_decode_slot(
        self,
        req: Request,
        queue: list[Request],
        preempted: list[Request],
        accepted_tokens: int,
        query_tokens: int,
    ) -> bool:
        """Find a KV slot for the token ``req`` computes this step.

        Preempts the youngest not-yet-scheduled running requests until a block
        is free; preempts ``req`` itself when nobody else is left. Returns
        whether ``req`` may run.
        """
        if self.block_manager.grouped_cache:
            lookahead = self._decode_lookahead(req)
            while not self.block_manager.can_ensure_slots(
                req, query_tokens, lookahead=lookahead
            ):
                if queue:
                    victim = queue.pop(-1)
                    self._preempt(victim)
                    preempted.append(victim)
                else:
                    self._preempt(req)
                    preempted.append(req)
                    return False
            self.block_manager.ensure_slots(req, query_tokens, lookahead=lookahead)
            return True
        while True:
            if self.block_manager.can_append_slot(req):
                self.block_manager.append_slot(req)
                return True
            if queue:
                victim = queue.pop(-1)
                self._preempt(victim)
                preempted.append(victim)
            else:
                self._preempt(req)
                preempted.append(req)
                return False

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
