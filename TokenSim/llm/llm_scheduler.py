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

    def free(self, req: Request):
        req.status = RequestStatus.FINISHED_STOPPED

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
    def __init__(self, max_parallem_sum=200, connector=None):
        super().__init__(connector=connector)
        self.max_parallem_sum = max_parallem_sum

    def schedule(self) -> tuple[list[Request], list[Request]]:
        running = []
        while self.waiting:
            request = self.waiting.pop()
            running.append(request)
            if sum([req.prefill_len for req in running]) == self.max_parallem_sum:
                break
        if running:
            self.running.extend(running)
            return self.running, []

        self.running = sorted(self.running, key=lambda r: r.arrival_time)
        while self.running:
            running.append(self.running.pop())
        self.running.extend(running)
        return self.running, []

    def cpu_workload(self):
        return 0


class LLMStaticScheduler(LLMScheduler):
    def __init__(self, max_parallem_sum=0, connector=None):
        super().__init__(connector=connector)
        self.max_parallem_sum = max_parallem_sum

    def schedule(self) -> tuple[list[Request], list[Request]]:
        running = []
        while self.running:
            running.append(self.running.pop())
            if sum([req.prefill_len for req in running]) == self.max_parallem_sum:
                break
        if running:
            self.running.extend(running)
            return running, []
        while self.waiting:
            running.append(self.waiting.pop())
            if sum([req.prefill_len for req in running]) == self.max_parallem_sum:
                break
        self.running.extend(running)
        return running, []

    def cpu_workload(self):
        return 0


class LLMPrefillScheduler(LLMScheduler):
    def __init__(self, max_parallem_sum=0, connector=None):
        super().__init__(connector=connector)
        self.max_parallem_sum = max_parallem_sum

    def schedule(self) -> tuple[list[Request], list[Request]]:
        running = []
        if len(self.waiting) >= self.max_parallem_sum:
            while self.waiting:
                running.append(self.waiting.pop())
        else:
            if self.running:
                while self.running:
                    running.append(self.running.pop())
            else:
                while self.waiting:
                    running.append(self.waiting.pop())
        self.running.extend(running)
        return running, []


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
        max_parallem_sum=None,
        max_occupy_ratio: float = 1,
        connector=None,
        max_num_batched_tokens: int | None = DEFAULT_MAX_NUM_BATCHED_TOKENS,
        long_prefill_token_threshold: int = 0,
        adaptive_long_prefill_threshold: bool = False,
        mtp_acceptance_length: float = 1.0,
        mtp_seed: int = 0,
    ):
        super().__init__(connector=connector)

        self.id = id
        self.cache_config = cache_config
        self.max_parallem_sum = max_parallem_sum
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
        self.mtp_enabled = bool(
            getattr(getattr(self.cache_config, "hybrid_layout", None), "speculative_tokens", 0)
            and self.mtp_acceptance_length > 1
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

        if getattr(self.cache_config, "hybrid_layout", None) is not None:
            from TokenSim.block.hybrid_manager import HybridKVBlockManager

            self.block_manager = HybridKVBlockManager(
                self.cache_config.hybrid_layout,
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

    def schedule(self, throttle_prefills=False) -> tuple[list[Request], list[Request]]:
        """Build one step.

        Returns the requests computing tokens in this step (each with
        ``scheduled_tokens`` set) and the requests preempted while making room
        for decode slots. Running requests that got no budget stay in
        ``self.running`` untouched; admissions of this step are recorded in
        ``self.last_admitted``.
        """
        budget = self.max_num_batched_tokens
        scheduled: list[Request] = []
        preempted: list[Request] = []
        self.last_admitted = []
        self.stats["steps"] += 1
        self.defer_prefills = bool(
            throttle_prefills
            and not self.prefill_capacity_bound
            and any(not r.is_context_build for r in self.running)
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

        # 1. Running requests, oldest first: decode tokens and context chunks.
        # [TODO: xuechao] sort self.running according to some priority policy.
        # vLLM FCFS preserves running insertion order, including resumed
        # requests appended after existing running requests.
        queue = list(self.running)
        self.running = []
        while queue:
            req = queue.pop(0)
            if self.defer_prefills and req.is_context_build:
                self.stats["deferred_running_prefills"] += 1
                self.running.append(req)
                continue
            tokens = self._step_tokens(req, budget)
            if tokens == 0:
                # Out of budget: the request keeps its place for the next step.
                self.running.append(req)
                continue
            query_tokens = self._query_tokens(req, tokens)
            if not req.is_context_build and not self._reserve_decode_slot(
                req, queue, preempted, tokens, query_tokens
            ):
                continue
            if req.is_context_build and getattr(
                self.block_manager, "native_hybrid", False
            ):
                if not self._reserve_context_slots(req, tokens, queue, preempted):
                    continue
            req.scheduled_tokens = tokens
            req.scheduled_query_tokens = query_tokens
            budget = None if budget is None else budget - self._budget_cost(req, tokens)
            scheduled.append(req)
            self.running.append(req)

        # 2. Admissions while budget and KV blocks last (head-of-line order).
        # vLLM V1: skip admissions when preemptions happened this step —
        # admitting new work right after evicting running requests would
        # thrash the KV cache.
        if not preempted:
            while (self.loaded_waiting or self.waiting) and (
                budget is None or budget > 0
            ):
                resuming_loaded = bool(self.loaded_waiting)
                if not resuming_loaded and not self._is_occupy_below_usage():
                    break
                if (
                    self.max_parallem_sum is not None
                    and len(self.running) >= self.max_parallem_sum
                ):
                    self.stats["max_num_seqs_gate"] += 1
                    break
                request_queue = self.loaded_waiting if resuming_loaded else self.waiting
                req = request_queue[0]
                if resuming_loaded:
                    # A KV-holding waiting request retains its imported pages
                    # when query allocation cannot fit. Never free/reload that
                    # prefix via the running-victim path merely to admit it.
                    if self.defer_prefills and req.remaining_context_tokens > 1:
                        self.stats["deferred_waiting_prefills"] += 1
                        break
                    tokens = self._step_tokens(req, budget)
                    manager = self.block_manager
                    query_tokens = self._query_tokens(req, tokens)
                    lookahead = self._decode_lookahead(req)
                    watermark = manager.watermark_blocks if self.running else 0
                    if manager.remaining_admission_slots(
                        req
                    ) + watermark > manager.get_num_free_blocks() or not manager.can_ensure_slots(
                        req, tokens, lookahead=lookahead
                    ):
                        self.stats["loaded_kv_capacity_gate"] += 1
                        break
                    manager.ensure_slots(req, tokens, lookahead=lookahead)
                    request_queue.pop(0)
                    self.loading.pop(req.id, None)
                    req.status = RequestStatus.RUNNING
                    req.scheduled_tokens = tokens
                    req.scheduled_query_tokens = query_tokens
                    budget = None if budget is None else budget - self._budget_cost(req, tokens)
                    self.running.append(req)
                    scheduled.append(req)
                    self.stats["loaded_kv_resumptions"] += 1
                    continue
                native_async = getattr(self.connector, "native_async", False)
                external = None
                if native_async:
                    local_plan = self.block_manager.plan_reuse(req)
                    external = self.connector.probe(req, local_plan)
                if self.defer_prefills:
                    plan = self.block_manager.kv_cache_manager.plan_reuse(req)
                    # A full hit that needs only its last token may still run
                    # on a deferred step (scheduler.py waiting gate).
                    hit_tokens = external[0] if external else plan.hit_tokens
                    if req.context_len - hit_tokens > 1:
                        self.stats["deferred_waiting_prefills"] += 1
                        break
                can_allocate = (
                    self._can_admit_external(req, local_plan, external)
                    if external and external[1]
                    else (
                        self.block_manager.can_allocate(
                            req, has_scheduled_requests=bool(self.running)
                        )
                        if getattr(self.block_manager, "native_hybrid", False)
                        else self.block_manager.can_allocate(req)
                    )
                )
                if not can_allocate:
                    self.stats["full_input_gate"] += 1
                    break

                req = self.waiting.pop(0)
                self._admit(req, external=external)
                if req.status == RequestStatus.WAITING_FOR_KV:
                    self.stats.setdefault("async_load_admissions", 0)
                    self.stats["async_load_admissions"] += 1
                    continue
                tokens = self._step_tokens(req, budget)
                query_tokens = self._query_tokens(req, tokens)
                if getattr(self.block_manager, "native_hybrid", False):
                    if not self.block_manager.can_ensure_slots(
                        req, tokens, lookahead=self._decode_lookahead(req)
                    ):
                        self._preempt(req)
                        preempted.append(req)
                        break
                    self.block_manager.ensure_slots(
                        req, tokens, lookahead=self._decode_lookahead(req)
                    )
                if (
                    getattr(self.block_manager, "native_hybrid", False)
                    and req.num_computed_tokens + tokens < req.context_len
                ):
                    self.inflight_prefills[req.id] = req
                req.scheduled_tokens = tokens
                req.scheduled_query_tokens = query_tokens
                budget = None if budget is None else budget - self._budget_cost(req, tokens)
                scheduled.append(req)
                self.running.append(req)
                self.last_admitted.append(req)

        if not preempted and not self.defer_prefills:
            self.prefill_capacity_bound = bool(self.waiting or self.loaded_waiting)
        if self.prefill_capacity_bound:
            self.stats["capacity_bound_steps"] += 1
        for req in scheduled:
            key = (
                "scheduled_context_tokens"
                if req.is_context_build
                else "scheduled_decode_tokens"
            )
            self.stats[key] += req.scheduled_tokens
            if not req.is_context_build:
                self.stats["scheduled_decode_query_tokens"] += req.query_tokens
        return scheduled, preempted

    def _admit(self, req: Request, *, external=None) -> None:
        """Allocate KV blocks for the whole context and record what is already cached."""
        local_plan = self.block_manager.kv_cache_manager.plan_reuse(req)
        if getattr(self.connector, "native_async", False):
            self.block_manager.allocate(req)
            req.gpu_cached_prefill_tokens = local_plan.hit_tokens
            req.external_cached_prefill_tokens = 0
            if external and external[1]:
                self.connector.begin_load(req, *external)
                req.record_prefill_admission(
                    local_plan.hit_tokens, req.external_cached_prefill_tokens
                )
                self.inflight_prefills[req.id] = req
                return
            if req.needs_recompute:
                req.recompute_tokens = max(0, req.context_len - local_plan.hit_tokens)
            req.start_context_build(local_plan.hit_tokens)
            req.record_prefill_admission(local_plan.hit_tokens, 0)
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
        if getattr(self.block_manager, "native_hybrid", False):
            req.record_prefill_admission(local_plan.hit_tokens, num_external_tokens)

    def _spec_decode_step_blocks(self):
        layout = getattr(self.block_manager, "layout", None)
        if layout is None or not layout.speculative_tokens:
            return 0
        # Current source uses num_spec_tokens == num_lookahead_tokens for MTP.
        width = 1 + 2 * layout.speculative_tokens
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
            accepted = self._decode_output_tokens(req, remaining)
            cost = self._query_tokens(req, accepted)
            return accepted if budget is None or budget >= cost else 0
        if self.effective_long_prefill_threshold:
            remaining = min(remaining, self.effective_long_prefill_threshold)
        return remaining if budget is None else min(remaining, budget)

    def _decode_output_tokens(self, req: Request, remaining: int) -> int:
        if not self.mtp_enabled:
            return min(1, remaining)
        low = max(1, int(math.floor(self.mtp_acceptance_length)))
        high = max(low, int(math.ceil(self.mtp_acceptance_length)))
        high = min(high, 1 + self.cache_config.hybrid_layout.speculative_tokens)
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
            return accepted
        # Target verification still processes the complete speculative input
        # even when only one output token remains. Output progress is clipped
        # separately in _decode_output_tokens; partial queries would also
        # defeat the serving runner's uniform decode graph.
        return 1 + self.cache_config.hybrid_layout.speculative_tokens

    def _budget_cost(self, req: Request, accepted: int) -> int:
        return self._query_tokens(req, accepted) if not req.is_context_build else accepted

    def _decode_lookahead(self, req: Request) -> int:
        # The initial draft pass also runs over prefill chunks. Only async
        # KV loading skips a forward (and therefore its lookahead allocation).
        if not self.mtp_enabled:
            return 0
        return int(
            getattr(getattr(self.cache_config, "hybrid_layout", None), "speculative_tokens", 0)
        )

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
        if getattr(self.block_manager, "native_hybrid", False):
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
            self.block_manager.ensure_slots(
                req, query_tokens, lookahead=lookahead
            )
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

    def finish_recompute(self, requests: list[Request], latency: float) -> None:
        for req in requests:
            self.block_manager.commit_input_cache(req)
            req.finish_recompute(latency)

    def release_reserved_blocks(self, num_blocks: int | None = None):
        self.block_manager.release_reserved_blocks(num_blocks)

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
        req.status = RequestStatus.FINISHED_STOPPED

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
            if blocks or getattr(self.block_manager, "native_hybrid", False):
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
        if getattr(self.connector, "native_async", False):
            return self.connector.store.used_bytes / _GB
        capacity = (
            self.block_manager.cpu_allocator.get_num_allocated_blocks()
            * self.cache_config.block_size
            * self.cache_config.size_per_token
            / _GB
        )
        # return self.block_manager.cpu_allocator.get_num_allocated_blocks()

        return capacity

    def _release_delayed_blocks(self, req: Request, block_ids: list[int]) -> None:
        self.block_manager.release_request_blocks(req)
        req.status = RequestStatus.FINISHED_STOPPED
