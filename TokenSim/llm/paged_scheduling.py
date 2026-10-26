"""Paged-attention scheduling steps and waiting-request admission."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque

from TokenSim.errors import ConfigurationError
from TokenSim.llm.llm_request import Request, RequestStatus


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


class PagedSchedulingMixin:
    """Plan running requests and admissions for one scheduling step."""

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
        queue = deque(sorted(self.running, key=lambda req: req.arrival_time))
        self.running = []
        while queue:
            req = queue.popleft()
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
