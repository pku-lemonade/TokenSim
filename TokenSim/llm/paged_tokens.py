"""Paged-attention token budgets, speculative queries and KV slot reservations."""

from __future__ import annotations

import math

from TokenSim.llm.llm_request import Request


class PagedTokenBudgetMixin:
    """Calculate token progress and reserve its required KV slots."""

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
