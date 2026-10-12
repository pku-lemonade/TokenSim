"""Shared slot allocation, request admission, window growth and conservation."""

from __future__ import annotations

from collections import OrderedDict

from TokenSim.errors import OutOfBlocksError, SimulationStateError
from .kv_layout import cdiv


class SlotAllocatorMixin:
    """Allocate and release the manager's group-aware request slots."""

    def _note_usage(self):
        used = self.num_total_gpu_blocks - self.get_num_free_blocks()
        if used > self.peak_used_slots:
            self.peak_used_slots = used

    def _acquire(self, block):
        if block.ref_count == 0:
            slot = block.block_number
            if slot in self._free_cached:
                del self._free_cached[slot]
            else:
                self._free_uncached.remove(block)
        block.ref_count += 1
        self._note_usage()

    def _allocate_many(self, count):
        """Pop ``count`` slots from the front of the free queue, in order."""
        if count <= 0:
            return []
        stack = self._free_uncached
        if count > len(stack) + len(self._free_cached):
            raise OutOfBlocksError("native hybrid shared pool exhausted")
        take = min(count, len(stack))
        blocks = stack[-take:][::-1] if take else []
        if take:
            del stack[-take:]
        for _ in range(count - take):
            slot, _ = self._free_cached.popitem(last=False)
            blocks.append(self.blocks[slot])
        for block in blocks:
            if block.ref_count:
                raise SimulationStateError("free hybrid slot has live references")
            if block.block_hash is not None or block.cached:
                self._evict(block)
            block.ref_count = 1
        self.allocation_count += count
        self._note_usage()
        return blocks

    def _allocate(self):
        return self._allocate_many(1)[0]

    def _release(self, block):
        self._release_many((block,))

    def _release_many(self, ordered_blocks):
        uncached = []
        cached = self._free_cached
        null = self.null_block
        count = 0
        for block in ordered_blocks:
            if block is null or block.ref_count <= 0:
                raise SimulationStateError("invalid hybrid slot release")
            block.ref_count -= 1
            count += 1
            if block.ref_count == 0:
                if block.block_hash is not None:
                    # Cached slots join the LRU tail for eviction.
                    cached[block.block_number] = None
                else:
                    uncached.append(block)
        self.release_count += count
        # vLLM BlockPool.free_blocks prepends uncached slots for LIFO locality
        # and keeps their input order at the front of the queue.
        if uncached:
            uncached.reverse()
            self._free_uncached.extend(uncached)

    def can_allocate(self, req, *, has_scheduled_requests=False):
        plan = self.plan_reuse(req)
        needed = 0
        context = req.context_len
        hit = plan.hit_tokens
        for spec in self.groups:
            required = cdiv(context, spec.block_tokens)
            if spec.window_tokens:
                required = min(required, spec.max_request_slots)
            # The source's full-input admission check uses virtual table
            # lengths (including skipped/null window positions). It checks
            # currently free slots; it does not reserve later chunks globally.
            needed += max(0, required - hit // spec.block_tokens)
        zero_ref_hits = sum(
            block.ref_count == 0 for _, pairs in plan.group_hits for _, block in pairs
        )
        watermark = self.watermark_blocks if has_scheduled_requests else 0
        return self.get_num_free_blocks() - zero_ref_hits >= needed + watermark

    def remaining_admission_slots(self, req):
        """vLLM SingleTypeKVCacheManager running-request reservation.

        Window tables have skipped/null prefixes in the reference. Our sparse
        tables omit those entries, so their virtual length is the last absolute
        position plus one, not the number of physically held window pages.
        """
        tables = self.tables.get(req.id)
        if tables is None:
            return 0
        tokens = min(req.context_len, self.layout.max_model_len)
        remaining = 0
        for spec in self.groups:
            required = cdiv(tokens, spec.block_tokens)
            if spec.window_tokens:
                required = min(required, spec.max_request_slots)
            table = tables[spec.group_id]
            virtual_length = max(table, default=-1) + 1
            remaining += max(0, required - virtual_length)
        return remaining

    def allocate(self, req):
        if req.id in self.tables:
            raise SimulationStateError("duplicate native hybrid request admission")
        plan = self.plan_reuse(req)
        tables = self.tables[req.id] = {spec.group_id: {} for spec in self.groups}
        self.requests[req.id] = req
        self.cached_progress[req.id] = plan.hit_tokens // self.block_size
        self._committed.pop(req.id, None)
        for gid, pairs in plan.group_hits:
            table = tables[gid]
            for index, block in pairs:
                self._acquire(block)
                table[index] = block
        self._spans[req.id] = self._window_spans(tables)
        req.cached_prefill_tokens = min(req.prefill_len, plan.hit_tokens)
        req.cached_prefill_blocks = req.cached_prefill_tokens // self.block_size
        req.gpu_cached_context_tokens = plan.hit_tokens
        req.effective_prefill_tokens = plan.effective_prefill_tokens
        req.reuse_hit_blocks = req.cached_prefill_blocks
        req.reuse_miss_blocks = plan.reuse_miss_blocks
        req.input_cache_keys = list(plan.input_keys)
        req.input_cache_committed = False

    @staticmethod
    def _span(table):
        """``(first, top)`` when a table holds exactly ``range(first, top)`` in order."""
        indices = list(table)
        if not indices:
            return (0, 0)
        first = indices[0]
        if indices == list(range(first, first + len(indices))):
            return (first, first + len(indices))
        return None

    def _window_spans(self, tables):
        return [
            self._span(tables[spec.group_id]) if spec.window_tokens else None
            for spec in self.groups
        ]

    def can_ensure_slots(self, req, scheduled_tokens, *, lookahead=0):
        computed = req.num_computed_tokens
        end = min(self.layout.max_model_len, computed + scheduled_tokens + lookahead)
        tables = self.tables[req.id]
        spans = self._spans[req.id]
        needed = reclaimable = 0
        for group_id, bt in self._full_growth:
            grow = (end + bt - 1) // bt - len(tables[group_id])
            if grow > 0:
                needed += grow
        for position, group_id, bt, history in self._window_growth:
            table = tables[group_id]
            high = (end + bt - 1) // bt
            low = (computed - history) // bt if computed > history else 0
            span = spans[position]
            if span is None:
                needed += sum(index not in table for index in range(low, high))
                reclaimable += sum(
                    block.ref_count == 1
                    for index, block in table.items()
                    if index < low
                )
                continue
            first, top = span
            if high > low:
                overlap = (high if high < top else top) - (
                    low if low > first else first
                )
                needed += high - low - (overlap if overlap > 0 else 0)
            # Window pages before the window are released before allocation.
            for index in range(first, low if low < top else top):
                if table[index].ref_count == 1:
                    reclaimable += 1
        return needed <= len(self._free_uncached) + len(self._free_cached) + reclaimable

    def ensure_slots(self, req, scheduled_tokens, *, lookahead=0):
        computed = req.num_computed_tokens
        end = min(self.layout.max_model_len, computed + scheduled_tokens + lookahead)
        tables = self.tables[req.id]
        spans = self._spans[req.id]
        # Reclaim all old windows before allocating any group's new blocks.
        # can_ensure_slots includes these reclaimable slots in its decision.
        # vLLM frees skipped pages newest first.
        for position, group_id, bt, history in self._window_growth:
            table = tables[group_id]
            low = (computed - history) // bt if computed > history else 0
            span = spans[position]
            if span is None:
                skipped = [index for index in reversed(table) if index < low]
                if skipped:
                    self._release_many([table.pop(index) for index in skipped])
                continue
            first, top = span
            cut = low if low < top else top
            if cut > first:
                self._release_many(
                    [table.pop(index) for index in range(cut - 1, first - 1, -1)]
                )
                spans[position] = (cut, top)
        # Allocation follows the layout's group order.
        for position, spec in enumerate(self.groups):
            table = tables[spec.group_id]
            bt = spec.block_tokens
            high = (end + bt - 1) // bt
            if not spec.window_tokens:
                if high > len(table):
                    table.update(
                        zip(
                            range(len(table), high),
                            self._allocate_many(high - len(table)),
                        )
                    )
                continue
            history = spec.window_tokens - 1
            low = (computed - history) // bt if computed > history else 0
            span = spans[position]
            if span is not None and (span[0] == span[1] or span[0] == low):
                # Contiguous from the window start (or empty): append the tail.
                start = max(low, span[1]) if span[0] != span[1] else low
                if high > start:
                    table.update(
                        zip(range(start, high), self._allocate_many(high - start))
                    )
                    spans[position] = (low if span[0] == span[1] else span[0], high)
                    if start * bt < computed:
                        # Pages behind the computed position may already be
                        # publishable; recheck at the next publication pass.
                        self._committed.pop(req.id, None)
                continue
            missing = [index for index in range(low, high) if index not in table]
            if missing:
                table.update(zip(missing, self._allocate_many(len(missing))))
                if missing[0] * bt < computed:
                    self._committed.pop(req.id, None)
            spans[position] = self._span(table)

    def free(self, req):
        self.cache_computed(req)
        self.release_request_blocks(req)

    def release_request_blocks(self, req):
        owned = req.id in self.tables
        tables = self.tables.pop(req.id, {})
        for table in tables.values():
            # Tail first, so the earliest full-prefix blocks are evicted last.
            self._release_many(reversed(table.values()))
        if owned:
            req._physical_token_blocks.clear()
        self.requests.pop(req.id, None)
        self.cached_progress.pop(req.id, None)
        self.keys.pop(req.id, None)
        self.fine_keys.pop(req.id, None)
        self._plans.pop(req.id, None)
        self._committed.pop(req.id, None)
        self._spans.pop(req.id, None)

    def can_append_slot(self, req):
        return self.can_ensure_slots(req, 1, lookahead=self.layout.speculative_tokens)

    def append_slot(self, req):
        self.ensure_slots(req, 1, lookahead=self.layout.speculative_tokens)

    @property
    def free_slots(self):
        """Free slots in allocation order (diagnostics; linear time)."""
        order = OrderedDict(
            (block.block_number, None) for block in reversed(self._free_uncached)
        )
        order.update(self._free_cached)
        return order

    def get_num_free_blocks(self):
        return len(self._free_uncached) + len(self._free_cached)

    def get_num_allocated_blocks(self):
        return self.num_total_gpu_blocks - self.get_num_free_blocks()

    def get_gpu_status(self):
        return (
            self.get_num_free_blocks(),
            self.get_num_allocated_blocks(),
            self.num_total_gpu_blocks,
        )

    def get_request_block_counts(self, requests):
        return {req.id: self.block_table.get_num_blocks(req.id) for req in requests}

    def get_num_blocks(self, req):
        return self.block_table.get_num_blocks(req.id)

    def validate_conservation(self):
        references = dict(self.transfer_references)
        for tables in self.tables.values():
            for table in tables.values():
                for block in table.values():
                    references[block.block_number] = (
                        references.get(block.block_number, 0) + 1
                    )
        free = [block.block_number for block in self._free_uncached]
        free.extend(self._free_cached)
        if len(free) != len(set(free)):
            raise SimulationStateError("hybrid free queue holds a slot twice")
        free = set(free)
        for block in self.blocks[1:]:
            if block.ref_count != references.get(block.block_number, 0):
                raise SimulationStateError(
                    "hybrid pool reference-count conservation failed"
                )
            if (block.ref_count == 0) != (block.block_number in free):
                raise SimulationStateError("hybrid free queue conservation failed")
        for request_id, spans in self._spans.items():
            for span, spec in zip(spans, self.groups):
                if span is not None and list(
                    self.tables[request_id][spec.group_id]
                ) != list(range(*span)):
                    raise SimulationStateError("hybrid window span conservation failed")
        for key, block in self.cache.items():
            if not block.cached or block.block_hash != key:
                raise SimulationStateError(
                    "hybrid prefix cache metadata conservation failed"
                )
            for duplicate in self.duplicates.get(key, {}).values():
                if not duplicate.cached or duplicate.block_hash != key:
                    raise SimulationStateError(
                        "hybrid duplicate cache metadata conservation failed"
                    )
        return True
