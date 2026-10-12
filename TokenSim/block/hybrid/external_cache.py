"""External suffix lookup, private import pages and transfer references."""

from __future__ import annotations

from TokenSim.errors import OutOfBlocksError, SimulationStateError
from .kv_layout import cdiv


class ExternalCacheMixin:
    """External storage operations on the manager's shared slot pool."""

    def plan_external_reuse(self, req, store, *, local_tokens=0):
        """Atomic all-group CPU suffix lookup after the local GPU prefix.

        Window pages use their absolute prefix identity. Full pages before
        local_tokens need not exist in external storage. Stored writes are
        invisible until ready; the connector must pin this plan before import.
        Store lookups refresh LRU order, so their sequence is part of the model.
        """
        keys, key_for = self._slot_key_function(req)
        start = local_tokens // self.block_size
        maximum = min(len(keys), max(0, (req.context_len - 1) // self.block_size))
        full = self.full_spec
        lookup = store.lookup
        contiguous = start
        while (
            contiguous < maximum
            and lookup((full.group_id, keys[contiguous], 0)) is not None
        ):
            contiguous += 1
        for blocks in range(contiguous, start, -1):
            boundary = blocks * self.block_size
            group_hits = []
            for spec in self.groups:
                if spec.window_tokens:
                    end = boundary // spec.block_tokens + spec.eagle
                    indices = range(max(0, end - spec.need), end)
                else:
                    indices = range(start, blocks)
                pairs = [(index, lookup(key_for(spec, index))) for index in indices]
                if any(page is None for _, page in pairs):
                    break
                group_hits.append(
                    (spec.group_id, tuple(pairs[:-1] if spec.eagle else pairs))
                )
            else:
                return boundary, tuple(group_hits)
        return local_tokens, ()

    def external_import_slots(self, req, external_hits):
        # Existing imported-window positions are replaced with private pages;
        # only sole-owner old pages can supply immediate reclaimable slots.
        return sum(len(pairs) for _, pairs in external_hits) - sum(
            self.tables[req.id][gid][index].ref_count == 1
            for gid, pairs in external_hits
            for index, _ in pairs
            if index in self.tables[req.id][gid]
        )

    def allocate_external_destinations(self, req, external_hits):
        """Reserve private writable GPU targets before asynchronous DMA."""
        needed = self.external_import_slots(req, external_hits)
        if needed > self.get_num_free_blocks():
            raise OutOfBlocksError("external hybrid import needs more free GPU slots")
        destinations = []
        for gid, pairs in external_hits:
            table = self.tables[req.id][gid]
            for index, _ in pairs:
                if index in table:
                    self._release(table.pop(index))
        for gid, pairs in external_hits:
            table = self.tables[req.id][gid]
            for index, page in pairs:
                block = self._allocate()
                table[index] = block
                destinations.append((gid, index, page, block))
        self._committed.pop(req.id, None)
        self._spans[req.id] = self._window_spans(self.tables[req.id])
        return destinations

    def finish_external_import(self, req, destinations, hit_tokens):
        for gid, index, page, block in destinations:
            key = page.key
            existing = self.cache.get(key)
            block.block_hash, block.cached, block.is_full = key, True, True
            if existing is None:
                self.cache[key] = block
            elif existing is not block:
                self.duplicates.setdefault(key, {})[block.block_number] = block
        self.cached_progress[req.id] = hit_tokens // self.block_size
        self._cache_version += 1
        self._committed.pop(req.id, None)

    def pin_transfer(self, block):
        """DMA owns a reference independently of the request's lifetime."""
        self._acquire(block)
        slot = block.block_number
        self.transfer_references[slot] = self.transfer_references.get(slot, 0) + 1

    def unpin_transfer(self, block):
        slot = block.block_number
        count = self.transfer_references.get(slot, 0)
        if count <= 0:
            raise SimulationStateError("invalid native DMA reference release")
        if count == 1:
            del self.transfer_references[slot]
        else:
            self.transfer_references[slot] = count - 1
        self._release(block)

    def can_allocate_external(
        self,
        req,
        local_plan,
        boundary,
        hits,
        *,
        reserved_blocks=0,
        has_scheduled_requests=False,
    ):
        """Include private DMA targets and the remaining full-input admission.

        Loaded windows replace local window positions. Shared local pages
        cannot be reclaimed; zero-reference local hits first leave the queue.
        """
        local = {gid: dict(pairs) for gid, pairs in local_plan.group_hits}
        destinations = sum(len(pairs) for _, pairs in hits)
        reclaimed = sum(
            local.get(gid, {}).get(index) is not None
            and local[gid][index].ref_count == 0
            for gid, pairs in hits
            for index, _ in pairs
        )
        local_free = sum(
            b.ref_count == 0 for _, pairs in local_plan.group_hits for _, b in pairs
        )
        future = 0
        for spec in self.groups:
            required = cdiv(req.context_len, spec.block_tokens)
            if spec.window_tokens:
                required = min(required, spec.max_request_slots)
            future += max(0, required - boundary // spec.block_tokens)
        initial = local_free + destinations - reclaimed
        watermark = self.watermark_blocks if has_scheduled_requests else 0
        free = self.get_num_free_blocks()
        # kv_cache_manager.allocate_slots: full_sequence_must_fit first checks
        # the whole input against current free capacity. reserved_blocks then
        # fences the actual async allocation, not this whole-input envelope.
        return (
            initial + future + watermark <= free
            and initial + watermark <= free - reserved_blocks
        )
