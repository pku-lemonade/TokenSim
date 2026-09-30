"""One physical pool coordinating DSV4 full and sliding-window cache groups."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from TokenSim.block.block import Device, PhysicalTokenBlock
from TokenSim.block.block_table import BlockTable
from TokenSim.block.prefix_cache import build_prefix_keys
from TokenSim.config.kv_layout import HybridKVLayout, cdiv
from TokenSim.errors import OutOfBlocksError, SimulationStateError


class NoHostSwapAllocator:
    def get_num_allocated_blocks(self):
        return 0

    def get_num_free_blocks(self):
        return 0

    def get_status(self):
        return 0, 0, 0


@dataclass(frozen=True)
class HybridReusePlan:
    request_id: int
    input_keys: tuple
    full_input_blocks: int
    total_logical_blocks: int
    has_hash_ids: bool
    hit_tokens: int
    effective_prefill_tokens: int
    hit_blocks: tuple
    group_hits: tuple

    @property
    def hit_block_count(self):
        return len(self.hit_blocks)

    @property
    def reuse_miss_blocks(self):
        return self.full_input_blocks - len(self.hit_blocks) if self.has_hash_ids else 0


class HybridKVBlockManager:
    """Group-aware slot allocation with atomic prefix recovery and retention.

    Short-block identities use the containing known prefix hash and a subblock
    ordinal; matching remains at complete engine-block boundaries. Unknown
    generated/tail content is never published under a trace prefix hash.
    """

    native_hybrid = True

    def __init__(
        self, layout: HybridKVLayout, num_slots: int, model: str, *, watermark=0.0
    ):
        if num_slots < 2:
            raise OutOfBlocksError(
                "native hybrid pool needs a null slot and usable slots"
            )
        self.layout, self.model = layout, model
        self.block_size = layout.logical_block_tokens
        self.num_total_gpu_blocks = num_slots
        self.num_total_cpu_blocks = 0
        self.watermark_blocks = int(num_slots * watermark)
        self.blocks = [
            PhysicalTokenBlock(Device.GPU, i, self.block_size) for i in range(num_slots)
        ]
        self.null_block = self.blocks[0]
        self.null_block.ref_count = 1
        self.free_slots = OrderedDict((i, None) for i in range(1, num_slots))
        self.cache = {}
        self.duplicates = {}
        self.cached_progress = {}
        self.tables = {}
        self.requests = {}
        self.keys = {}
        self.fine_keys = {}
        self.transfer_references = {}
        self.publish_callbacks = []
        self.block_table = BlockTable()
        self.kv_cache_manager = self
        self.gpu_allocator = self
        self.cpu_allocator = NoHostSwapAllocator()
        self.allocation_count = self.release_count = self.eviction_count = 0
        self.peak_used_slots = 1

    def _input_keys(self, req):
        if req.id not in self.keys:
            self.keys[req.id] = tuple(
                build_prefix_keys(
                    (req.hash_ids or [])[: req.prefill_len // self.block_size],
                    self.model,
                    req.cache_salt,
                    req.reuse_group,
                )
            )
        return self.keys[req.id]

    def _slot_key(self, group, index, keys, req=None):
        if (
            req is not None
            and group.block_tokens < self.block_size
            and getattr(req, "native_trace_hash_ids", None) is not None
        ):
            trace_size = req.native_trace_block_size
            if req.id not in self.fine_keys:
                self.fine_keys[req.id] = tuple(
                    build_prefix_keys(
                        req.native_trace_hash_ids[: req.prefill_len // trace_size],
                        self.model,
                        req.cache_salt,
                        req.reuse_group,
                    )
                )
            fine = self.fine_keys[req.id]
            containing = ((index + 1) * group.block_tokens - 1) // trace_size
            if containing >= len(fine):
                return None
            return (
                group.group_id,
                fine[containing],
                index % (trace_size // group.block_tokens),
            )

        end_token = (index + 1) * group.block_tokens
        containing = (end_token - 1) // self.block_size
        if containing >= len(keys):
            return None
        return (
            group.group_id,
            keys[containing],
            index % (self.block_size // group.block_tokens),
        )

    def _tail_indices(self, group, boundary, *, for_hit=True):
        end = boundary // group.block_tokens + int(group.eagle)
        need = group.hit_tail_slots
        return range(max(0, end - need), end)

    def plan_reuse(self, req):
        keys = self._input_keys(req)
        maximum = min(len(keys), max(0, (req.prefill_len - 1) // self.block_size))
        full = self.layout.full_group
        contiguous = 0
        while (
            contiguous < maximum and (full.group_id, keys[contiguous], 0) in self.cache
        ):
            contiguous += 1
        matched = 0
        group_hits = ()
        for blocks in range(contiguous, 0, -1):
            boundary = blocks * self.block_size
            hits = []
            for group in self.layout.groups:
                indices = (
                    range(blocks)
                    if not group.window_tokens
                    else self._tail_indices(group, boundary)
                )
                pairs = [
                    (index, self.cache.get(self._slot_key(group, index, keys, req)))
                    for index in indices
                ]
                if any(block is None for _, block in pairs):
                    break
                # The extra EAGLE block proves the resume boundary is valid,
                # then is dropped so the next forward recomputes its hidden
                # states in a private allocation (vLLM SWA find_longest_hit).
                hits.append(
                    (group.group_id, tuple(pairs[:-1] if group.eagle else pairs))
                )
            else:
                matched = boundary
                group_hits = tuple(hits)
                break
        if contiguous * self.block_size > matched:
            req.shared_prefix_boundaries = (contiguous * self.block_size,)
        full_hits = next(
            (
                tuple(block for _, block in pairs)
                for gid, pairs in group_hits
                if gid == full.group_id
            ),
            (),
        )
        return HybridReusePlan(
            req.id,
            keys,
            req.prefill_len // self.block_size,
            req.num_logical_token_blocks,
            bool(req.hash_ids),
            matched,
            req.prefill_len - matched,
            full_hits,
            group_hits,
        )

    def _evict(self, block):
        key = block.block_hash
        if key is not None:
            copies = self.duplicates.get(key, {})
            if self.cache.get(key) is block:
                if copies:
                    first = next(iter(copies))
                    self.cache[key] = copies.pop(first)
                else:
                    self.cache.pop(key, None)
            else:
                copies.pop(block.block_number, None)
            if not copies:
                self.duplicates.pop(key, None)
            self.eviction_count += 1
        block.clear_cache_metadata()

    def plan_external_reuse(self, req, store, *, local_tokens=0):
        """Atomic all-group CPU suffix lookup after the local GPU prefix.

        Window pages use their absolute prefix identity. Full pages before
        local_tokens need not exist in external storage. Stored writes are
        invisible until ready; the connector must pin this plan before import.
        """
        keys = self._input_keys(req)
        start = local_tokens // self.block_size
        maximum = min(len(keys), max(0, (req.prefill_len - 1) // self.block_size))
        full = self.layout.full_group
        contiguous = start
        while (
            contiguous < maximum
            and store.lookup((full.group_id, keys[contiguous], 0)) is not None
        ):
            contiguous += 1
        for blocks in range(contiguous, start, -1):
            boundary = blocks * self.block_size
            group_hits = []
            for group in self.layout.groups:
                indices = (
                    range(start, blocks)
                    if not group.window_tokens
                    else self._tail_indices(group, boundary)
                )
                pairs = [
                    (index, store.lookup(self._slot_key(group, index, keys, req)))
                    for index in indices
                ]
                if any(page is None for _, page in pairs):
                    break
                group_hits.append(
                    (group.group_id, tuple(pairs[:-1] if group.eagle else pairs))
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
        if needed > len(self.free_slots):
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
        self._sync(req)
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

    def _acquire(self, block):
        if block.ref_count == 0:
            self.free_slots.pop(block.block_number, None)
        block.ref_count += 1
        self.peak_used_slots = max(
            self.peak_used_slots, self.num_total_gpu_blocks - len(self.free_slots)
        )

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

    def _allocate(self):
        if not self.free_slots:
            raise OutOfBlocksError("native hybrid shared pool exhausted")
        slot, _ = self.free_slots.popitem(last=False)
        block = self.blocks[slot]
        if block.ref_count:
            raise SimulationStateError("free hybrid slot has live references")
        self._evict(block)
        block.ref_count = 1
        self.allocation_count += 1
        self.peak_used_slots = max(
            self.peak_used_slots, self.num_total_gpu_blocks - len(self.free_slots)
        )
        return block

    def _release(self, block):
        self._release_many([block])

    def _release_many(self, ordered_blocks):
        uncached, cached = [], []
        for block in ordered_blocks:
            if block is self.null_block or block.ref_count <= 0:
                raise SimulationStateError("invalid hybrid slot release")
            block.ref_count -= 1
            self.release_count += 1
            if block.ref_count == 0:
                (cached if block.block_hash is not None else uncached).append(block)
        # vLLM BlockPool.free_blocks: prepend non-cached blocks for LIFO
        # locality, append cached blocks for LRU eviction. Preserve the input
        # priority order within both lists (prepend_n, not individual pushes).
        for block in reversed(uncached):
            self.free_slots[block.block_number] = None
            self.free_slots.move_to_end(block.block_number, last=False)
        for block in cached:
            self.free_slots[block.block_number] = None

    def _publish(self, group, index, block, keys, req):
        key = self._slot_key(group, index, keys, req)
        if key is None:
            return
        if block.block_hash != key:
            self._evict(block)
        existing = self.cache.get(key)
        if existing is not None and existing is not block:
            # vLLM BlockHashToBlockMap retains every concurrent duplicate;
            # evicting one copy must not erase the other copies' cache entry.
            self.duplicates.setdefault(key, {})[block.block_number] = block
        block.block_hash, block.cached, block.is_full = key, True, True
        if existing is None:
            self.cache[key] = block
        for callback in self.publish_callbacks:
            callback(group, index, block, req)

    def _retained(self, group, index, computed, req):
        if not group.window_tokens:
            return True
        boundaries = []
        interval = self.layout.retention_interval
        if interval:
            # EAGLE's retained run extends one block beyond the boundary.
            # Shift before locating the segment, including its first block.
            shifted = (index - int(group.eagle)) * group.block_tokens
            start = (shifted // interval) * interval
            boundaries.append(start + interval)
        boundaries.extend(getattr(req, "shared_prefix_boundaries", ()))
        boundaries.append(
            max(0, req.prefill_len - 1) // self.block_size * self.block_size
        )
        return any(
            boundary <= computed and index in self._tail_indices(group, boundary)
            for boundary in boundaries
            if boundary
        )

    def _sync(self, req):
        flattened = [
            block
            for group in self.layout.groups
            for block in self.tables[req.id][group.group_id].values()
        ]
        self.block_table.pop_blocks(req.id)
        self.block_table.add_blocks(req.id, flattened)
        req._physical_token_blocks[:] = flattened

    def can_allocate(self, req, *, has_scheduled_requests=False):
        plan = self.plan_reuse(req)
        needed = 0
        for group in self.layout.groups:
            required = cdiv(req.context_len, group.block_tokens)
            if group.window_tokens:
                required = min(
                    required,
                    group.max_request_slots(
                        self.layout.max_model_len, self.layout.max_in_flight_tokens
                    ),
                )
            # The source's full-input admission check uses virtual table
            # lengths (including skipped/null window positions). It checks
            # currently free slots; it does not reserve later chunks globally.
            needed += max(0, required - plan.hit_tokens // group.block_tokens)
        zero_ref_hits = sum(
            block.ref_count == 0 for _, pairs in plan.group_hits for _, block in pairs
        )
        watermark = self.watermark_blocks if has_scheduled_requests else 0
        return len(self.free_slots) - zero_ref_hits >= needed + watermark

    def remaining_admission_slots(self, req):
        """vLLM SingleTypeKVCacheManager running-request reservation.

        Window tables have skipped/null prefixes in the reference. Our sparse
        tables omit those entries, so their virtual length is the last absolute
        position plus one, not the number of physically held window pages.
        """
        if req.id not in self.tables:
            return 0
        tokens = min(req.context_len, self.layout.max_model_len)
        remaining = 0
        for group in self.layout.groups:
            required = cdiv(tokens, group.block_tokens)
            if group.window_tokens:
                required = min(
                    required,
                    group.max_request_slots(
                        self.layout.max_model_len, self.layout.max_in_flight_tokens
                    ),
                )
            table = self.tables[req.id][group.group_id]
            virtual_length = max(table, default=-1) + 1
            remaining += max(0, required - virtual_length)
        return remaining

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
        for group in self.layout.groups:
            if group.window_tokens:
                required = min(
                    cdiv(req.context_len, group.block_tokens),
                    group.max_request_slots(
                        self.layout.max_model_len, self.layout.max_in_flight_tokens
                    ),
                )
                future += max(0, required - boundary // group.block_tokens)
            else:
                future += max(
                    0,
                    cdiv(req.context_len, group.block_tokens)
                    - boundary // group.block_tokens,
                )
        initial = local_free + destinations - reclaimed
        watermark = self.watermark_blocks if has_scheduled_requests else 0
        free = len(self.free_slots)
        # kv_cache_manager.allocate_slots: full_sequence_must_fit first checks
        # the whole input against current free capacity. reserved_blocks then
        # fences the actual async allocation, not this whole-input envelope.
        return (
            initial + future + watermark <= free
            and initial + watermark <= free - reserved_blocks
        )

    def allocate(self, req):
        if req.id in self.tables:
            raise SimulationStateError("duplicate native hybrid request admission")
        plan = self.plan_reuse(req)
        self.tables[req.id] = {g.group_id: {} for g in self.layout.groups}
        self.requests[req.id] = req
        self.cached_progress[req.id] = plan.hit_tokens // self.block_size
        for gid, pairs in plan.group_hits:
            for index, block in pairs:
                self._acquire(block)
                self.tables[req.id][gid][index] = block
        req.cached_prefill_blocks = plan.hit_tokens // self.block_size
        req.cached_prefill_tokens = plan.hit_tokens
        req.effective_prefill_tokens = plan.effective_prefill_tokens
        req.reuse_hit_blocks = req.cached_prefill_blocks
        req.reuse_miss_blocks = plan.reuse_miss_blocks
        req.input_cache_keys = list(plan.input_keys)
        req.input_cache_committed = False
        self._sync(req)

    def _required_indices(self, group, computed, end):
        start = (
            max(0, computed - (group.window_tokens - 1)) if group.window_tokens else 0
        )
        return range(start // group.block_tokens, cdiv(end, group.block_tokens))

    def can_ensure_slots(self, req, scheduled_tokens, *, lookahead=0):
        end = req.num_computed_tokens + scheduled_tokens + lookahead
        needed = sum(
            (
                max(0, cdiv(end, g.block_tokens) - len(self.tables[req.id][g.group_id]))
                if not g.window_tokens
                else sum(
                    index not in self.tables[req.id][g.group_id]
                    for index in self._required_indices(g, req.num_computed_tokens, end)
                )
            )
            for g in self.layout.groups
        )
        reclaimable = 0
        for group in self.layout.groups:
            if not group.window_tokens:
                continue
            start = max(0, req.num_computed_tokens - (group.window_tokens - 1))
            reclaimable += sum(
                block.ref_count == 1
                for index, block in self.tables[req.id][group.group_id].items()
                if (index + 1) * group.block_tokens <= start
            )
        return needed <= len(self.free_slots) + reclaimable

    def ensure_slots(self, req, scheduled_tokens, *, lookahead=0):
        end = req.num_computed_tokens + scheduled_tokens + lookahead
        changed = False
        # Reclaim all old windows before allocating any group's new blocks.
        # can_ensure_slots includes these reclaimable slots in its decision.
        for group in self.layout.groups:
            table = self.tables[req.id][group.group_id]
            start = (
                max(0, req.num_computed_tokens - (group.window_tokens - 1))
                if group.window_tokens
                else 0
            )
            skipped = [
                (index, block)
                for index, block in reversed(table.items())
                if (index + 1) * group.block_tokens <= start
            ]
            self._release_many(block for _, block in skipped)
            for index, _ in skipped:
                del table[index]
                changed = True
        for group in self.layout.groups:
            table = self.tables[req.id][group.group_id]
            indices = (
                range(len(table), cdiv(end, group.block_tokens))
                if not group.window_tokens
                else self._required_indices(group, req.num_computed_tokens, end)
            )
            for index in indices:
                if index not in table:
                    table[index] = self._allocate()
                    changed = True
        if changed:
            self._sync(req)

    def commit_input_cache(self, req):
        self.cache_computed(req)

    def cache_computed(self, req):
        if req.id not in self.tables:
            return
        keys = self._input_keys(req)
        for group in self.layout.groups:
            table = self.tables[req.id][group.group_id]
            if not group.window_tokens:
                end = min(req.num_computed_tokens // group.block_tokens, len(keys))
                items = (
                    (index, table[index])
                    for index in range(self.cached_progress[req.id], end)
                )
                self.cached_progress[req.id] = end
            else:
                items = table.items()
            for index, block in items:
                if block.cached:
                    continue
                if (
                    index + 1
                ) * group.block_tokens <= req.num_computed_tokens and self._retained(
                    group, index, req.num_computed_tokens, req
                ):
                    self._publish(group, index, block, keys, req)
        req.input_cache_committed = not req.is_prefill

    def free(self, req):
        self.cache_computed(req)
        self.release_request_blocks(req)

    def release_request_blocks(self, req):
        owned = req.id in self.tables
        tables = self.tables.pop(req.id, {})
        for table in tables.values():
            # Tail first, so the earliest full-prefix blocks are evicted last.
            self._release_many(reversed(table.values()))
        self.block_table.pop_blocks(req.id)
        if owned:
            req._physical_token_blocks.clear()
        self.requests.pop(req.id, None)
        self.cached_progress.pop(req.id, None)
        self.keys.pop(req.id, None)
        self.fine_keys.pop(req.id, None)

    def forget_request(self, request_id):
        self.keys.pop(request_id, None)
        self.fine_keys.pop(request_id, None)

    def can_append_slot(self, req):
        return self.can_ensure_slots(req, 1, lookahead=self.layout.speculative_tokens)

    def append_slot(self, req):
        self.ensure_slots(req, 1, lookahead=self.layout.speculative_tokens)

    def get_num_free_blocks(self):
        return len(self.free_slots)

    def get_num_allocated_blocks(self):
        return self.num_total_gpu_blocks - len(self.free_slots)

    def get_gpu_status(self):
        return (
            len(self.free_slots),
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
        for block in self.blocks[1:]:
            if block.ref_count != references.get(block.block_number, 0):
                raise SimulationStateError(
                    "hybrid pool reference-count conservation failed"
                )
            if (block.ref_count == 0) != (block.block_number in self.free_slots):
                raise SimulationStateError("hybrid free queue conservation failed")
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
