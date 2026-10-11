"""One physical pool coordinating compressed attention full and sliding-window cache groups."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from TokenSim.block.block import Device, PhysicalTokenBlock
from TokenSim.block.prefix_cache import (
    build_prefix_keys,
    make_extra_hash,
    RequestLocalPrefixKey,
)
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
        return (
            max(0, self.full_input_blocks - len(self.hit_blocks))
            if self.has_hash_ids
            else 0
        )


class _GroupSpec:
    """Per-group constants read on every scheduling step."""

    __slots__ = (
        "group",
        "group_id",
        "block_tokens",
        "window_tokens",
        "eagle",
        "need",
        "max_request_slots",
    )

    def __init__(self, group, layout):
        self.group = group
        self.group_id = group.group_id
        self.block_tokens = group.block_tokens
        self.window_tokens = group.window_tokens
        self.eagle = int(group.eagle)
        self.need = group.hit_tail_slots
        self.max_request_slots = group.max_request_slots(
            layout.max_model_len, layout.max_in_flight_tokens
        )


class _HybridBlockTable:
    """Request-to-slot view derived from the group tables in layout order."""

    def __init__(self, manager):
        self._manager = manager

    def get_blocks(self, request_id):
        tables = self._manager.tables.get(request_id)
        if not tables:
            return []
        return [block for table in tables.values() for block in table.values()]

    def get_num_blocks(self, request_id):
        tables = self._manager.tables.get(request_id)
        return sum(len(table) for table in tables.values()) if tables else 0


class HybridKVBlockManager:
    """Group-aware slot allocation with atomic prefix recovery and retention.

    Short-block identities use the containing known prefix hash and a subblock
    ordinal; matching remains at complete engine-block boundaries. Unknown
    generated/tail content is never published under a trace prefix hash.

    The free queue follows vLLM's BlockPool order: uncached slots first, most
    recently released first, then cached slots from least to most recently
    released. Uncached slots are interchangeable, so they live in a stack and
    only cached slots need an ordered map.
    """

    grouped_cache = True

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
        # Stack top (list end) is the front of the free queue: slot 1 first.
        self._free_uncached = self.blocks[:0:-1]
        self._free_cached = OrderedDict()
        self.cache = {}
        self.duplicates = {}
        self.cached_progress = {}
        self.tables = {}
        self.requests = {}
        self.keys = {}
        self.fine_keys = {}
        self.transfer_references = {}
        self.publish_callbacks = []
        self.block_table = _HybridBlockTable(self)
        self.kv_cache_manager = self
        self.gpu_allocator = self
        self.cpu_allocator = NoHostSwapAllocator()
        self.allocation_count = self.release_count = self.eviction_count = 0
        self.peak_used_slots = 1
        self.groups = tuple(_GroupSpec(group, layout) for group in layout.groups)
        self.full_spec = next(
            spec for spec in self.groups if spec.group is layout.full_group
        )
        # (position, group id, page tokens, window history) per group: window
        # pages before ``computed - history`` fall out of the window.
        self._full_growth = tuple(
            (spec.group_id, spec.block_tokens)
            for spec in self.groups
            if not spec.window_tokens
        )
        self._window_growth = tuple(
            (position, spec.group_id, spec.block_tokens, spec.window_tokens - 1)
            for position, spec in enumerate(self.groups)
            if spec.window_tokens
        )
        # Retained tails can be enumerated per boundary when each tail run
        # fits in one retention segment; otherwise test every window page.
        interval = layout.retention_interval
        self._tails_fit_segments = not interval or all(
            spec.need * spec.block_tokens <= interval for spec in self.groups
        )
        # Reuse plans are pure functions of the request and the prefix map;
        # any change to the map bumps the version and invalidates them.
        self._cache_version = 0
        self._plans = {}
        # Per request: ((computed, context_len, shared boundaries) of the last
        # publication pass, computed position at which a window page can next
        # become publishable). Cleared whenever the tables change shape.
        self._committed = {}
        # Per request: for each window group, the held contiguous index range
        # ``(first, top)`` of its table, or None when not known to be one.
        self._spans = {}

    # -- prefix identities -----------------------------------------------------

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
        return self._extend_context_keys(req, self.keys, self.block_size)

    def _extend_context_keys(self, req, cache, block_tokens):
        keys = cache[req.id]
        blocks = req.context_len // block_tokens
        if len(keys) < blocks:
            # Known prompt hashes remain shareable. Everything beyond that
            # proof uses a request-private identity, including a mixed block
            # containing the prompt's partial tail and generated outputs.
            parent = next(
                (
                    key
                    for key in reversed(keys)
                    if not isinstance(key, RequestLocalPrefixKey)
                ),
                None,
            )
            extra = make_extra_hash(self.model, req.cache_salt, req.reuse_group)
            keys += tuple(
                RequestLocalPrefixKey(
                    req.cache_identity, block_tokens, index, parent, extra
                )
                for index in range(len(keys), blocks)
            )
            cache[req.id] = keys
        return keys

    def _fine_input_keys(self, req):
        """Trace-granularity keys for groups shorter than an engine block, or None."""
        hashes = getattr(req, "trace_hash_ids", None)
        if hashes is None:
            return None
        trace_size = req.trace_block_size
        if req.id not in self.fine_keys:
            self.fine_keys[req.id] = tuple(
                build_prefix_keys(
                    hashes[: req.prefill_len // trace_size],
                    self.model,
                    req.cache_salt,
                    req.reuse_group,
                )
            )
        return self._extend_context_keys(req, self.fine_keys, trace_size)

    def _slot_key(self, group, index, keys, req=None):
        if (
            req is not None
            and group.block_tokens < self.block_size
            and getattr(req, "trace_hash_ids", None) is not None
        ):
            return self._fine_slot_key(
                group.group_id,
                group.block_tokens,
                index,
                self._fine_input_keys(req),
                req.trace_block_size,
            )
        return self._coarse_slot_key(group.group_id, group.block_tokens, index, keys)

    def _coarse_slot_key(self, group_id, block_tokens, index, keys):
        containing = ((index + 1) * block_tokens - 1) // self.block_size
        if containing >= len(keys):
            return None
        return (group_id, keys[containing], index % (self.block_size // block_tokens))

    @staticmethod
    def _fine_slot_key(group_id, block_tokens, index, fine, trace_size):
        containing = ((index + 1) * block_tokens - 1) // trace_size
        if containing >= len(fine):
            return None
        return (group_id, fine[containing], index % (trace_size // block_tokens))

    def _slot_key_function(self, req):
        """``(spec, index) -> key`` with the request's keys resolved once."""
        keys = self._input_keys(req)
        traced = getattr(req, "trace_hash_ids", None) is not None
        fine = []
        coarse = self._coarse_slot_key
        fine_key = self._fine_slot_key

        def key_for(spec, index):
            if traced and spec.block_tokens < self.block_size:
                if not fine:
                    fine.append(self._fine_input_keys(req))
                return fine_key(
                    spec.group_id,
                    spec.block_tokens,
                    index,
                    fine[0],
                    req.trace_block_size,
                )
            return coarse(spec.group_id, spec.block_tokens, index, keys)

        return keys, key_for

    # -- reuse planning ---------------------------------------------------------

    def plan_reuse(self, req):
        state = (
            req.context_len,
            req.num_logical_token_blocks,
            self._cache_version,
            len(self.cache),
        )
        memo = self._plans.get(req.id)
        if memo is not None and memo[0] == state:
            return memo[1]
        plan = self._plan_reuse(req)
        self._plans[req.id] = (state, plan)
        return plan

    def _plan_reuse(self, req):
        keys, key_for = self._slot_key_function(req)
        maximum = min(len(keys), max(0, (req.context_len - 1) // self.block_size))
        cache = self.cache
        full = self.full_spec
        found = []
        for key in keys[:maximum]:
            block = cache.get((full.group_id, key, 0))
            if block is None:
                break
            found.append(block)
        contiguous = len(found)
        matched = 0
        group_hits = ()
        for blocks in range(contiguous, 0, -1):
            boundary = blocks * self.block_size
            # Every block before ``contiguous`` is cached in the primary full
            # group, so only the other groups can reject this boundary.
            hits = []
            for spec in self.groups:
                if spec is full:
                    hits.append(None)
                    continue
                if spec.window_tokens:
                    end = boundary // spec.block_tokens + spec.eagle
                    indices = range(max(0, end - spec.need), end)
                else:
                    indices = range(blocks)
                pairs = [(index, cache.get(key_for(spec, index))) for index in indices]
                if any(block is None for _, block in pairs):
                    break
                # The extra EAGLE block proves the resume boundary is valid,
                # then is dropped so the next forward recomputes its hidden
                # states in a private allocation (vLLM SWA find_longest_hit).
                hits.append((spec.group_id, tuple(pairs[:-1] if spec.eagle else pairs)))
            else:
                full_pairs = tuple(enumerate(found[:blocks]))
                hits = [
                    (
                        (full.group_id, full_pairs[:-1] if full.eagle else full_pairs)
                        if hit is None
                        else hit
                    )
                    for hit in hits
                ]
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
            max(0, req.prefill_len - matched),
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
            self._cache_version += 1
        block.clear_cache_metadata()

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

    # -- external imports -----------------------------------------------------

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

    # -- slot pool ------------------------------------------------------------

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

    # -- publication ------------------------------------------------------------

    def _publish(self, group, index, block, keys, req):
        """Publish one complete page; False when its slot key is not known yet."""
        key = self._slot_key(group, index, keys, req)
        if key is None:
            return False
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
            self._cache_version += 1
        for callback in self.publish_callbacks:
            callback(group, index, block, req)
        return True

    def commit_input_cache(self, req):
        self.cache_computed(req)

    def cache_computed(self, req):
        tables = self.tables.get(req.id)
        if tables is None:
            return
        computed = req.num_computed_tokens
        shared = getattr(req, "shared_prefix_boundaries", None)
        state = (computed, req.context_len, shared)
        entry = self._committed.get(req.id)
        if entry is None or entry[0] != state:
            # A window page becomes publishable only once its retention
            # boundary (or EAGLE extension) is crossed; until the recorded
            # threshold, only full-group pages can change.
            windows = (
                entry is None
                or entry[0][2] != shared
                or entry[1] is None
                or computed >= entry[1]
            )
            threshold = self._publish_computed(req, tables, computed, windows)
            self._committed[req.id] = (state, threshold if windows else entry[1])
        req.input_cache_committed = not req.is_prefill

    def _publish_computed(self, req, tables, computed, windows=True):
        """Publish newly complete retained pages; return the next window threshold.

        The threshold is None when window pages must be checked again at the
        next change, for example after a page lacked its slot key.
        """
        keys = self._input_keys(req)
        interval = self.layout.retention_interval
        prefill_boundary = (
            max(0, req.prefill_len - 1) // self.block_size * self.block_size
        )
        all_boundaries = [
            boundary
            for boundary in (
                *getattr(req, "shared_prefix_boundaries", ()),
                prefill_boundary,
            )
            if boundary
        ]
        extra_boundaries = [
            boundary for boundary in all_boundaries if boundary <= computed
        ]
        deferred = False
        for spec in self.groups:
            table = tables[spec.group_id]
            group = spec.group
            bt = spec.block_tokens
            if not spec.window_tokens:
                end = min(computed // bt, len(keys))
                start = self.cached_progress[req.id]
                self.cached_progress[req.id] = end
                for index in range(start, end):
                    block = table[index]
                    if not block.cached:
                        self._publish(group, index, block, keys, req)
                continue
            if not windows or not table:
                continue
            if not self._tails_fit_segments:
                for index, block in list(table.items()):
                    if (
                        not block.cached
                        and (index + 1) * bt <= computed
                        and self._retained(spec, index, computed, extra_boundaries)
                    ):
                        deferred |= not self._publish(group, index, block, keys, req)
                continue
            # Window pages are retained only in the tail runs before each
            # retention segment end, shared prefix and prompt boundary. Window
            # tables hold ascending indices, so visiting the retained indices
            # in order publishes in table order.
            lowest = next(iter(table))
            eagle, need = spec.eagle, spec.need
            boundaries = list(extra_boundaries)
            if interval:
                # Smallest segment end whose tail reaches ``lowest``.
                first = max(1, -(-(lowest - eagle + 1) * bt // interval))
                boundaries.extend(
                    k * interval for k in range(first, computed // interval + 1)
                )
            candidates = set()
            for boundary in boundaries:
                end = boundary // bt + eagle
                candidates.update(range(max(lowest, end - need), end))
            for index in sorted(candidates):
                block = table.get(index)
                if (
                    block is not None
                    and not block.cached
                    and (index + 1) * bt <= computed
                ):
                    deferred |= not self._publish(group, index, block, keys, req)
        if not windows:
            return None
        if deferred or not self._tails_fit_segments:
            return None
        return self._window_threshold(computed, all_boundaries)

    def _window_threshold(self, computed, boundaries):
        """Smallest position past ``computed`` that can complete a retained run.

        Tails before a boundary are complete at the boundary; an EAGLE run
        ends one page past it. Boundaries are multiples of every page width.
        """
        candidates = list(boundaries)
        interval = self.layout.retention_interval
        if interval:
            segment = computed // interval * interval
            candidates += [segment, segment + interval]
        threshold = float("inf")
        for spec in self.groups:
            if not spec.window_tokens:
                continue
            extension = spec.block_tokens if spec.eagle else 0
            for boundary in candidates:
                for value in (boundary, boundary + extension):
                    if computed < value < threshold:
                        threshold = value
        return threshold

    def _retained(self, spec, index, computed, extra_boundaries):
        """Whether window page ``index`` lies in a retained tail run."""
        bt, eagle, need = spec.block_tokens, spec.eagle, spec.need
        boundaries = list(extra_boundaries)
        interval = self.layout.retention_interval
        if interval:
            # EAGLE's retained run extends one block beyond the boundary.
            # Shift before locating the segment, including its first block.
            boundary = ((index - eagle) * bt // interval) * interval + interval
            if 0 < boundary <= computed:
                boundaries.append(boundary)
        for boundary in boundaries:
            end = boundary // bt + eagle
            if max(0, end - need) <= index < end:
                return True
        return False

    # -- admission ---------------------------------------------------------------

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

    # -- per-step growth --------------------------------------------------------

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

    # -- release ------------------------------------------------------------------

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

    def forget_request(self, request_id):
        self.keys.pop(request_id, None)
        self.fine_keys.pop(request_id, None)
        self._plans.pop(request_id, None)

    def can_append_slot(self, req):
        return self.can_ensure_slots(req, 1, lookahead=self.layout.speculative_tokens)

    def append_slot(self, req):
        self.ensure_slots(req, 1, lookahead=self.layout.speculative_tokens)

    # -- accounting ---------------------------------------------------------------

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
