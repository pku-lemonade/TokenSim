"""Prefix identities, all-group recovery and computed-page publication."""

from __future__ import annotations

from dataclasses import dataclass

from TokenSim.block.prefix_cache import (
    RequestLocalPrefixKey,
    build_prefix_keys,
    make_extra_hash,
)


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


class PrefixCacheMixin:
    """Prefix operations on the manager's shared group tables and pool."""

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

    def forget_request(self, request_id):
        self.keys.pop(request_id, None)
        self.fine_keys.pop(request_id, None)
        self._plans.pop(request_id, None)
