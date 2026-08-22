"""O(log n) free-block queue replacing linear deque scans.

The previous implementation kept one deque mixing uncached (FIFO) and cached
(sorted by ``last_accessed`` via linear scan + insert) blocks; membership
checks, removals, and sorted inserts were all O(queue length), which dominated
runtime at large block counts. This structure keeps the same eviction
intent — uncached blocks are handed out first, then cached blocks in
least-recently-accessed order — with O(1)/O(log n) operations.

Known (accepted) drift vs. the old mixed deque: previously an uncached block
appended *after* a cached block could be popped after it; now uncached blocks
are always preferred before evicting any cached block.
"""

from __future__ import annotations

import heapq
from collections import deque

from TokenSim.block.block import PhysicalTokenBlock


class _Entry:
    __slots__ = ("block", "valid")

    def __init__(self, block: PhysicalTokenBlock):
        self.block = block
        self.valid = True


class FreeBlockQueue:
    def __init__(self, blocks: list[PhysicalTokenBlock] | None = None):
        self._fifo: deque[_Entry] = deque()
        self._heap: list[tuple[int, int, _Entry]] = []
        self._entries: dict[int, _Entry] = {}
        self._seq = 0
        for block in blocks or []:
            self.append(block)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __contains__(self, block: PhysicalTokenBlock) -> bool:
        return block.block_number in self._entries

    def __iter__(self):
        for entry in self._entries.values():
            yield entry.block

    def append(self, block: PhysicalTokenBlock) -> None:
        """FIFO append (uncached blocks)."""
        self._invalidate(block)
        entry = _Entry(block)
        self._entries[block.block_number] = entry
        self._fifo.append(entry)

    def append_unique(self, block: PhysicalTokenBlock) -> None:
        if block not in self:
            self.append(block)

    def append_lru(self, block: PhysicalTokenBlock) -> None:
        """Insert a cached block ordered by its ``last_accessed`` stamp."""
        self._invalidate(block)
        entry = _Entry(block)
        self._entries[block.block_number] = entry
        self._seq += 1
        heapq.heappush(self._heap, (block.last_accessed, self._seq, entry))

    def popleft(self) -> PhysicalTokenBlock:
        while self._fifo:
            entry = self._fifo.popleft()
            if entry.valid:
                return self._finalize_pop(entry)
        while self._heap:
            _, _, entry = heapq.heappop(self._heap)
            if entry.valid:
                return self._finalize_pop(entry)
        raise IndexError("pop from an empty FreeBlockQueue")

    def remove(self, block: PhysicalTokenBlock) -> None:
        """Remove a block if present (no-op otherwise)."""
        self._invalidate(block)

    def _invalidate(self, block: PhysicalTokenBlock) -> None:
        entry = self._entries.pop(block.block_number, None)
        if entry is not None:
            entry.valid = False

    def _finalize_pop(self, entry: _Entry) -> PhysicalTokenBlock:
        entry.valid = False
        self._entries.pop(entry.block.block_number, None)
        return entry.block
