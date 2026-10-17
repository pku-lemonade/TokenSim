from __future__ import annotations

from TokenSim.block.block import PhysicalTokenBlock


class BlockTable:
    """Tracks request-to-physical-block mappings.

    This is intentionally mapping-only. Prefix-cache lookup, sharing, eviction,
    and hash-aware reuse belong in KVCacheManager and BlockManager.
    """

    def __init__(self):
        self._request_blocks: dict[int, list[PhysicalTokenBlock]] = {}

    def add_block(self, request_id: int, block: PhysicalTokenBlock) -> None:
        self._request_blocks.setdefault(request_id, []).append(block)

    def add_blocks(self, request_id: int, blocks: list[PhysicalTokenBlock]) -> None:
        self._request_blocks.setdefault(request_id, []).extend(blocks)

    def get_blocks(self, request_id: int) -> list[PhysicalTokenBlock]:
        return list(self._request_blocks.get(request_id, []))

    def get_num_blocks(self, request_id: int) -> int:
        return len(self.get_blocks(request_id))

    def pop_blocks(self, request_id: int) -> list[PhysicalTokenBlock]:
        return self._request_blocks.pop(request_id, [])

    def set_blocks(self, request_id: int, blocks: list[PhysicalTokenBlock]) -> None:
        if blocks:
            self._request_blocks[request_id] = list(blocks)
        else:
            self._request_blocks.pop(request_id, None)

    def pop_last_block(self, request_id: int) -> PhysicalTokenBlock | None:
        blocks = self._request_blocks.get(request_id)
        if not blocks:
            return None
        block = blocks.pop()
        if not blocks:
            self._request_blocks.pop(request_id, None)
        return block

    def request_ids(self) -> tuple[int, ...]:
        return tuple(self._request_blocks)
