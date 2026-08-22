from __future__ import annotations

from TokenSim.block.block import Device, PhysicalTokenBlock
from TokenSim.block.free_queue import FreeBlockQueue
from TokenSim.errors import SimulationStateError


class BlockPool:
    """Owns physical block objects and the free queue for one device.

    Block objects are materialized lazily: never-used blocks exist only as the
    ``_next_fresh`` counter. At large capacities (hundreds of thousands of
    blocks per worker) eagerly building one Python object per block dominated
    startup time and added a large constant RSS term. Hand-out order matches
    the eager implementation: fresh ids in ascending order first, then
    recycled uncached blocks (FIFO), then cached blocks in LRU order.
    """

    def __init__(self, device: Device, block_size: int, num_blocks: int):
        self.device = device
        self.block_size = block_size
        self.num_blocks = int(num_blocks)
        self._materialized: dict[int, PhysicalTokenBlock] = {}
        self._next_fresh = 0
        self.free_blocks: FreeBlockQueue = FreeBlockQueue()

    def pop_free_block(self) -> PhysicalTokenBlock:
        if self._next_fresh < self.num_blocks:
            block = PhysicalTokenBlock(self.device, self._next_fresh, self.block_size)
            self._materialized[block.block_number] = block
            self._next_fresh += 1
            return block
        return self.free_blocks.popleft()

    def append_free_block(self, block: PhysicalTokenBlock) -> None:
        self.free_blocks.append(block)

    def remove_from_free_queue(self, block: PhysicalTokenBlock) -> None:
        self.free_blocks.remove(block)

    def get_num_free_blocks(self) -> int:
        return (self.num_blocks - self._next_fresh) + len(self.free_blocks)

    def owns(self, block: PhysicalTokenBlock) -> bool:
        return self._materialized.get(block.block_number) is block

    def validate_owned(self, block: PhysicalTokenBlock) -> None:
        if block.device != self.device:
            raise SimulationStateError(
                f"cannot use {block.device} block {block.block_number} "
                + f"through {self.device} pool"
            )
        if not self.owns(block):
            raise SimulationStateError(
                f"block {block.block_number} does not belong to {self.device} pool"
            )

    def set_num_free_blocks(self, block_num: int) -> None:
        if block_num < 0 or block_num > self.num_blocks:
            raise SimulationStateError(f"invalid free block count {block_num}")
        # Mirror the eager implementation: ids [0, block_num) are free (queued
        # in ascending order), the rest are considered in use (ref_count 1).
        self.free_blocks = FreeBlockQueue()
        self._materialized = {}
        self._next_fresh = self.num_blocks
        for block_id in range(self.num_blocks):
            block = PhysicalTokenBlock(self.device, block_id, self.block_size)
            self._materialized[block_id] = block
            if block_id < block_num:
                self.free_blocks.append(block)
            else:
                block.ref_count = 1
