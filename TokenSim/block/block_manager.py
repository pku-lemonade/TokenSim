from TokenSim.llm.llm_request import Request
from TokenSim.block.block import Device, PhysicalTokenBlock
from TokenSim.block.block_pool import BlockPool
from TokenSim.block.block_table import BlockTable
from TokenSim.block.kv_cache_manager import KVCacheManager, PrefixReusePlan
from TokenSim.errors import OutOfBlocksError, SimulationStateError
from typing import Tuple


class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
        kv_cache_manager: KVCacheManager | None = None,
    ) -> None:
        self.device = device
        self.block_size: int = block_size
        self.num_blocks: int = int(num_blocks)
        self.kv_cache_manager = kv_cache_manager
        self.pool = BlockPool(device, block_size, num_blocks)
        self.free_blocks = self.pool.free_blocks

    def allocate(self) -> PhysicalTokenBlock:
        if not self.get_num_free_blocks():
            raise OutOfBlocksError(
                f"out of {self.device.name} memory; no free blocks are available"
            )
        block = self.pool.pop_free_block()
        if block.cached:
            if self.kv_cache_manager is None:
                block.clear_cache_metadata()
            else:
                self.kv_cache_manager.evict(block)
        if block.ref_count != 0:
            raise SimulationStateError(
                f"{self.device.name} block {block.block_number} is free but "
                + f"has ref_count {block.ref_count}"
            )
        block.ref_count += 1
        return block

    def acquire(self, block: PhysicalTokenBlock) -> None:
        self.pool.validate_owned(block)
        if block.ref_count == 0:
            self.pool.remove_from_free_queue(block)
        block.ref_count += 1
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.touch(block)

    def free(self, block: PhysicalTokenBlock | None = None) -> None:
        if block is None:
            raise SimulationStateError(
                "BlockAllocator.free() requires a physical block"
            )
        self.pool.validate_owned(block)
        if block.ref_count <= 0:
            raise SimulationStateError(
                f"block {block.block_number} ref_count is already {block.ref_count}"
            )
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.release_block(block, self.free_blocks)
        else:
            block.ref_count -= 1
            if block.ref_count == 0:
                block.clear_cache_metadata()
                self.pool.append_free_block(block)

    def get_num_free_blocks(self) -> int:
        return self.pool.get_num_free_blocks()

    def get_num_allocated_blocks(self) -> int:
        return self.num_blocks - self.get_num_free_blocks()

    def set_num_free_blocks(self, block_num):
        self.pool.set_num_free_blocks(block_num)
        self.free_blocks = self.pool.free_blocks

    def get_status(self) -> Tuple[int, int, int]:
        return (
            self.get_num_free_blocks(),
            self.get_num_allocated_blocks(),
            self.num_blocks,
        )


class BlockManager:
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        model: str = "unknown",
        watermark: float = 0.01,
    ):
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.block_table = BlockTable()
        self.kv_cache_manager = KVCacheManager(block_size=block_size, model=model)

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(
            Device.GPU,
            block_size,
            num_gpu_blocks,
            kv_cache_manager=self.kv_cache_manager,
        )
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size, num_cpu_blocks)
        self._reserved_gpu_blocks: list[PhysicalTokenBlock] = []

    def can_allocate(self, req: Request) -> bool:
        plan = self.kv_cache_manager.plan_reuse(req)
        num_required_blocks = plan.miss_block_count
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        # Zero-ref hits sit in the free queue until acquired, so they must not
        # be counted as capacity available for miss/private allocations.
        num_free_gpu_blocks -= plan.zero_ref_hit_blocks
        return num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks

    def allocate(self, req: Request):
        # Allocate new physical token blocks that will store the prompt&generated tokens.
        plan = self.kv_cache_manager.plan_reuse(req)
        self.kv_cache_manager.apply_plan(req, plan)

        blocks = list(plan.hit_blocks)
        for block in plan.hit_blocks:
            self.gpu_allocator.acquire(block)

        for _ in range(plan.miss_block_count):
            block = self.gpu_allocator.allocate()
            blocks.append(block)
        self.block_table.add_blocks(req.id, blocks)
        self._sync_request_blocks(req, blocks)
        self._mark_input_blocks_pending(req, blocks, plan)

    def get_gpu_status(self) -> Tuple[int, int, int]:
        return self.gpu_allocator.get_status()

    def get_num_blocks(self, req: Request) -> int:
        return self._num_blocks(req)

    def _num_blocks(self, req: Request) -> int:
        return self.block_table.get_num_blocks(req.id) or req.num_physical_token_blocks

    def free(self, req: Request):
        self.commit_finished_cache(req)
        self.release_request_blocks(req)

    def commit_finished_cache(self, req: Request) -> None:
        self._commit_input_cache(req)
        self.kv_cache_manager.register_output_blocks(
            req,
            self.block_table.get_blocks(req.id),
        )

    def release_request_blocks(self, req: Request) -> None:
        blocks = self.block_table.pop_blocks(req.id) or list(req._physical_token_blocks)
        for block in blocks:
            self._free_block(block)
        req._physical_token_blocks.clear()
        self.kv_cache_manager.forget_request(req.id)

    def can_append_slot(self, req: Request, reserved_blocks: int = 0) -> bool:
        required_blocks = int(
            self.block_table.get_num_blocks(req.id) < req.num_logical_token_blocks
        )
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        return num_free_gpu_blocks >= required_blocks + reserved_blocks

    def append_slot(self, req: Request):
        """Allocate a physical slot for a new token."""
        if self.block_table.get_num_blocks(req.id) < req.num_logical_token_blocks:
            # The request has a new logical block, which
            # happens in Scheduler.update_output_tokens().
            # Allocate a new physical block.
            block = self.gpu_allocator.allocate()
            self.block_table.add_block(req.id, block)
            req._append_physical_block(block)

    def get_request_block_counts(self, requests: list[Request]) -> dict[int, int]:
        return {req.id: self._num_blocks(req) for req in requests}

    def reserve_gpu_blocks(self, num_blocks: int):
        allocated: list[PhysicalTokenBlock] = []
        for _ in range(num_blocks):
            try:
                block = self.gpu_allocator.allocate()
            except Exception:
                for block in allocated:
                    self.gpu_allocator.free(block)
                raise
            allocated.append(block)
        self._reserved_gpu_blocks.extend(allocated)

    def release_reserved_blocks(self, num_blocks: int | None = None):
        if num_blocks is None:
            num_blocks = len(self._reserved_gpu_blocks)
        if num_blocks < 0:
            raise SimulationStateError(
                f"cannot release a negative block count: {num_blocks}"
            )
        if num_blocks > len(self._reserved_gpu_blocks):
            raise SimulationStateError(
                f"cannot release {num_blocks} reserved GPU blocks; "
                + f"only {len(self._reserved_gpu_blocks)} are reserved"
            )
        for _ in range(num_blocks):
            self.gpu_allocator.free(self._reserved_gpu_blocks.pop(0))

    def _free_block(self, block: PhysicalTokenBlock) -> None:
        if block.device == Device.GPU:
            self.gpu_allocator.free(block)
        elif block.device == Device.CPU:
            self.cpu_allocator.free(block)
        else:
            raise SimulationStateError(f"unknown block device {block.device}")

    @staticmethod
    def _validate_blocks_device(
        req: Request,
        blocks: list[PhysicalTokenBlock],
        expected_device: Device,
        operation: str,
    ) -> None:
        for block in blocks:
            if block.device != expected_device:
                raise SimulationStateError(
                    f"request {req.id} has {block.device.name} block {block} "
                    + f"during {operation}; expected {expected_device.name}"
                )

    @staticmethod
    def _sync_request_blocks(req: Request, blocks: list[PhysicalTokenBlock]) -> None:
        req._physical_token_blocks.clear()
        for block in blocks:
            req._append_physical_block(block)

    def _mark_input_blocks_pending(
        self,
        req: Request,
        blocks: list[PhysicalTokenBlock],
        plan: PrefixReusePlan,
    ) -> None:
        if not plan.input_keys:
            req.input_cache_keys = []
            return
        req.input_cache_keys = list(plan.input_keys)
        for index, key in enumerate(plan.input_keys):
            if index < plan.hit_block_count or index >= len(blocks):
                continue
            block = blocks[index]
            block.block_hash = key
            block.is_full = True
            block.cached = False

    def commit_input_cache(self, req: Request) -> None:
        self._commit_input_cache(req)

    def _commit_input_cache(self, req: Request) -> None:
        if req.input_cache_committed:
            return
        blocks = self.block_table.get_blocks(req.id)
        input_keys = getattr(req, "input_cache_keys", [])
        if input_keys:
            self.kv_cache_manager.register_blocks(blocks[: len(input_keys)], input_keys)
        req.input_cache_committed = True
