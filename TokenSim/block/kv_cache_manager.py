from __future__ import annotations

from dataclasses import dataclass

from TokenSim.block.block import Device, PhysicalTokenBlock
from TokenSim.block.free_queue import FreeBlockQueue
from TokenSim.block.prefix_cache import PrefixCacheKey, build_prefix_keys
from TokenSim.errors import SimulationStateError
from TokenSim.llm.llm_request import Request


@dataclass(frozen=True)
class PrefixReusePlan:
    request_id: int
    hit_blocks: tuple[PhysicalTokenBlock, ...]
    input_keys: tuple[PrefixCacheKey, ...]
    full_input_blocks: int
    total_logical_blocks: int
    has_hash_ids: bool
    hit_tokens: int
    effective_prefill_tokens: int

    @property
    def hit_block_count(self) -> int:
        return len(self.hit_blocks)

    @property
    def miss_block_count(self) -> int:
        return self.total_logical_blocks - self.hit_block_count

    @property
    def reuse_miss_blocks(self) -> int:
        if not self.has_hash_ids:
            return 0
        return self.full_input_blocks - self.hit_block_count

    @property
    def zero_ref_hit_blocks(self) -> int:
        return sum(1 for block in self.hit_blocks if block.ref_count == 0)


class KVCacheManager:
    """GPU-only prefix cache index for workload-provided block hashes."""

    def __init__(self, block_size: int, model: str):
        self.block_size = block_size
        self.model = model
        self._cache: dict[PrefixCacheKey, PhysicalTokenBlock] = {}
        self._access_counter = 0
        # plan_reuse() runs for the waiting-queue head on every scheduling
        # attempt; cache the sha256 key chain per live request.
        self._input_keys_cache: dict[int, tuple[PrefixCacheKey, ...]] = {}

    def plan_reuse(self, req: Request) -> PrefixReusePlan:
        full_input_blocks = req.prefill_len // self.block_size
        input_keys = self._build_input_keys(req, full_input_blocks)

        hit_blocks: list[PhysicalTokenBlock] = []
        for key in input_keys:
            block = self._cache.get(key)
            if not self._is_gpu_hit(block):
                break
            hit_blocks.append(block)

        hit_tokens = len(hit_blocks) * self.block_size
        return PrefixReusePlan(
            request_id=req.id,
            hit_blocks=tuple(hit_blocks),
            input_keys=tuple(input_keys),
            full_input_blocks=full_input_blocks,
            total_logical_blocks=req.num_logical_token_blocks,
            has_hash_ids=bool(req.hash_ids),
            hit_tokens=hit_tokens,
            effective_prefill_tokens=req.prefill_len - hit_tokens,
        )

    def apply_plan(self, req: Request, plan: PrefixReusePlan) -> None:
        self.apply_reuse_metrics(req, plan)
        req.input_cache_keys = list(plan.input_keys)
        req.input_cache_committed = False

    @staticmethod
    def apply_reuse_metrics(req: Request, plan: PrefixReusePlan) -> None:
        req.cached_prefill_blocks = plan.hit_block_count
        req.cached_prefill_tokens = plan.hit_tokens
        req.effective_prefill_tokens = plan.effective_prefill_tokens
        req.reuse_hit_blocks = plan.hit_block_count
        req.reuse_miss_blocks = plan.reuse_miss_blocks

    def register_blocks(
        self,
        blocks: list[PhysicalTokenBlock],
        keys: list[PrefixCacheKey],
    ) -> None:
        for block, key in zip(blocks, keys):
            if block.block_hash is not None and block.block_hash != key:
                self.evict(block)
            block.block_hash = key
            block.is_full = True
            block.cached = True
            self._cache[key] = block
            self.touch(block)

    def register_output_blocks(self, req: Request, blocks: list[PhysicalTokenBlock]) -> None:
        if not req.output_hash_ids:
            return
        if req.prefill_len % self.block_size != 0:
            return

        full_input_blocks = req.prefill_len // self.block_size
        if len(req.hash_ids or []) < full_input_blocks:
            return

        output_full_blocks = req.decode_len // self.block_size
        if output_full_blocks <= 0:
            return

        output_hash_ids = req.output_hash_ids[:output_full_blocks]
        if not output_hash_ids:
            return

        input_hash_ids = (req.hash_ids or [])[:full_input_blocks]
        keys = build_prefix_keys(
            [*input_hash_ids, *output_hash_ids],
            model=self.model,
            cache_salt=req.cache_salt,
            reuse_group=req.reuse_group,
        )[len(input_hash_ids) :]

        output_start = full_input_blocks
        output_end = min(output_start + len(keys), len(blocks))
        if output_start >= output_end:
            return
        self.register_blocks(
            blocks[output_start:output_end],
            keys[: output_end - output_start],
        )

    def release_block(
        self,
        block: PhysicalTokenBlock,
        free_queue: FreeBlockQueue,
    ) -> None:
        if block.ref_count <= 0:
            raise SimulationStateError(
                f"block {block.block_number} ref_count is already {block.ref_count}"
            )
        block.ref_count -= 1
        if block.ref_count != 0:
            return

        if not block.cached:
            block.clear_cache_metadata()
            free_queue.append_unique(block)
        else:
            free_queue.append_lru(block)

    def evict(self, block: PhysicalTokenBlock) -> None:
        if block.block_hash is not None and self._cache.get(block.block_hash) is block:
            self._cache.pop(block.block_hash, None)
        block.clear_cache_metadata()

    def touch(self, block: PhysicalTokenBlock) -> None:
        self._access_counter += 1
        block.last_accessed = self._access_counter

    def _build_input_keys(
        self,
        req: Request,
        full_input_blocks: int,
    ) -> tuple[PrefixCacheKey, ...]:
        if not req.hash_ids:
            return ()
        cached = self._input_keys_cache.get(req.id)
        if cached is None or len(cached) != full_input_blocks:
            cached = tuple(
                build_prefix_keys(
                    req.hash_ids[:full_input_blocks],
                    model=self.model,
                    cache_salt=req.cache_salt,
                    reuse_group=req.reuse_group,
                )
            )
            self._input_keys_cache[req.id] = cached
        return cached

    def forget_request(self, req_id: int) -> None:
        self._input_keys_cache.pop(req_id, None)

    @staticmethod
    def _is_gpu_hit(block: PhysicalTokenBlock | None) -> bool:
        return (
            block is not None
            and block.cached
            and block.is_full
            and block.device == Device.GPU
        )
