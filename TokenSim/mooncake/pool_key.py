from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from TokenSim.block.prefix_cache import build_prefix_keys, next_parent_hash
from TokenSim.config.parallel_config import ParallelRankInfo
from TokenSim.llm.llm_request import Request


@dataclass(frozen=True, order=True)
class KeyMetadata:
    model_name: str
    tp_rank: int = 0
    pp_rank: int = 0
    dp_rank: int = 0
    engine_id: str = "default"
    kv_cache_group_id: str = "dp0-pp0"
    group_id: int = 0
    pcp_rank: int = 0
    dcp_rank: int = 0


@dataclass(frozen=True, order=True)
class PoolKey:
    key_metadata: KeyMetadata
    chunk_hash: str

    @cached_property
    def string(self) -> str:
        md = self.key_metadata
        return (
            f"{md.model_name}"
            f"@tp_rank:{md.tp_rank}"
            f"@pcp{md.pcp_rank}"
            f"@dcp{md.dcp_rank}"
            f"@pp_rank:{md.pp_rank}"
            f"@dp_rank:{md.dp_rank}"
            f"@engine:{md.engine_id}"
            f"@kv_group:{md.kv_cache_group_id}"
            f"@group:{md.group_id}"
            f"@{self.chunk_hash}"
        )

    def to_string(self) -> str:
        return self.string


def pool_keys_for_request(
    req: Request,
    *,
    model_name: str,
    rank_info: ParallelRankInfo | None = None,
    engine_id: str = "default",
    group_id: int = 0,
    pcp_rank: int = 0,
    dcp_rank: int = 0,
) -> list[PoolKey]:
    if not req.hash_ids:
        return []
    rank_info = rank_info or ParallelRankInfo()
    full_input_blocks = req.prefill_len // req.block_size
    prefix_keys = list(req.input_cache_keys[:full_input_blocks])
    if len(prefix_keys) != full_input_blocks:
        prefix_keys = build_prefix_keys(
            req.hash_ids[:full_input_blocks],
            model=model_name,
            cache_salt=req.cache_salt,
            reuse_group=req.reuse_group,
        )
    metadata = KeyMetadata(
        model_name=model_name,
        tp_rank=rank_info.tp_rank,
        pp_rank=rank_info.pp_rank,
        dp_rank=rank_info.dp_rank,
        engine_id=engine_id,
        kv_cache_group_id=rank_info.kv_cache_group_id,
        group_id=group_id,
        pcp_rank=pcp_rank,
        dcp_rank=dcp_rank,
    )
    if not prefix_keys:
        return []
    chunk_hashes = [key.parent_hash for key in prefix_keys[1:]]
    chunk_hashes.append(next_parent_hash(prefix_keys[-1]))
    return [PoolKey(metadata, chunk_hash) for chunk_hash in chunk_hashes]
