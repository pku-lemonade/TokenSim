from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from TokenSim.block.prefix_cache import PrefixCacheKey, build_prefix_keys
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

    def to_string(self) -> str:
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
    return [
        PoolKey(metadata, _prefix_key_digest(key))
        for key in prefix_keys
    ]


def _prefix_key_digest(key: PrefixCacheKey) -> str:
    return _stable_hash(
        {
            "parent_hash": key.parent_hash,
            "block_signature": key.block_signature,
            "extra_hash": key.extra_hash,
        }
    )


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
