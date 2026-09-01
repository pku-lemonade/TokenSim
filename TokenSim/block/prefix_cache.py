from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable


@dataclass(frozen=True)
class PrefixCacheKey:
    parent_hash: str
    block_signature: str
    extra_hash: str


def make_extra_hash(
    model: str,
    cache_salt: str | None = None,
    reuse_group: str | None = None,
) -> str:
    return _stable_hash(
        {
            "model": model,
            "cache_salt": cache_salt,
            "reuse_group": reuse_group,
        }
    )


def make_root_parent_hash(extra_hash: str) -> str:
    return _stable_hash({"root": "tokensim-prefix-cache", "extra_hash": extra_hash})


def make_prefix_key(
    parent_hash: str,
    block_hash: str | int,
    extra_hash: str,
) -> PrefixCacheKey:
    return PrefixCacheKey(
        parent_hash=parent_hash,
        block_signature=str(block_hash),
        extra_hash=extra_hash,
    )


def next_parent_hash(key: PrefixCacheKey) -> str:
    return _stable_hash(
        {
            "parent_hash": key.parent_hash,
            "block_signature": key.block_signature,
            "extra_hash": key.extra_hash,
        }
    )


def build_prefix_keys(
    block_hashes: Iterable[str | int],
    model: str,
    cache_salt: str | None = None,
    reuse_group: str | None = None,
) -> list[PrefixCacheKey]:
    extra_hash = make_extra_hash(model, cache_salt, reuse_group)
    parent_hash = make_root_parent_hash(extra_hash)
    keys: list[PrefixCacheKey] = []
    for block_hash in block_hashes:
        key = make_prefix_key(parent_hash, block_hash, extra_hash)
        keys.append(key)
        parent_hash = next_parent_hash(key)
    return keys


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
