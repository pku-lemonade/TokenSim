from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from typing import Iterable, NamedTuple

from TokenSim.utils import BoundedCache

# json.dumps(..., sort_keys=True, separators=(",", ":")) quotes strings with
# this encoder; the hot hash chain below builds the same payload directly.
_quote = json.encoder.encode_basestring_ascii


class PrefixCacheKey(NamedTuple):
    parent_hash: str
    block_signature: str
    extra_hash: str


class RequestLocalPrefixKey(NamedTuple):
    """Unknown token content can only recover the same request's KV state.

    Request IDs are unique within an engine/replay, and native stores belong
    to that engine. A separate key type prevents trace hashes from granting
    cross-request reuse of synthetic generated tokens or a partial prompt.
    """

    request_id: int
    block_tokens: int
    block_index: int
    prompt_parent: PrefixCacheKey | None
    extra_hash: str


@lru_cache(maxsize=65536)
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


@lru_cache(maxsize=65536)
def make_root_parent_hash(extra_hash: str) -> str:
    return _stable_hash({"root": "tokensim-prefix-cache", "extra_hash": extra_hash})


def make_prefix_key(
    parent_hash: str,
    block_hash: str | int,
    extra_hash: str,
) -> PrefixCacheKey:
    return PrefixCacheKey(parent_hash, str(block_hash), extra_hash)


def _chain_hash(parent_hash: str, signature: str, quoted_extra: str) -> str:
    payload = (
        '{"block_signature":'
        + _quote(signature)
        + ',"extra_hash":'
        + quoted_extra
        + ',"parent_hash":'
        + _quote(parent_hash)
        + "}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def next_parent_hash(key: PrefixCacheKey) -> str:
    return _chain_hash(key.parent_hash, key.block_signature, _quote(key.extra_hash))


# Successive requests of a conversation share their prefix, so the same chain
# links are hashed again and again. Links are pure functions of their inputs;
# the memo is bounded and simply restarts when full.
_CHAIN_MEMO_LIMIT = 1 << 20
_chain_memo = BoundedCache(_CHAIN_MEMO_LIMIT)


def build_prefix_keys(
    block_hashes: Iterable[str | int],
    model: str,
    cache_salt: str | None = None,
    reuse_group: str | None = None,
) -> list[PrefixCacheKey]:
    extra_hash = make_extra_hash(model, cache_salt, reuse_group)
    quoted_extra = _quote(extra_hash)
    parent_hash = make_root_parent_hash(extra_hash)
    keys: list[PrefixCacheKey] = []
    append = keys.append
    new_key = tuple.__new__  # skips the NamedTuple constructor frame
    memo = _chain_memo
    for block_hash in block_hashes:
        signature = str(block_hash)
        append(new_key(PrefixCacheKey, (parent_hash, signature, extra_hash)))
        link = (parent_hash, signature, extra_hash)
        child = memo.get(link)
        if child is None:
            child = _chain_hash(parent_hash, signature, quoted_extra)
            memo[link] = child
        parent_hash = child
    return keys


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
