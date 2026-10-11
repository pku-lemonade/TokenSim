from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
from typing import Iterable, NamedTuple

# json.dumps(..., sort_keys=True, separators=(",", ":")) quotes strings with
# this encoder; the hot hash chain below builds the same payload directly.
_quote = json.encoder.encode_basestring_ascii


class PrefixCacheKey(NamedTuple):
    parent_hash: str
    block_signature: str
    extra_hash: str


class RequestLocalPrefixKey(NamedTuple):
    """Unknown token content can only recover the same request's KV state.

    An opaque request identity survives its recovery attempts and cannot
    collide with another engine's public request ID. A separate key type
    prevents synthetic outputs or partial prompts from acquiring trace reuse.
    """

    request_identity: object
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


def _common_prefix(left: list[str], right: list[str]) -> int:
    """Length of the longest common prefix of two signature lists."""
    size = min(len(left), len(right))
    if not size or left[0] != right[0]:
        return 0
    if left[:size] == right[:size]:
        return size
    low, high = 1, size  # left[:low] == right[:low]; left[:high] differs
    while high - low > 1:
        middle = (low + high) // 2
        if left[:middle] == right[:middle]:
            low = middle
        else:
            high = middle
    return low


class _ChainCache:
    """Recently built key chains, grouped by extra hash.

    Successive turns of a conversation extend one prompt, so a new chain
    reuses the keys of the cached chain sharing the most leading signatures.
    A key depends only on its extra hash and leading signatures, so reuse
    returns exactly the keys a full rebuild would compute.
    """

    def __init__(self, scopes: int = 1024, chains_per_scope: int = 4) -> None:
        self.scope_limit = scopes
        self.chain_limit = chains_per_scope
        # extra hash -> [(signatures, keys, parent hash after the last key)]
        self.scopes: OrderedDict[str, list] = OrderedDict()

    def longest(self, extra_hash: str, signatures: list[str]):
        chains = self.scopes.get(extra_hash)
        if not chains:
            return 0, None
        self.scopes.move_to_end(extra_hash)
        best, best_chain = 0, None
        for chain in chains:
            shared = _common_prefix(chain[0], signatures)
            if shared > best:
                best, best_chain = shared, chain
                if shared == len(signatures):
                    break
        return best, best_chain

    def store(
        self, extra_hash: str, signatures: list[str], keys: tuple, parent: str
    ) -> None:
        chains = self.scopes.get(extra_hash)
        if chains is None:
            chains = self.scopes[extra_hash] = []
            if len(self.scopes) > self.scope_limit:
                self.scopes.popitem(last=False)
        else:
            self.scopes.move_to_end(extra_hash)
        # A cached chain that this one extends is redundant.
        chains[:] = [
            chain
            for chain in chains
            if _common_prefix(chain[0], signatures) < len(chain[0])
        ]
        chains.append((signatures, keys, parent))
        if len(chains) > self.chain_limit:
            del chains[0]


_chains = _ChainCache()


def build_prefix_keys(
    block_hashes: Iterable[str | int],
    model: str,
    cache_salt: str | None = None,
    reuse_group: str | None = None,
) -> list[PrefixCacheKey]:
    extra_hash = make_extra_hash(model, cache_salt, reuse_group)
    signatures = list(map(str, block_hashes))
    if not signatures:
        return []
    shared, chain = _chains.longest(extra_hash, signatures)
    if chain is None:
        keys: list[PrefixCacheKey] = []
        parent_hash = make_root_parent_hash(extra_hash)
    else:
        cached_keys = chain[1]
        keys = list(cached_keys[:shared])
        parent_hash = (
            cached_keys[shared].parent_hash if shared < len(cached_keys) else chain[2]
        )
    if shared < len(signatures):
        quoted_extra = _quote(extra_hash)
        append = keys.append
        new_key = tuple.__new__  # skips the NamedTuple constructor frame
        for signature in signatures[shared:]:
            append(new_key(PrefixCacheKey, (parent_hash, signature, extra_hash)))
            parent_hash = _chain_hash(parent_hash, signature, quoted_extra)
        _chains.store(extra_hash, signatures, tuple(keys), parent_hash)
    return keys


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
