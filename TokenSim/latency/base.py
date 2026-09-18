from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from TokenSim.llm.llm_request import Request

PREFILL_ROUNDING_ALIGNMENT = 128


class LatencyBackend(ABC):
    """Estimates the wall time (seconds) of one scheduler step on one worker."""

    @abstractmethod
    def estimate_step_latency(self, requests: list[Request]) -> float:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """Provenance of the estimates produced by this backend (for results)."""
        return {"backend": type(self).__name__}

    def stats_dict(self) -> dict[str, Any]:
        return {}


def backend_prefill_len(prefill_len: int) -> int:
    return max(1, prefill_len)


def rounded_prefill_len(prefill_len: int) -> int:
    return int(prefill_len / PREFILL_ROUNDING_ALIGNMENT) * PREFILL_ROUNDING_ALIGNMENT + (
        PREFILL_ROUNDING_ALIGNMENT
    )


def is_context_build(requests: list[Request]) -> bool:
    """Prefill and recompute steps both build KV context from scratch."""
    first = requests[0]
    return first.is_prefill or bool(getattr(first, "needs_recompute", False))


def request_context_tokens(req: Request) -> int:
    """Tokens this request contributes to a context-building step."""
    if getattr(req, "needs_recompute", False):
        return int(req.recompute_tokens)
    return int(req.prefill_compute_len)


def request_kv_len(req: Request) -> int:
    """Full KV length that a context-building step attends against."""
    if getattr(req, "needs_recompute", False):
        return int(req.recompute_tokens)
    return int(req.prefill_len)
