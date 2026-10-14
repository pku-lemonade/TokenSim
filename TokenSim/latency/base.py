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


def step_tokens(req: Request) -> int:
    """Tokens ``req`` computes in the current step (see ``Request.step_tokens``)."""
    return int(req.step_tokens)


def step_kv_len(req: Request) -> int:
    """KV length the step's queries attend against: cached context plus this step's tokens."""
    return int(getattr(req, "num_computed_tokens", 0)) + step_tokens(req)


def is_decode_step(req: Request) -> bool:
    """A single-token step of a request that already sampled: decode-attention shape.

    Prefill chunks and recompute chunks are context builds and use the
    context-attention shape even when the chunk is one token long.
    """
    return step_tokens(req) == 1 and int(getattr(req, "generation_idx", 0)) > 0 and not bool(
        getattr(req, "needs_recompute", False)
    )
