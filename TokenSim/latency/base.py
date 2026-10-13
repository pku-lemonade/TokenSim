from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from TokenSim.llm.llm_request import Request


class LatencyBackend(ABC):
    """Estimates the wall time (seconds) of one scheduler step on one worker."""

    @abstractmethod
    def estimate_step_latency(self, requests: list[Request]) -> float:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """Provenance of the estimates produced by this backend (for results)."""
        return {"backend": type(self).__name__}


def query_tokens(req: Request) -> int:
    """Target forward query width (MTP verify may exceed accepted output)."""
    return int(getattr(req, "query_tokens", req.step_tokens))


def step_kv_len(req: Request) -> int:
    """KV length the step's queries attend against: cached context plus this step's tokens."""
    return int(getattr(req, "num_computed_tokens", 0)) + query_tokens(req)


def is_decode_step(req: Request) -> bool:
    """A single-token step of a request that already sampled: decode-attention shape.

    Prefill chunks and recompute chunks are context builds and use the
    context-attention shape even when the chunk is one token long.
    """
    return int(getattr(req, "generation_idx", 0)) > 0 and not bool(
        getattr(req, "needs_recompute", False)
    )


def is_decode_attention_step(req: Request, max_query_len: int) -> bool:
    """SM100 varlen SWA/indexer split: short extends use decode metadata."""
    return query_tokens(req) <= max_query_len


def roofline_latency_us(
    device,
    bytes_: float,
    flops: float = 0,
    *,
    dtype: str = "fp32",
    compute_efficiency: float = 1.0,
    memory_efficiency: float | None = None,
    launches: int = 1,
) -> float:
    """Price streaming work from device rates and explicit launch count."""
    efficiency = (
        device.analytical["memory_efficiency"]
        if memory_efficiency is None
        else memory_efficiency
    )
    memory = bytes_ / (device.memory_bandwidth_bytes_per_s.value * efficiency)
    compute = flops / (device.peak_compute_for(dtype).value * compute_efficiency)
    return max(memory, compute) * 1e6 + launches * device.analytical["kernel_launch_us"]
