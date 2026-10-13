"""Measured compressed-attention estimate and source geometry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionEstimate:
    latency_us: float
    match: str
    source_id: str
    ragged_grid: bool = False
    source_batch_size: int | None = None
    source_context_len: int | None = None
    source_query_tokens: int | None = None
