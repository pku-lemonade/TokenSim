from __future__ import annotations

from abc import ABC, abstractmethod

from TokenSim.llm.llm_request import Request


# Calibration constants preserved from the original roofline latency path.
PREFILL_ROUNDING_ALIGNMENT = 128
PREFILL_SCALE = 1.41
PREFILL_OFFSET_SECONDS = 0.009
DECODE_SCALE = 1 / 0.55
LLMCOMPASS_LAYER_NUM = 40


class LatencyBackend(ABC):
    @abstractmethod
    def estimate_step_latency(self, requests: list[Request]) -> float:
        raise NotImplementedError


def backend_prefill_len(prefill_len: int) -> int:
    return max(1, prefill_len)


def rounded_prefill_len(prefill_len: int) -> int:
    return int(prefill_len / PREFILL_ROUNDING_ALIGNMENT) * PREFILL_ROUNDING_ALIGNMENT + (
        PREFILL_ROUNDING_ALIGNMENT
    )
