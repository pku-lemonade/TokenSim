from __future__ import annotations

import logging
from typing import Any

from TokenSim.latency.base import (
    LLMCOMPASS_LAYER_NUM,
    LatencyBackend,
    backend_prefill_len,
    rounded_prefill_len,
)
from TokenSim.latency.roofline import RooflineLatencyBackend
from TokenSim.llm.llm_request import Request


logger = logging.getLogger(__name__)


class LLMCompassLatencyBackend(LatencyBackend):
    def __init__(
        self,
        fallback_backend: RooflineLatencyBackend,
        wrapped_llmcompass_vars: tuple[Any, Any, Any],
    ):
        self.fallback_backend = fallback_backend
        self.llmcm_system, self.llmcm_prefill, self.llmcm_decode = (
            wrapped_llmcompass_vars
        )
        self.mems = [{}, {}, {}, {}]

    def estimate_step_latency(self, requests: list[Request]) -> float:
        if all(
            getattr(req, "needs_recompute", False) and req.recompute_tokens == 0
            for req in requests
        ):
            return 0.0
        try:
            return self._estimate_llmcompass(requests)
        except Exception:
            logger.exception("LLMCompass failed; falling back to roofline")
            return self.fallback_backend.estimate_step_latency(requests)

    def _estimate_llmcompass(self, requests: list[Request]) -> float:
        from LLMCompass.software_model.utils import Tensor, data_type_dict

        batch_size = len(requests)
        sum_attn_latency = 0
        proj_latency = 0

        if requests[0].is_prefill or getattr(requests[0], "needs_recompute", False):
            total_prefill_len = sum(
                (
                    req.recompute_tokens
                    if getattr(req, "needs_recompute", False)
                    else req.prefill_compute_len
                )
                for req in requests
            )
            total_prefill_len = rounded_prefill_len(total_prefill_len)
            if total_prefill_len in self.mems[0]:
                proj_latency = self.mems[0][total_prefill_len]
            else:
                _ = self.llmcm_prefill(
                    Tensor([1, total_prefill_len, 4096], data_type_dict["fp16"])
                )
                proj_latency, _ = self.llmcm_prefill.compile_and_simulate_proj_attn(
                    self.llmcm_system, "heuristic-GPU"
                )
                self.mems[0][total_prefill_len] = proj_latency
            proj_latency *= LLMCOMPASS_LAYER_NUM
            for req in requests:
                prefill_len = backend_prefill_len(
                    req.recompute_tokens
                    if getattr(req, "needs_recompute", False)
                    else req.prefill_compute_len
                )
                if prefill_len in self.mems[1]:
                    attn = self.mems[1][prefill_len]
                else:
                    _ = self.llmcm_prefill(
                        Tensor([1, prefill_len, 4096], data_type_dict["fp16"])
                    )
                    _, attn = self.llmcm_prefill.compile_and_simulate_proj_attn(
                        self.llmcm_system, "heuristic-GPU"
                    )
                    self.mems[1][prefill_len] = attn
                sum_attn_latency += attn * LLMCOMPASS_LAYER_NUM
        else:
            max_generation_idx = max([req.generation_idx for req in requests])
            if (batch_size, max_generation_idx) in self.mems[2]:
                proj_latency = self.mems[2][(batch_size, max_generation_idx)]
            else:
                _ = self.llmcm_decode(
                    Tensor([batch_size, 1, 4096], data_type_dict["fp16"]),
                    max_generation_idx,
                )
                proj_latency, _ = self.llmcm_decode.compile_and_simulate_proj_attn(
                    self.llmcm_system, "heuristic-GPU"
                )
                self.mems[2][(batch_size, max_generation_idx)] = proj_latency
            proj_latency *= LLMCOMPASS_LAYER_NUM
            for req in requests:
                context_len = req.prefill_len + req.generation_idx
                if context_len in self.mems[3]:
                    attn = self.mems[3][context_len]
                else:
                    _ = self.llmcm_decode(
                        Tensor([1, 1, 4096], data_type_dict["fp16"]),
                        context_len,
                    )
                    _, attn = self.llmcm_decode.compile_and_simulate_proj_attn(
                        self.llmcm_system, "heuristic-GPU"
                    )
                    self.mems[3][context_len] = attn
                sum_attn_latency += attn * LLMCOMPASS_LAYER_NUM

        return proj_latency + sum_attn_latency
