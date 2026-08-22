from __future__ import annotations

import copy
from typing import Any

from TokenSim.config.config import ParallelConfig, ParallelRankInfo, _GB
from TokenSim.latency.base import (
    DECODE_SCALE,
    PREFILL_OFFSET_SECONDS,
    PREFILL_SCALE,
    LatencyBackend,
    backend_prefill_len,
    rounded_prefill_len,
)
from TokenSim.llm.llm_request import Request
from TokenSim.moe.config import MoEModelConfig
from TokenSim.moe.placement import ExpertPlacement
from TokenSim.moe.routing import ExpertRouting
from TokenSim.moe.stats import MoEStats
from TokenSim.parallel import ParallelCommunicator


class RooflineLatencyBackend(LatencyBackend):
    # Compute_Timebreakdown_Iteration is deterministic in its arguments once
    # the backend registered its model/hardware adaptations at init; memoize
    # per backend instance with a hard cap (cleared wholesale on overflow).
    _ROOFLINE_CACHE_LIMIT = 131072

    def __init__(
        self,
        roofline: Any,
        model: str,
        hardware: str,
        parallel_config: ParallelConfig | None = None,
        rank_info: ParallelRankInfo | None = None,
        communicator: ParallelCommunicator | None = None,
        moe_config: MoEModelConfig | None = None,
        expert_placement: ExpertPlacement | None = None,
        random_seed: int = 0,
    ):
        self.roofline = roofline
        self.model = model
        self.hardware = hardware
        self.parallel_config = parallel_config or ParallelConfig.default()
        self.rank_info = rank_info or ParallelRankInfo()
        self.communicator = communicator
        self.roofline_hardware = self._roofline_hardware_name()
        self.moe_config = moe_config or MoEModelConfig()
        self.roofline_model = self._roofline_model_name()
        self.expert_placement = expert_placement
        self.expert_routing = ExpertRouting(self.moe_config, seed=random_seed)
        self._expert_to_rank: dict[int, int] | None = None
        self._roofline_cache: dict[tuple[int, int, int], tuple[float, float]] = {}
        self.moe_stats = MoEStats(
            effective_moe_config=self.moe_config.to_dict(),
            expert_placement=(
                self.expert_placement.to_dict() if self.expert_placement else None
            ),
        )

    def estimate_step_latency(self, requests: list[Request]) -> float:
        if all(
            getattr(req, "needs_recompute", False) and req.recompute_tokens == 0
            for req in requests
        ):
            return 0.0
        batch_size = len(requests)
        sum_attn_latency = 0
        proj_latency = 0

        is_context_build = requests[0].is_prefill or getattr(
            requests[0], "needs_recompute", False
        )
        if is_context_build:
            total_prefill_len = sum(
                (
                    req.recompute_tokens
                    if getattr(req, "needs_recompute", False)
                    else req.prefill_compute_len
                )
                for req in requests
            )
            total_prefill_len = rounded_prefill_len(total_prefill_len)
            # Prefill projection is calibrated as one packed context. Batch effects are
            # represented by summing prefill tokens, while per-request attention is added below.
            packed_prefill_batch_size = 1
            proj_latency, _ = self._timebreakdown(
                total_prefill_len,
                0,
                packed_prefill_batch_size,
            )
        else:
            proj_latency, _ = self._timebreakdown(
                requests[0].prefill_len,
                requests[0].generation_idx,
                batch_size,
            )

        for req in requests:
            prefill_len = (
                req.recompute_tokens
                if getattr(req, "needs_recompute", False)
                else req.prefill_compute_len if req.is_prefill else req.prefill_len
            )
            _, attn_latency = self._timebreakdown(
                backend_prefill_len(prefill_len),
                0 if getattr(req, "needs_recompute", False) else req.generation_idx,
                # TODO: check the num "1" here
                1,
            )
            sum_attn_latency += attn_latency

        proj_latency, sum_attn_latency = self._apply_tensor_parallel_conversion(
            proj_latency,
            sum_attn_latency,
        )
        total_time = proj_latency + sum_attn_latency
        total_time += self._moe_step_latency(requests)
        total_time += self._parallel_sync_latency(requests)
        if is_context_build:
            return total_time * PREFILL_SCALE + PREFILL_OFFSET_SECONDS
        return total_time * DECODE_SCALE

    def _timebreakdown(
        self,
        prompt_len: int,
        step: int,
        batch_size: int,
    ) -> tuple[float, float]:
        key = (prompt_len, step, batch_size)
        cached = self._roofline_cache.get(key)
        if cached is None:
            cached = self.roofline.Compute_Timebreakdown_Iteration(
                prompt_len,
                step,
                batch_size,
                self.roofline_model,
                self.roofline_hardware,
                Pipeline_Stage=self.parallel_config.pipeline_parallel_size,
            )
            if len(self._roofline_cache) >= self._ROOFLINE_CACHE_LIMIT:
                self._roofline_cache.clear()
            self._roofline_cache[key] = cached
        return cached

    def _roofline_hardware_name(self) -> str:
        if (
            self.parallel_config.tensor_parallel_size == 1
            and self.parallel_config.pipeline_parallel_size == 1
        ):
            return self.hardware
        hardwares = getattr(self.roofline, "hardwares", None)
        if hardwares is None or self.hardware not in hardwares:
            return self.hardware
        name = (
            f"{self.hardware}__tokensim_parallel_"
            + f"tp{self.parallel_config.tensor_parallel_size}_"
            + f"pp{self.parallel_config.pipeline_parallel_size}"
        )
        if name in hardwares:
            return name

        parallel_hardware = copy.copy(hardwares[self.hardware])
        parallel_hardware.Name = name
        # TransformerRoofline can consume Pipeline_Stage but has no TokenSim TP
        # topology input. Keep its card count just large enough for PP, then
        # convert TP compute and collectives in this adapter.
        card_count = self.parallel_config.pipeline_parallel_size
        for attr in ("MM_Card_Num", "MM_GP_Card_Num", "MV_Card_Num", "num"):
            setattr(parallel_hardware, attr, card_count)
        hardwares[name] = parallel_hardware
        return name

    def _roofline_model_name(self) -> str:
        if not self.moe_config.enabled:
            return self.model
        models = getattr(self.roofline, "models", None)
        if models is None or self.model not in models:
            return self.model
        name = f"{self.model}__tokensim_dense_base"
        if name in models:
            return name
        dense_model = copy.copy(models[self.model])
        dense_model.Name = name
        dense_model.FFN_MOE = False
        dense_model.MOE_Quantity = 1
        dense_model.MOE_Activate = 1
        if self.moe_config.intermediate_size is not None:
            dense_model.FFN_Hidden = self.moe_config.intermediate_size
        models[name] = dense_model
        return name

    # divide the computation to each tensor parallel rank, and the communication cost is added in _parallel_sync_latency
    def _apply_tensor_parallel_conversion(
        self,
        proj_latency: float,
        attn_latency: float,
    ) -> tuple[float, float]:
        tp_size = self.parallel_config.tensor_parallel_size
        if tp_size <= 1:
            return proj_latency, attn_latency
        # TransformerRoofline has no TokenSim ParallelConfig input. Convert dense
        # transformer TP locally: QKV/FFN1 are column-sharded, W0/FFN2 are
        # row-sharded, and attention work follows local head ownership.
        return proj_latency / tp_size, attn_latency / tp_size

    # estimate the latency of parallel synchronization
    # (e.g., all-reduce for TP, stage transfer for PP) after each step, which is not captured in the roofline model
    def _parallel_sync_latency(self, requests: list[Request]) -> float:
        if self.communicator is None:
            return 0.0
        model_config = self.roofline.models[self.model]
        tokens = self._step_token_count(requests)
        hidden_bytes = int(tokens * model_config.Dmodel * 2)
        latency = 0.0
        # TP
        if self.parallel_config.tensor_parallel_size > 1:
            self.communicator.record_roofline_conversion()
            # Row-parallel W0 and FFN2 both synchronize the hidden state.
            latency += self.communicator.estimate_tp_collective(hidden_bytes)  # W0
            latency += self.communicator.estimate_tp_collective(hidden_bytes)  # FFN2
        # PP
        if self.parallel_config.pipeline_parallel_size > 1:
            latency += self.communicator.estimate_pp_stage_transfer(hidden_bytes)
        return latency

    def _moe_step_latency(self, requests: list[Request]) -> float:
        if not self.moe_config.enabled or self.expert_placement is None:
            return 0.0
        owned_layers = self.expert_placement.moe_layers_for_pp_rank(
            self.rank_info.pp_rank
        )
        if not owned_layers:
            return 0.0
        histogram = self.expert_routing.histogram_for_step(requests)
        if not histogram:
            return 0.0
        per_rank_load = self._per_rank_load(histogram)
        local_load = per_rank_load.get(
            self.expert_placement.ep_rank_for_rank_info(self.rank_info),
            0,
        )
        if local_load == 0:
            return 0.0
        compute_latency = self._estimate_moe_compute_latency(
            local_load,
            len(owned_layers),
        )
        all2all_latency = self._estimate_moe_all2all_latency(histogram, requests)
        straggler_latency = self._estimate_moe_straggler_latency(
            per_rank_load,
            compute_latency,
        )
        self.moe_stats.record_step(
            histogram=histogram,
            per_rank_load=per_rank_load,
            compute_latency=compute_latency,
            all2all_latency=all2all_latency,
            straggler_latency=straggler_latency,
            all2all_event_count=1 if all2all_latency > 0 else 0,
        )
        return compute_latency + all2all_latency + straggler_latency

    def _per_rank_load(self, histogram: dict[int, int]) -> dict[int, int]:
        if self.expert_placement is None:
            return {}
        if self._expert_to_rank is None:
            expert_to_rank: dict[int, int] = {}
            for ep_rank, experts in self.expert_placement.rank_to_experts.items():
                for expert_id in experts:
                    expert_to_rank[expert_id] = ep_rank
            self._expert_to_rank = expert_to_rank
        per_rank: dict[int, int] = {}
        for expert_id, count in histogram.items():
            ep_rank = self._expert_to_rank.get(expert_id)
            if ep_rank is None:
                continue
            per_rank[ep_rank] = per_rank.get(ep_rank, 0) + count
        return per_rank

    def _estimate_moe_compute_latency(self, routed_load: int, moe_layers: int) -> float:
        model_config = self.roofline.models[self.model]
        hidden_size = self.moe_config.hidden_size or model_config.Dmodel
        intermediate_size = (
            self.moe_config.moe_intermediate_size
            or self.moe_config.intermediate_size
            or getattr(model_config, "FFN_Hidden", 4 * hidden_size)
        )
        flops = routed_load * moe_layers * hidden_size * intermediate_size * 2 * 2
        hardware_config = self.roofline.hardwares[self.roofline_hardware]
        tflops = max(1e-9, getattr(hardware_config, "MM_TFLOPS", 1.0))
        return flops / (tflops * 1e12)

    def _estimate_moe_all2all_latency(
        self,
        histogram: dict[int, int],
        requests: list[Request],
    ) -> float:
        if (
            not self.parallel_config.enable_expert_parallel
            or self.expert_placement is None
            or self.expert_placement.ep_rank_count <= 1
        ):
            return 0.0
        model_config = self.roofline.models[self.model]
        hidden_size = self.moe_config.hidden_size or model_config.Dmodel
        tokens = self._step_token_count(requests)
        bytes_ = int(tokens * hidden_size * 2 * 2)
        if bytes_ <= 0:
            return 0.0
        base_latency = (
            self.communicator.estimate_ep_all2all(bytes_)
            if self.communicator is not None
            else 0.0
        )
        if base_latency == 0.0:
            base_latency = self._local_link_latency(bytes_)
        return base_latency * self._all2all_backend_scale()

    def _estimate_moe_straggler_latency(
        self,
        per_rank_load: dict[int, int],
        compute_latency: float,
    ) -> float:
        if not per_rank_load:
            return 0.0
        mean_load = sum(per_rank_load.values()) / len(per_rank_load)
        if mean_load <= 0:
            return 0.0
        max_load = max(per_rank_load.values())
        if max_load <= mean_load:
            return 0.0
        imbalance_fraction = (max_load - mean_load) / mean_load
        return compute_latency * imbalance_fraction

    def _local_link_latency(self, bytes_: int) -> float:
        links = getattr(self.roofline, "links", {})
        hardwares = getattr(self.roofline, "hardwares", {})
        hardware = hardwares.get(self.roofline_hardware) or hardwares.get(self.hardware)
        link_name = getattr(hardware, "Nvlink", None) or getattr(hardware, "Pcie", None)
        if link_name in links:
            link = links[link_name]
            return link.Latency + bytes_ / _GB / link.UniBW
        return 0.0

    def _all2all_backend_scale(self) -> float:
        backend = self.parallel_config.all2all_backend
        return {
            "naive": 1.5,
            "allgather_reducescatter": 1.0,
            "deepep_high_throughput": 0.7,
            "deepep_low_latency": 0.5,
        }.get(backend, 1.0)

    @staticmethod
    def _step_token_count(requests: list[Request]) -> int:
        if getattr(requests[0], "needs_recompute", False):
            return max(1, sum(req.recompute_tokens for req in requests))
        if requests[0].is_prefill:
            return max(1, sum(req.prefill_compute_len for req in requests))
        return len(requests)
