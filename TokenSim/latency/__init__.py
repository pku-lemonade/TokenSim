from typing import Any

from TokenSim.config.config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.latency.base import LatencyBackend, backend_prefill_len
from TokenSim.latency.llmcompass import LLMCompassLatencyBackend
from TokenSim.latency.roofline import RooflineLatencyBackend
from TokenSim.parallel import ParallelCommunicator
from TokenSim.moe.config import MoEModelConfig
from TokenSim.moe.placement import ExpertPlacement


def build_latency_backend(
    backend_type: str,
    roofline: Any,
    model: str,
    hardware: str,
    parallel_config: ParallelConfig | None = None,
    rank_info: ParallelRankInfo | None = None,
    communicator: ParallelCommunicator | None = None,
    moe_config: MoEModelConfig | None = None,
    expert_placement: ExpertPlacement | None = None,
    random_seed: int = 0,
    wrapped_llmcompass_vars: tuple[Any, Any, Any] | None = None,
) -> LatencyBackend:
    roofline_backend = RooflineLatencyBackend(
        roofline=roofline,
        model=model,
        hardware=hardware,
        parallel_config=parallel_config,
        rank_info=rank_info,
        communicator=communicator,
        moe_config=moe_config,
        expert_placement=expert_placement,
        random_seed=random_seed,
    )
    if backend_type == "roofline":
        return roofline_backend
    if backend_type == "llm_compass":
        if wrapped_llmcompass_vars is None:
            raise ConfigurationError(
                "latency_backend='llm_compass' requires wrapped LLMCompass variables"
            )
        return LLMCompassLatencyBackend(
            fallback_backend=roofline_backend,
            wrapped_llmcompass_vars=wrapped_llmcompass_vars,
        )
    raise ConfigurationError(f"unsupported latency backend {backend_type!r}")


__all__ = [
    "LLMCompassLatencyBackend",
    "LatencyBackend",
    "RooflineLatencyBackend",
    "backend_prefill_len",
    "build_latency_backend",
]
