from typing import Any

from TransformerRoofline import TransformerRoofline

from TokenSim.config.constants import _GB
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.moe.config import MoEModelConfig
from TokenSim.moe.placement import ExpertPlacement


class CacheConfig:
    def __init__(
        self,
        block_size: int,
        hardware: str,
        model: str,
        roofline: TransformerRoofline,
        parallel_config: ParallelConfig | None = None,
        rank_info: ParallelRankInfo | None = None,
        moe_config: MoEModelConfig | None = None,
        expert_placement: ExpertPlacement | None = None,
    ):
        # block size
        self.block_size: int = block_size
        self.model: str = model
        self.parallel_config = parallel_config or ParallelConfig.default()
        self.rank_info = rank_info or ParallelRankInfo()
        self.moe_config = moe_config or MoEModelConfig()
        self.expert_placement = expert_placement
        hardware_conf = roofline.hardwares[hardware]
        model_conf = roofline.models[model]
        self.total_num_layers = model_conf.Nlayer
        self.num_layers_per_rank = stage_layer_count(
            model_conf.Nlayer,
            self.parallel_config.pipeline_parallel_size,
            self.rank_info.pp_rank,
        )
        self.num_kv_heads = num_kv_heads(model_conf)
        self.local_kv_heads = local_kv_heads(
            self.num_kv_heads,
            self.parallel_config.tensor_parallel_size,
        )
        self.head_dim = model_conf.Dmodel / model_conf.Nhead
        self.size_per_token_unsharded = model_conf.Dmodel * 2 * 2 * model_conf.Nlayer
        self.size_per_token = int(
            self.local_kv_heads
            * self.head_dim
            * 2
            * 2
            * self.num_layers_per_rank
        )

        self.model_param_size_unsharded = self._estimate_unsharded_model_params(
            model_conf,
        )
        self.model_param_size = self._estimate_rank_model_params(model_conf)
        # num blocks
        # FIXME: actual host memory size can reach terrabytes, much larger than accelerator memory
        # however in our application, host memory serves as swap space, and leveraging large swap space is not desirable for performance
        #  so the number is set small so out simulation exits early
        gpu_memory_bytes = hardware_conf.MM_Card_Num * hardware_conf.Capacity * _GB
        if gpu_memory_bytes <= self.model_param_size:
            raise ConfigurationError(
                f"model {model!r} does not fit on hardware {hardware!r}: "
                + f"model_param_size={self.model_param_size}, gpu_memory={gpu_memory_bytes}"
            )
        self.num_cpu_blocks: int = int(
            min(32768, 32768 * _GB / self.size_per_token // self.block_size)
        )
        self.num_gpu_blocks: int = (
            (gpu_memory_bytes - self.model_param_size)
            / self.size_per_token
            // self.block_size
        )

    def _estimate_unsharded_model_params(self, model_conf) -> float:
        if not self.moe_config.enabled:
            return (
                12 * model_conf.Nlayer * model_conf.Dmodel * model_conf.Dmodel
                + 50000 * model_conf.Dmodel
            ) * 2
        dense_layers = model_conf.Nlayer - self.moe_config.num_moe_layers
        hidden = self.moe_config.hidden_size or model_conf.Dmodel
        dense_ffn = self.moe_config.intermediate_size or getattr(
            model_conf,
            "FFN_Hidden",
            4 * hidden,
        )
        moe_ffn = self.moe_config.moe_intermediate_size or dense_ffn
        dense_params = (
            4 * model_conf.Nlayer * hidden * hidden
            + 2 * dense_layers * hidden * dense_ffn
            + 50000 * hidden
        )
        routed_expert_params = (
            2
            * self.moe_config.num_moe_layers
            * self.moe_config.num_experts
            * hidden
            * moe_ffn
        )
        shared_expert_params = (
            2
            * self.moe_config.num_moe_layers
            * self.moe_config.num_shared_experts
            * hidden
            * moe_ffn
        )
        return (dense_params + routed_expert_params + shared_expert_params) * 2

    def _estimate_rank_model_params(self, model_conf) -> float:
        if not self.moe_config.enabled or self.expert_placement is None:
            return self.model_param_size_unsharded / (
                self.parallel_config.tensor_parallel_size
                * self.parallel_config.pipeline_parallel_size
            )
        hidden = self.moe_config.hidden_size or model_conf.Dmodel
        dense_ffn = self.moe_config.intermediate_size or getattr(
            model_conf,
            "FFN_Hidden",
            4 * hidden,
        )
        moe_ffn = self.moe_config.moe_intermediate_size or dense_ffn
        owned_moe_layers = len(
            self.expert_placement.moe_layers_for_pp_rank(self.rank_info.pp_rank)
        )
        total_moe_layers_in_stage = owned_moe_layers
        dense_layers_in_stage = max(0, self.num_layers_per_rank - total_moe_layers_in_stage)
        dense_params = (
            4 * self.num_layers_per_rank * hidden * hidden
            + 2 * dense_layers_in_stage * hidden * dense_ffn
            + (50000 * hidden / self.parallel_config.pipeline_parallel_size)
        )
        shared_params = (
            2
            * owned_moe_layers
            * self.moe_config.num_shared_experts
            * hidden
            * moe_ffn
        )
        owned_experts = self.expert_placement.experts_for_rank(self.rank_info)
        routed_params = (
            2 * owned_moe_layers * len(owned_experts) * hidden * moe_ffn
        )
        return (
            dense_params + shared_params + routed_params
        ) * 2 / self.parallel_config.tensor_parallel_size


def stage_layer_count(total_layers: int, pp_size: int, pp_rank: int) -> int:
    base = total_layers // pp_size
    remainder = total_layers % pp_size
    return base + (1 if pp_rank < remainder else 0)


# MHA/GQA/MQA
def num_kv_heads(model_conf: Any) -> int:
    if getattr(model_conf, "Multi_Query", False):
        return 1
    if getattr(model_conf, "Grouped_Query", False):
        grouped_num = int(getattr(model_conf, "Grouped_Num", 1))
        if grouped_num < 1:
            raise ConfigurationError("Grouped_Num must be at least 1")
        if model_conf.Nhead % grouped_num != 0:
            raise ConfigurationError(
                f"Nhead={model_conf.Nhead} is not divisible by Grouped_Num={grouped_num}"
            )
        return int(model_conf.Nhead // grouped_num)
    return int(model_conf.Nhead)


def local_kv_heads(kv_heads: int, tensor_parallel_size: int) -> int:
    if kv_heads % tensor_parallel_size != 0:
        raise ConfigurationError(
            f"KV heads {kv_heads} are not divisible by tensor_parallel_size "
            + f"{tensor_parallel_size}"
        )
    return kv_heads // tensor_parallel_size
