"""Translate command-line values into validated simulation configurations."""

from __future__ import annotations

import argparse
from dataclasses import fields
from TokenSim.config.config import ClusterConfig, ParallelConfig
from TokenSim.config.model_config import ModelSpec
from TokenSim.config.simulation import SimulationConfig
from TokenSim.errors import ConfigurationError


def simulation_config(args):
    defaults = SimulationConfig()
    names = {"latency_backend_type": "latency_backend"}
    values = {
        field.name: getattr(
            args, names.get(field.name, field.name), getattr(defaults, field.name)
        )
        for field in fields(defaults)
    }
    values["max_num_batched_tokens"] = max_num_batched_tokens(args)
    values["moe_execution"] = {
        "backend": args.moe_backend,
        "shared_expert_fusion": not args.disable_moe_shared_fusion,
    }
    return SimulationConfig(**values)


def max_num_batched_tokens(args: argparse.Namespace) -> int | None:
    """Per-step token budget of the paged-attention scheduler; ``None`` disables chunking."""
    if not args.chunked_prefill:
        return None
    return args.max_num_batched_tokens


def build_parallel_config(
    args: argparse.Namespace,
    cluster: ClusterConfig,
) -> ParallelConfig:
    base = cluster.effective_parallel_config()
    return base.override(
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        data_parallel_rank=args.data_parallel_rank,
        data_parallel_size_local=args.data_parallel_size_local,
        enable_expert_parallel=args.enable_expert_parallel,
        expert_parallel_scope=args.expert_parallel_scope,
        expert_parallel_size=args.expert_parallel_size,
        expert_placement_strategy=args.expert_placement_strategy,
        all2all_backend=args.all2all_backend,
    )


def validate_moe_parallel_config(
    model: ModelSpec,
    parallel_config: ParallelConfig,
) -> None:
    if parallel_config.enable_expert_parallel and not model.is_moe:
        raise ConfigurationError("enable_expert_parallel requires a MoE model config")
