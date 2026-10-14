"""Compatibility imports for AF workload drivers."""

from TokenSim.workload.source import LLMSource, get_requests as _get_requests


def get_requests(args, model_config, block_size, tqdm_submit_func=None):
    from TokenSim.config.workload_config import WorkloadConfig

    config = WorkloadConfig(
        **{
            name: getattr(model_config, name)
            for name in WorkloadConfig.__dataclass_fields__
        }
    )
    return _get_requests(
        args, config, block_size, tqdm_submit_func, num_experts=model_config.num_experts
    )
