"""Public imports for the simulation engine and worker runtime."""

from TokenSim.llm.engine import (
    LLMEngine as _LLMEngine,
    LLMWorker,
    Message,
    Task,
    Worker,
    WorkerStatus,
    RequestCompletionDebugPrinter,
)
from TokenSim.llm.engine.af_worker import AFComputeWorker
from TokenSim.llm.engine.messages import _merge_connector_metadata, _request_ids


class LLMEngine(_LLMEngine):
    """Engine accepting the current config API and existing AF call sites."""

    def __init__(self, *args, psla_config=None, **kwargs):
        if psla_config is not None:
            from TokenSim.config.simulation import SimulationConfig
            from TokenSim.config.workload_config import WorkloadConfig

            hardware = kwargs["hardware"]
            kwargs["model_spec"] = hardware.model(psla_config.model).with_moe_override(
                psla_config.moe_config
            )
            kwargs.setdefault(
                "workload_config",
                WorkloadConfig(
                    **{
                        name: getattr(psla_config, name)
                        for name in WorkloadConfig.__dataclass_fields__
                    }
                ),
            )
            values = {
                name: kwargs.pop(name)
                for name in SimulationConfig.__dataclass_fields__
                if name in kwargs
            }
            if "max_parallem_sum" in kwargs:
                values["max_num_seqs"] = kwargs.pop("max_parallem_sum") or 64
            kwargs.setdefault("config", SimulationConfig(**values))
        super().__init__(*args, **kwargs)
        if psla_config is not None:
            from TokenSim.llm.llm_request import g_time
            from TokenSim.timing import TimingRecorder

            self.timing_recorder = TimingRecorder(g_time)
