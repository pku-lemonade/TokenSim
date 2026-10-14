"""Executor."""

from __future__ import annotations

import math
from typing import Any

import simpy

from TokenSim.af.caching.estimates import FrozenDict
from TokenSim.af.caching.snapshots import ConfigurationOwner, snapshot
from TokenSim.af.calibration.host import HostOverheadProfile, runtime_fingerprint
from TokenSim.af.calibration.operation import OperationServiceProfile
from TokenSim.af.calibration.routing import RoutingReplay
from TokenSim.af.communication.resources import AFResourceRegistry
from TokenSim.af.execution.communication import ExecutorCommunicationMixin
from TokenSim.af.execution.compute import ExecutorComputeMixin
from TokenSim.af.execution.control import NativeControlMixin
from TokenSim.af.execution.host import ExecutorHostMixin
from TokenSim.af.execution.iteration import ExecutorIterationMixin
from TokenSim.af.execution.native_order import NativeOrderMixin
from TokenSim.af.execution.stages import ExecutorResourcesMixin
from TokenSim.af.execution.submission import ExecutorSubmissionMixin
from TokenSim.af.execution.transport import ExecutorTransportMixin
from TokenSim.af.planning.batches import AFBlock, AFIteration
from TokenSim.af.planning.batching import AFBlockBuilder
from TokenSim.af.planning.planner import ParallelExecutionPlanner
from TokenSim.af.planning.replicas import IndependentRoundRobinAFReplicaSelector
from TokenSim.af.providers.queries import (
    AFComputeLatencyProvider,
    AFServiceEstimate,
    AFTransferLatencyProvider,
    AF_MATCH_TYPES,
)
from TokenSim.af.runtime.plan import AFRuntimePlan
from TokenSim.af.trace.recorder import AFTraceRecorder
from TokenSim.config.cluster.core import ClusterConfig
from TokenSim.config.cluster.workers import WorkerConfig
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.moe.routing import ExpertRouting


class AFExecutor(
    ExecutorHostMixin,
    ExecutorSubmissionMixin,
    ExecutorIterationMixin,
    ExecutorComputeMixin,
    ExecutorCommunicationMixin,
    ExecutorTransportMixin,
    ExecutorResourcesMixin,
    NativeOrderMixin,
    NativeControlMixin,
    ConfigurationOwner,
):
    _configuration_fields = (
        "cluster_config",
        "runtime_plan",
        "model_id",
        "dtype",
        "total_layers",
        "hidden_size",
    )

    def __init__(
        self,
        env: simpy.Environment,
        cluster_config: ClusterConfig,
        worker_configs: list[WorkerConfig],
        compute_provider: AFComputeLatencyProvider | None,
        transfer_provider: AFTransferLatencyProvider | None,
        model_id: str,
        dtype: str,
        total_layers: int,
        hidden_size: int,
        resources: AFResourceRegistry | None = None,
        trace_recorder: AFTraceRecorder | None = None,
        moe_config: Any | None = None,
        random_seed: int = 0,
        runtime_plan: AFRuntimePlan | None = None,
        transfer_readiness=None,
    ) -> None:
        cluster_config = snapshot(cluster_config)
        # Compiled AFRuntimePlan is already a graph of frozen records/tuples.
        # Share it; copying the entire ownership graph adds startup work only.
        self._validate_runtime(
            cluster_config,
            compute_provider,
            transfer_provider,
            model_id,
            dtype,
            total_layers,
            hidden_size,
        )
        self.env = env
        self.cluster_config = cluster_config
        self.compute_provider = compute_provider
        self.transfer_provider = transfer_provider
        self.model_id = model_id
        self.dtype = (
            runtime_plan.effective_dtypes.activation_dtype
            if runtime_plan is not None
            else dtype
        )
        self.total_layers = total_layers
        self.hidden_size = hidden_size
        self.runtime_plan = runtime_plan
        self.overhead_profile = None
        if cluster_config.af.engine_overhead_profile is not None:
            self.overhead_profile = HostOverheadProfile(
                cluster_config.af.engine_overhead_profile,
                expected_runtime=runtime_fingerprint(
                    runtime_plan,
                    cluster_config,
                    worker_configs,
                    getattr(compute_provider, "hardware", None),
                ),
                expected_environment=cluster_config.af.calibration_environment,
                explicit_costs=cluster_config.af.communication_timing._explicit_cost_fields,
            )
            if self.overhead_profile.document["schema_version"] == 4:
                source = self.overhead_profile.document["device_service_source_id"]
                hardware = getattr(compute_provider, "hardware", None)
                if (
                    getattr(compute_provider, "fallback", None) != "analytical_only"
                    or hardware is None
                    or any(
                        hardware.device(w.hardware).analytical_source_id != source
                        for w in worker_configs
                    )
                ):
                    raise ConfigurationError(
                        "CPU API submission costs require matching device-only calibration; old launch assumptions conflict"
                    )
            if any(
                name in self.overhead_profile.document["costs"]
                for name in ("engine_output_process_us", "engine_runner_bookkeeping_us")
            ) and not getattr(self, "supports_sampled_token_readback", False):
                raise ConfigurationError(
                    "engine output processing requires the whole-model executor"
                )
        self.operation_profile = None
        if cluster_config.af.operation_service_profile is not None:
            self.operation_profile = OperationServiceProfile(
                cluster_config.af.operation_service_profile,
                expected_runtime=runtime_fingerprint(
                    runtime_plan,
                    cluster_config,
                    worker_configs,
                    getattr(compute_provider, "hardware", None),
                ),
                expected_environment=cluster_config.af.calibration_environment,
                explicit_costs=cluster_config.af.communication_timing._explicit_cost_fields,
            )
            if (
                "sampled_token_readback"
                in self.operation_profile.document["operations"]
            ):
                if not getattr(self, "supports_sampled_token_readback", False):
                    raise ConfigurationError(
                        "sampled_token_readback requires the whole-model executor"
                    )
                if getattr(compute_provider, "fallback", None) != "analytical_only":
                    raise ConfigurationError(
                        "sampled_token_readback requires audited analytical_only compute coverage"
                    )
            if any(
                name in self.operation_profile.document["operations"]
                for name in ("a_input_prepare", "a_attention_metadata")
            ):
                if getattr(compute_provider, "fallback", None) != "analytical_only":
                    raise ConfigurationError(
                        "a_input_prepare requires audited analytical_only compute coverage"
                    )
                if (
                    self.overhead_profile is not None
                    and "a_step_prepare_us" in self.overhead_profile.document["costs"]
                ):
                    raise ConfigurationError(
                        "duplicate operation/host profile coverage: a_input_prepare"
                    )
            if (
                "f_dp_metadata" in self.operation_profile.document["operations"]
                and self.overhead_profile is not None
                and "f_step_prepare_us" in self.overhead_profile.document["costs"]
            ):
                raise ConfigurationError(
                    "duplicate operation/host profile coverage: f_dp_metadata"
                )
        timing = cluster_config.af.communication_timing
        self.routing_replay = None
        if cluster_config.af.routing_replay is not None:
            self.routing_replay = RoutingReplay(
                cluster_config.af.routing_replay,
                expected_runtime=runtime_fingerprint(
                    runtime_plan,
                    cluster_config,
                    worker_configs,
                    getattr(compute_provider, "hardware", None),
                ),
                expected_environment=cluster_config.af.calibration_environment,
            )
        if runtime_plan is None and (
            timing.a_step_prepare_us
            or timing.f_step_prepare_us
            or timing.native_control_enabled
        ):
            raise ConfigurationError(
                "fixed step preparation requires a compiled AF runtime plan"
            )
        # This callback supplies invocation events, not cached service times.
        self.transfer_readiness = transfer_readiness
        self._validate_provider_dtype_contract()
        self.resources = resources or AFResourceRegistry(
            env,
            cluster_config,
            worker_configs,
            getattr(transfer_provider, "placement", None),
        )
        self.trace = trace_recorder or AFTraceRecorder(
            cluster_config.af.trace_enabled,
            self.resources.resource_capacities(),
        )
        self.builder = AFBlockBuilder()
        self.replica_selector = IndependentRoundRobinAFReplicaSelector()
        self.expert_routing = (
            ExpertRouting(moe_config, seed=random_seed)
            if moe_config is not None and moe_config.enabled
            else None
        )
        self.plan_builder = (
            ParallelExecutionPlanner(cluster_config, runtime_plan=runtime_plan)
            if runtime_plan is not None
            else None
        )
        self._links = {link.name: link for link in cluster_config.af.links}
        self._next_iteration_id = 0
        self.active_iterations: dict[int, AFIteration] = {}
        self._fused_combines = {}
        self.active_stages: dict[tuple[int, int, str], dict[str, object]] = {}
        self._compute_resource_bindings = {}
        self._communication_resource_bindings = {}
        self._endpoint_worker_bindings = {}
        self._effective_dtype_bindings = {}
        self._query_mappings = {}
        self._step_preparation = {}
        self._native_controls = {}

    def _effective_dtypes(self, link_name: str) -> dict[str, str]:
        cached = self._effective_dtype_bindings.get(link_name)
        if cached is not None:
            return cached
        if self.runtime_plan is None:
            result = FrozenDict(
                {
                    "weight_dtype": self.dtype,
                    "activation_dtype": self.dtype,
                    "kv_cache_dtype": self.dtype,
                    "a_to_f_dtype": self.dtype,
                    "f_to_a_dtype": self.dtype,
                }
            )
            self._effective_dtype_bindings[link_name] = result
            return result
        effective = self.runtime_plan.dtypes_for_link(link_name)
        result = FrozenDict(
            {
                "weight_dtype": effective.weight_dtype,
                "activation_dtype": effective.activation_dtype,
                "kv_cache_dtype": effective.kv_cache_dtype,
                "a_to_f_dtype": effective.a_to_f_dtype,
                "f_to_a_dtype": effective.f_to_a_dtype,
            }
        )
        self._effective_dtype_bindings[link_name] = result
        return result

    def _query_mapping(self, items):
        key = tuple((name, type(value), value) for name, value in items)
        result = self._query_mappings.get(key)
        if result is None:
            result = FrozenDict(items)
            if len(self._query_mappings) >= 4096:
                self._query_mappings.pop(next(iter(self._query_mappings)))
            self._query_mappings[key] = result
        return result

    def _validate_provider_dtype_contract(self) -> None:
        if self.runtime_plan is None:
            return
        validator = getattr(self.compute_provider, "validate_effective_dtypes", None)
        if not callable(validator):
            return
        contracts = [self.runtime_plan.effective_dtypes]
        contracts.extend(
            effective
            for _, effective in self.runtime_plan.link_effective_dtypes
            if effective not in contracts
        )
        try:
            for effective in contracts:
                validator(self.model_id, effective)
        except ValueError as exc:
            raise ConfigurationError(
                f"AF provider dtype contract validation failed: {exc}"
            ) from exc

    @staticmethod
    def _validate_runtime(
        cluster_config: ClusterConfig,
        compute_provider: AFComputeLatencyProvider | None,
        transfer_provider: AFTransferLatencyProvider | None,
        model_id: str,
        dtype: str,
        total_layers: int,
        hidden_size: int,
    ) -> None:
        if not cluster_config.af.enabled:
            raise ConfigurationError("AFExecutor requires af.enabled=true")
        missing = []
        if compute_provider is None:
            missing.append("compute latency provider")
        if transfer_provider is None:
            missing.append("transfer latency provider")
        if not isinstance(model_id, str) or not model_id:
            missing.append("model_id")
        if not isinstance(dtype, str) or not dtype:
            missing.append("dtype")
        if type(total_layers) is not int or total_layers < 1:
            missing.append("positive total_layers")
        if type(hidden_size) is not int or hidden_size < 1:
            missing.append("positive hidden_size")
        if missing:
            raise ConfigurationError(
                "AF runtime is missing required dependencies: " + ", ".join(missing)
            )

    @staticmethod
    def _estimate(provider, query, block, layer_id, stage) -> AFServiceEstimate:
        try:
            estimate = provider.estimate(query)
        except Exception as exc:
            raise SimulationStateError(
                f"AF estimate unsupported for iteration={block.iteration_id}, "
                f"block={block.block_id}, layer={layer_id}, stage={stage}, "
                f"model={getattr(query, 'model_id', '?')}, "
                f"domain={getattr(query, 'source_domain', '?')}, "
                f"rank={getattr(query, 'tp_rank', '?')}, "
                f"ep_rank={getattr(query, 'ep_rank', '?')}, "
                f"operation={getattr(query, 'operation', '?')}: {exc}"
            ) from exc
        if not isinstance(estimate, AFServiceEstimate):
            raise SimulationStateError(
                f"AF provider returned {type(estimate).__name__} for "
                f"iteration={block.iteration_id}, block={block.block_id}, "
                f"layer={layer_id}, stage={stage}; expected AFServiceEstimate"
            )
        duration = estimate.service_duration
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            valid = False
        else:
            valid = math.isfinite(duration) and duration >= 0
        if not valid:
            raise SimulationStateError(
                f"AF provider returned invalid service_duration={duration!r} for "
                f"iteration={block.iteration_id}, block={block.block_id}, "
                f"layer={layer_id}, stage={stage}"
            )
        if not isinstance(estimate.source_id, str) or not estimate.source_id:
            raise SimulationStateError(
                f"AF provider returned an empty source_id for iteration="
                f"{block.iteration_id}, block={block.block_id}, layer={layer_id}, "
                f"stage={stage}"
            )
        if (
            not isinstance(estimate.dataset_version, str)
            or not estimate.dataset_version
        ):
            raise SimulationStateError(
                f"AF provider returned an empty dataset_version for iteration="
                f"{block.iteration_id}, block={block.block_id}, layer={layer_id}, "
                f"stage={stage}"
            )
        if estimate.match_type not in AF_MATCH_TYPES:
            raise SimulationStateError(
                f"AF provider returned invalid match_type={estimate.match_type!r} "
                f"for iteration={block.iteration_id}, block={block.block_id}, "
                f"layer={layer_id}, stage={stage}"
            )
        return estimate

    def _set_active_stage(
        self,
        block: AFBlock,
        layer_id: int,
        stage: str,
        ready_at: float,
        start_at: float | None = None,
    ) -> None:
        node_id = stage if ":" in stage else f"legacy:{stage}"
        display_stage = stage
        if ":" in stage:
            if ":router" in stage:
                display_stage = "router"
            elif ":expert:" in stage:
                display_stage = "expert"
            elif ":a:" in stage:
                display_stage = "a"
        self.active_stages[(block.iteration_id, block.block_id, node_id)] = {
            "iteration_id": block.iteration_id,
            "block_id": block.block_id,
            "layer_id": layer_id,
            "stage": display_stage,
            "node_id": node_id,
            "ready_at": ready_at,
            "start_at": start_at,
        }

    def unfinished_snapshot(self) -> list[dict[str, object]]:
        snapshots: list[dict[str, object]] = []
        for iteration in self.active_iterations.values():
            snapshots.append(
                {
                    "iteration_id": iteration.iteration_id,
                    "phase": iteration.phase,
                    "block_ids": [block.block_id for block in iteration.blocks],
                    "completed_block_ids": sorted(iteration.block_completed_at),
                    "active_stages": [
                        dict(stage)
                        for key, stage in self.active_stages.items()
                        if key[0] == iteration.iteration_id
                    ],
                }
            )
        return snapshots
