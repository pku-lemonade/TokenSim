from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from collections.abc import Sequence
from typing import Any

import simpy

from TokenSim.af.snapshot import ConfigurationOwner, snapshot
from TokenSim.af.estimate_cache import FrozenDict, prepare_query

from TokenSim.af.builder import AFBlockBuilder, phase_for_requests
from TokenSim.af.latency import (
    AF_MATCH_TYPES,
    AFComputeLatencyProvider,
    AFComputeQuery,
    AFServiceEstimate,
    AFTransferLatencyProvider,
    AFTransferQuery,
)
from TokenSim.af.models import AFBlock, AFIteration, AFPhase
from TokenSim.af.planning import (
    AFExecutionDAG,
    ExecutionNode,
    IndependentRoundRobinAFReplicaSelector,
    ParallelExecutionPlanner,
)
from TokenSim.af.resources import AFResourceRegistry
from TokenSim.af.runtime import AFRuntimePlan
from TokenSim.af.readiness import TransferReadiness
from TokenSim.af.control import NativeControlMixin
from TokenSim.af.native_order import NativeOrderMixin
from TokenSim.af.calibration import (
    HostOverheadProfile,
    runtime_fingerprint,
    COMPUTE_CALL_FIELDS,
    MESSAGE_CALL_FIELDS,
)
from TokenSim.af.routing_replay import RoutingReplay
from TokenSim.af.operation_profile import OperationServiceProfile
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.af.trace import AFTraceEvent, AFTraceRecorder
from TokenSim.config.cluster_config import AFLinkConfig, ClusterConfig, WorkerConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.moe.routing import ExpertRouting


class AFExecutor(NativeOrderMixin, NativeControlMixin, ConfigurationOwner):
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
        self.plan_builder = ParallelExecutionPlanner(
            cluster_config,
            hidden_size=hidden_size,
            dtype_size_bytes=dtype_size_bytes(self.dtype, role="activation_dtype"),
            num_experts=(moe_config.num_experts if moe_config is not None else 0),
            top_k=(moe_config.num_experts_per_tok if moe_config is not None else 1),
            runtime_plan=runtime_plan,
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

    def _host_cost_us(self, name, block=None, *, tokens=None):
        fallback = getattr(self.cluster_config.af.communication_timing, name, 0.0)
        if self.overhead_profile is None:
            return fallback
        count = tokens if tokens is not None else block.token_count
        return self.overhead_profile.cost(name, count, fallback)

    def _native_call_cost_field(self, step, action):
        if action == "compute":
            if (
                step.operation == "router_topk"
                and self.overhead_profile is not None
                and "compute_f_moe_block_us" in self.overhead_profile.document["costs"]
            ):
                return "compute_f_moe_block_us"
            field = COMPUTE_CALL_FIELDS.get(step.operation)
            fallback = "compute_submit_us"
        else:
            backend = step.metadata.get("transport_backend", "pynccl")
            field = MESSAGE_CALL_FIELDS.get((backend, step.direction, action))
            fallback = (
                "host_submit_us" if action == "send" else "host_receive_submit_us"
            )
        if (
            self.overhead_profile is not None
            and field in self.overhead_profile.document["costs"]
        ):
            return field
        return fallback

    def host_cost_provenance(self, name, *, tokens):
        """Materialized per-call cost, distinct from whole-engine validation."""
        profile = self.overhead_profile
        covered = profile is not None and name in profile.document["costs"]
        explicit = (
            name in self.cluster_config.af.communication_timing._explicit_cost_fields
        )
        measured = covered and profile.document["evidence"]["kind"] == "measured"
        return dict(
            value_us=self._host_cost_us(name, tokens=tokens),
            unit="us",
            calibrated=measured,
            origin=(
                "measured_profile"
                if measured
                else (
                    "synthetic_test_profile"
                    if covered
                    else "manual_assumption" if explicit else "unmeasured_default"
                )
            ),
            profile_id=profile.document["profile_id"] if covered else None,
            profile_fingerprint=profile.fingerprint if covered else None,
            coverage=(
                list(profile.document["costs"][name]["coverage"]) if covered else []
            ),
            validation_status=(
                profile.document["evidence"]["validation_status"] if covered else None
            ),
        )

    def _start_step_preparation(self, iteration):
        """One independent host preparation per participating worker/iteration."""
        timing = self.cluster_config.af.communication_timing
        preparations = {}
        a_cost = self._host_cost_us(
            "a_step_prepare_us", tokens=len(iteration.request_ids)
        )
        f_cost = self._host_cost_us(
            "f_step_prepare_us", tokens=len(iteration.request_ids)
        )
        schedule_cost = self._host_cost_us(
            "engine_step_schedule_us", tokens=len(iteration.request_ids)
        )
        input_operations = tuple(
            name
            for name in ("a_input_prepare", "a_attention_metadata")
            if self.operation_profile is not None
            and name in self.operation_profile.document["operations"]
        )
        input_operation = bool(input_operations)
        if input_operation:
            a_cost = sum(
                self.operation_profile.document["operations"][name]["fixed_us"]
                for name in input_operations
            )
        if not timing.enabled or not (
            a_cost or f_cost or input_operation or schedule_cost
        ):
            return preparations
        for block in iteration.blocks:
            for domain, instance, duration, field in (
                (
                    block.placement.a_domain,
                    block.placement.a_instance,
                    a_cost,
                    "a_step_prepare_us",
                ),
                (
                    block.placement.f_domain,
                    block.placement.f_instance,
                    f_cost,
                    "f_step_prepare_us",
                ),
            ):
                if not duration and not (
                    (input_operation or schedule_cost) and field == "a_step_prepare_us"
                ):
                    continue
                for binding in self.runtime_plan.bindings_for_instance(
                    domain, instance
                ):
                    identity = binding.physical_identity
                    if identity in preparations:
                        continue
                    node_id = f"i{iteration.iteration_id}:prepare:{identity}"
                    if (
                        input_operation or schedule_cost
                    ) and field == "a_step_prepare_us":
                        process = self.env.process(
                            self._run_a_step_preparation(
                                block, identity, node_id, duration
                            )
                        )
                        preparations[identity] = (process, node_id)
                        continue
                    step = SimpleNamespace(
                        operation="step_prepare",
                        direction=None,
                        logical_stage_id=None,
                        leg_id=None,
                        source_domain=domain,
                        target_domain=domain,
                        layer_type=None,
                    )
                    process = self.env.process(
                        self._run_preparation_stage(
                            block,
                            -1,
                            step,
                            node_id,
                            (),
                            (identity,),
                            duration / 1e6,
                            "host_prepare",
                            details={
                                "cost_field": field,
                                "cost_scope": "per_step_per_worker",
                                "calibrated": False,
                            },
                        )
                    )
                    preparations[identity] = (process, node_id)
        return preparations

    def _run_a_step_preparation(self, block, identity, node_id, duration):
        try:
            predecessors = ()
            schedule = self._host_cost_us("engine_step_schedule_us", block)
            if schedule:
                scheduled = node_id + ":schedule"
                template = self._control_nodes(block)[0]
                yield from self._run_communication_host_stage(
                    block,
                    -1,
                    template,
                    scheduled,
                    (),
                    (identity,),
                    schedule / 1e6,
                    "host_prepare",
                    details={
                        "cost_field": "engine_step_schedule_us",
                        "cost_scope": "per_step_per_worker",
                    },
                )
                predecessors = (scheduled,)
            if self.operation_profile is not None and any(
                n in self.operation_profile.document["operations"]
                for n in ("a_input_prepare", "a_attention_metadata")
            ):
                yield from self._run_input_preparation(
                    block, identity, node_id, predecessors
                )
            else:
                yield from self._run_communication_host_stage(
                    block,
                    -1,
                    self._control_nodes(block)[0],
                    node_id,
                    predecessors,
                    (identity,),
                    duration / 1e6,
                    "host_prepare",
                    details={
                        "cost_field": "a_step_prepare_us",
                        "cost_scope": "per_step_per_worker",
                    },
                )
        except simpy.Interrupt:
            return

    def _run_input_preparation(self, block, identity, node_id, predecessors=()):
        try:
            names = [
                name
                for name in ("a_input_prepare", "a_attention_metadata")
                if name in self.operation_profile.document["operations"]
            ]
            for index, name in enumerate(names):
                completed = yield from self._control_stage(
                    block,
                    self._control_nodes(block)[0],
                    name,
                    predecessors,
                    identity,
                    0,
                    device=True,
                    node_id=(
                        node_id if index == len(names) - 1 else node_id + ":" + name
                    ),
                )
                predecessors = (completed,)
        except simpy.Interrupt:
            return

    def _run_preparation_stage(self, *args, **kwargs):
        try:
            yield from self._run_communication_host_stage(*args, **kwargs)
        except simpy.Interrupt:
            return

    def _run_receiver_submit(
        self,
        block,
        layer_id,
        step,
        node_id,
        duration,
        cost_field="host_receive_submit_us",
    ):
        try:
            prepared = yield from self._wait_step_preparation(
                block, (step.physical_workers[-1],)
            )
            control = self._native_controls.get(block.iteration_id)
            if (
                control is not None
                and step.direction == "a_to_f"
                and step.metadata.get("message_class") != "control"
            ):
                yield control["f_ready"]
                prepared = (*prepared, control["f_node"])
            yield from self._run_communication_host_stage(
                block,
                layer_id,
                step,
                node_id,
                tuple(dict.fromkeys((*step.predecessor_ids, *prepared))),
                (step.physical_workers[-1],),
                duration,
                "host_submit",
                details={
                    "cost_field": cost_field,
                    "cost_scope": "per_receive_call",
                    "calibrated": False,
                },
            )
        except simpy.Interrupt:
            return

    def _wait_step_preparation(self, block, identities):
        preparations = self._step_preparation.get(block.iteration_id, {})
        selected = [preparations[x] for x in identities if x in preparations]
        if selected:
            yield self.env.all_of([event for event, _ in selected])
        return tuple(node_id for _, node_id in selected)

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

    def submit(
        self,
        source_a_worker_id: int,
        requests: Sequence[Any],
        phase: AFPhase | None = None,
    ) -> AFIteration:
        resolved_phase = phase or phase_for_requests(requests)
        if self.operation_profile is not None:
            self.operation_profile.validate_workload(requests, resolved_phase)
        if self.overhead_profile is not None:
            self.overhead_profile.validate_workload(requests, resolved_phase)
        if resolved_phase not in self.cluster_config.af.phases:
            raise ConfigurationError(
                f"AF phase {resolved_phase!r} is not selected by af.phases"
            )
        link = self.cluster_config.phase_to_af_link[resolved_phase]
        source_group = self.resources.group_for_worker(source_a_worker_id)
        if source_group != link.attention_group:
            raise ConfigurationError(
                f"AF phase {resolved_phase!r} requires attention group "
                f"{link.attention_group!r}, but worker {source_a_worker_id} belongs to "
                f"{source_group!r}"
            )
        source_worker = self.resources.worker(source_a_worker_id)
        if source_worker.physical_worker_ref is None:
            raise ConfigurationError(
                f"AF source worker {source_a_worker_id} has no physical identity"
            )
        source_identity = source_worker.physical_worker_ref.identity
        a_domain, f_domain = self.cluster_config.phase_to_af_domains[resolved_phase]
        a_instance = self.plan_builder.instance_for_physical(a_domain, source_identity)
        owner_key = a_domain.name
        for request in requests:
            owners = getattr(request, "af_a_owners", None)
            if owners is None:
                owners = {}
                request.af_a_owners = owners
            previous = owners.get(owner_key)
            if previous is not None and previous != a_instance.name:
                raise ConfigurationError(
                    f"request {request.id} AF {resolved_phase} owner is "
                    f"{previous!r}, cannot move to {a_instance.name!r} without KV transfer"
                )
            owners[owner_key] = a_instance.name
        iteration_id = self._next_iteration_id
        self._next_iteration_id += 1
        runtime_layer_types = self._runtime_layer_types()
        formal_routing = (
            self.runtime_plan is not None
            and (
                resolved_phase == "decode"
                or self.runtime_plan.layers[0].mla is not None
            )
            and all(
                not domain.name.startswith("implicit:")
                for domain in self.cluster_config.af.execution_domains
            )
        )
        layer_types = runtime_layer_types if formal_routing else None

        def placement_factory(_block_id, _block_requests):
            selection = self.replica_selector.select_for_preselected_a(
                resolved_phase,
                a_domain,
                f_domain,
                a_instance,
            )
            placement = self.plan_builder.placement(
                resolved_phase, source_identity, selection.f_service
            )
            if placement.a_instance != selection.a_owner.name:
                raise ConfigurationError(
                    "AF scheduler owner changed while selecting the F service "
                    f"replica: expected={selection.a_owner.name!r}, "
                    f"actual={placement.a_instance!r}"
                )
            return placement

        def histogram_factory(block_requests):
            if self.expert_routing is None:
                return {}
            return self.expert_routing.histogram_for_step(
                list(block_requests),
                phase=resolved_phase,
                decode_step=self._routing_step(resolved_phase, block_requests),
            )

        def routing_snapshot_factory(
            snapshot_iteration_id,
            snapshot_phase,
            layer_id,
            block_requests,
            placement,
        ):
            if self.expert_routing is None:
                raise ConfigurationError(
                    f"AF runtime layer {layer_id} is MoE but no ExpertRouting "
                    "configuration is available"
                )
            if self.routing_replay is not None:
                block_requests = self.routing_replay.requests_for(
                    block_requests,
                    phase=snapshot_phase,
                    layer_id=layer_id,
                    decode_step=self._routing_step(snapshot_phase, block_requests),
                )
            return self.expert_routing.snapshot_for_layer(
                block_requests,
                iteration_id=snapshot_iteration_id,
                phase=snapshot_phase,
                decode_step=self._routing_step(snapshot_phase, block_requests),
                layer_id=layer_id,
                a_dp_rank=placement.a_dp_rank,
                a_tp_rank=self._canonical_a_tp_rank(placement),
            )

        return self.builder.build(
            iteration_id=iteration_id,
            requests=requests,
            af_block_size=self.cluster_config.af.af_block_size,
            source_a_worker_id=source_a_worker_id,
            link=link,
            phase=resolved_phase,
            placement_factory=placement_factory,
            histogram_factory=None if formal_routing else histogram_factory,
            layer_types=layer_types,
            routing_snapshot_factory=(
                routing_snapshot_factory
                if formal_routing and "moe" in layer_types.values()
                else None
            ),
        )

    def _runtime_layer_types(self) -> dict[int, str] | None:
        if self.runtime_plan is None:
            return None
        layer_types = {
            layer.layer_id: layer.layer_type for layer in self.runtime_plan.layers
        }
        expected_ids = set(range(self.total_layers))
        if set(layer_types) != expected_ids:
            raise ConfigurationError(
                "AF runtime layer metadata must cover every executor layer once: "
                f"expected={sorted(expected_ids)}, actual={sorted(layer_types)}"
            )
        return layer_types

    @staticmethod
    def _routing_step(phase: AFPhase, requests: Sequence[Any]) -> int:
        if phase == "prefill":
            return 0
        steps = {int(getattr(request, "generation_idx", 0)) for request in requests}
        if not steps or min(steps) < 0:
            raise ConfigurationError(
                f"AF Decode steps must be non-negative; got {sorted(steps)}"
            )
        return min(steps)

    def _canonical_a_tp_rank(self, placement) -> int:
        ownerships = self.runtime_plan.ownerships_for_replica(
            placement.a_domain,
            placement.a_dp_rank,
        )
        canonical = tuple(item for item in ownerships if item.canonical_producer)
        if len(canonical) != 1:
            raise ConfigurationError(
                "AF runtime A replica must have exactly one canonical routing producer"
            )
        return canonical[0].tp_rank

    def run_iteration(self, iteration: AFIteration):
        if iteration.iteration_id in self.active_iterations:
            raise SimulationStateError(
                f"AF iteration {iteration.iteration_id} is already active"
            )
        native_control = (
            self.cluster_config.af.communication_timing.native_control_enabled
        )
        if native_control and self.active_iterations:
            raise SimulationStateError(
                "native control eager mode supports one in-flight iteration"
            )
        if native_control and len(iteration.blocks) > 1:
            raise ConfigurationError(
                "native control eager DP=1 requires one unsplit block per iteration"
            )
        iteration.start_at = self.env.now
        self.active_iterations[iteration.iteration_id] = iteration
        self._step_preparation[iteration.iteration_id] = self._start_step_preparation(
            iteration
        )
        failure_event = self.env.event()
        failures: list[BaseException] = []
        done_events = [self.env.event() for _ in iteration.blocks]
        processes = [
            self.env.process(
                self._run_block_guarded(block, failure_event, failures, done_event)
            )
            for block, done_event in zip(iteration.blocks, done_events)
        ]
        if native_control and iteration.blocks:
            state = {
                "a_ready": self.env.event(),
                "f_ready": self.env.event(),
                "a_node": f"i{iteration.iteration_id}:control:payload:transport",
                "f_node": f"i{iteration.iteration_id}:control:decode",
            }
            self._native_controls[iteration.iteration_id] = state
            if self.cluster_config.af.communication_timing.native_call_order_enabled:
                self._start_native_call_order(iteration.blocks[0], state)
            control_done = self.env.event()
            done_events.append(control_done)
            processes.append(
                self.env.process(
                    self._run_control_guarded(
                        iteration.blocks[0],
                        state,
                        failure_event,
                        failures,
                        control_done,
                    )
                )
            )
        try:
            if processes:
                completion = self.env.all_of(done_events)
                yield self.env.any_of((completion, failure_event))
                if failures:
                    for process in processes:
                        if process.is_alive:
                            process.interrupt(
                                f"AF iteration {iteration.iteration_id} failed"
                            )
                    yield self.env.all_of(done_events)
                    exc = failures[0]
                    if isinstance(exc, SimulationStateError):
                        raise exc
                    raise SimulationStateError(
                        f"AF iteration {iteration.iteration_id} failed: {exc}"
                    ) from exc
        except simpy.Interrupt:
            for process in processes:
                if process.is_alive:
                    process.interrupt(
                        f"AF iteration {iteration.iteration_id} cancelled"
                    )
            if processes:
                yield self.env.all_of(done_events)
            raise
        finally:
            state = self._native_controls.pop(iteration.iteration_id, None)
            receivers = (
                [call["receiver"] for call in state.get("calls", {}).values()]
                if state
                else []
            )
            for receiver in receivers:
                if receiver.is_alive:
                    receiver.interrupt("native iteration finished or cancelled")
            if receivers:
                yield self.env.all_of(receivers)
            self._fused_combines.pop(iteration.iteration_id, None)
            preparations = self._step_preparation.pop(iteration.iteration_id, {})
            for process, _ in preparations.values():
                if process.is_alive:
                    process.interrupt("AF iteration preparation cancelled")
            if preparations:
                yield self.env.all_of([process for process, _ in preparations.values()])
            self.active_iterations.pop(iteration.iteration_id, None)
            for key in tuple(self.active_stages):
                if key[0] == iteration.iteration_id:
                    self.active_stages.pop(key, None)

        iteration.end_at = self.env.now
        iteration.barrier_wait = {
            block_id: iteration.end_at - completed_at
            for block_id, completed_at in iteration.block_completed_at.items()
        }
        if self.trace.enabled:
            self.trace.validate_invariants(iteration.iteration_id)
        return iteration

    def _run_control_guarded(self, block, state, failure_event, failures, done):
        try:
            yield from self._run_native_control(block, state)
        except simpy.Interrupt:
            return
        except BaseException as exc:
            failures.append(exc)
            if not failure_event.triggered:
                failure_event.succeed()
        finally:
            if not done.triggered:
                done.succeed()

    def _run_block_guarded(
        self,
        block: AFBlock,
        failure_event: simpy.Event,
        failures: list[BaseException],
        done_event: simpy.Event,
    ):
        try:
            control = self._native_controls.get(block.iteration_id)
            if control is not None:
                yield self.env.any_of((control["a_ready"], failure_event))
                if failure_event.triggered:
                    return
            yield from self._run_block(block)
        except simpy.Interrupt:
            return
        except BaseException as exc:
            failures.append(exc)
            if not failure_event.triggered:
                failure_event.succeed()
            return
        finally:
            if not done_event.triggered:
                done_event.succeed()

    def _run_block(self, block: AFBlock):
        link = self._links[block.link_name]
        for layer_id in range(self.total_layers):
            dag = self.plan_builder.build_layer(block, layer_id, link)
            yield from self._run_dag(block, layer_id, link, dag)
        iteration = self.active_iterations[block.iteration_id]
        iteration.block_completed_at[block.block_id] = self.env.now

    def _run_dag(
        self,
        block: AFBlock,
        layer_id: int,
        link: AFLinkConfig,
        dag: AFExecutionDAG,
    ):
        processes: list[simpy.Process] = []
        done_events: list[simpy.Event] = []
        failure_event = self.env.event()
        failures: list[BaseException] = []
        if hasattr(dag, "predecessor_indices"):
            predecessor_indices = dag.predecessor_indices
            terminal_indices = dag.terminal_indices
        else:
            indices = {node.node_id: i for i, node in enumerate(dag.nodes)}
            predecessor_indices = tuple(
                tuple(indices[p] for p in node.predecessor_ids) for node in dag.nodes
            )
            terminal_indices = tuple(indices[p] for p in dag.terminal_node_ids)
        for index, node in enumerate(dag.nodes):
            done_event = self.env.event()
            done_events.append(done_event)
            predecessors = tuple(
                done_events[item] for item in predecessor_indices[index]
            )
            bindings = getattr(node, "input_producer_ids", ())
            subgroup_inputs = None
            if (
                bindings
                and self.cluster_config.af.transfer_topology_mode == "physical"
                and hasattr(self.transfer_provider, "communication_plan")
            ):
                events_by_id = dict(zip(node.predecessor_ids, predecessors))
                subgroup_inputs = {
                    endpoint: (producer, events_by_id[producer])
                    for endpoint, producer in bindings
                }
                data_ids = {producer for _, producer in bindings}
                predecessors = tuple(
                    event
                    for producer, event in events_by_id.items()
                    if producer not in data_ids
                )
            processes.append(
                self.env.process(
                    self._run_dag_node_guarded(
                        block,
                        layer_id,
                        link,
                        node,
                        predecessors,
                        failure_event,
                        failures,
                        done_event,
                        subgroup_inputs,
                    )
                )
            )
        # Tensor-parallel A execution may produce one terminal combine node
        # per rank.  The DAG makespan is defined by the last terminal, so wait
        # for all terminal nodes instead of requiring a single sink.
        terminal_done = self.env.all_of(
            tuple(done_events[index] for index in terminal_indices)
        )
        try:
            yield self.env.any_of((terminal_done, failure_event))
            if failures:
                for process in processes:
                    if process.is_alive:
                        process.interrupt(
                            f"AF DAG failed at iteration={block.iteration_id}, "
                            f"block={block.block_id}, layer={layer_id}"
                        )
                yield self.env.all_of(tuple(done_events))
                raise failures[0]
        except simpy.Interrupt:
            for process in processes:
                if process.is_alive:
                    process.interrupt("AF model block cancelled")
            yield self.env.all_of(tuple(done_events))
            raise

    def _run_dag_node_guarded(
        self,
        block: AFBlock,
        layer_id: int,
        link: AFLinkConfig,
        node: ExecutionNode,
        predecessors: tuple[simpy.Event, ...],
        failure_event: simpy.Event,
        failures: list[BaseException],
        done_event: simpy.Event,
        subgroup_inputs=None,
    ):
        try:
            if predecessors:
                predecessor_done = self.env.all_of(predecessors)
                yield self.env.any_of((predecessor_done, failure_event))
                if failure_event.triggered:
                    return
            if failure_event.triggered:
                return
            if subgroup_inputs is not None:
                yield from self._run_planned_communication(
                    block,
                    layer_id,
                    link,
                    node,
                    input_readiness=(subgroup_inputs, failure_event),
                )
            else:
                yield from self._run_planned_step(block, layer_id, link, node)
        except simpy.Interrupt:
            return
        except BaseException as exc:
            error = SimulationStateError(
                f"AF DAG node failed for iteration={block.iteration_id}, "
                f"block={block.block_id}, layer={layer_id}, node={node.node_id}: {exc}"
            )
            error.__cause__ = exc
            failures.append(error)
            if not failure_event.triggered:
                failure_event.succeed()
        finally:
            if not done_event.triggered:
                done_event.succeed()

    def _run_planned_step(
        self,
        block: AFBlock,
        layer_id: int,
        link: AFLinkConfig,
        step: ExecutionNode,
    ):
        if step.kind == "compute":
            yield from self._run_planned_compute(block, layer_id, step)
        else:
            yield from self._run_planned_communication(block, layer_id, link, step)

    def _run_planned_compute(
        self,
        block: AFBlock,
        layer_id: int,
        step: ExecutionNode,
    ):
        preparation_ids = yield from self._wait_step_preparation(
            block, step.physical_workers
        )
        control = self._native_controls.get(block.iteration_id)
        if control is not None and not step.predecessor_ids:
            preparation_ids = (*preparation_ids, control["a_node"])
        timing = self.cluster_config.af.communication_timing
        compute_cost_field = self._native_call_cost_field(step, "compute")
        compute_submit_us = self._host_cost_us(compute_cost_field, block)
        if (
            step.operation in ("routed_expert", "f_weighted_combine")
            and self.overhead_profile is not None
            and "compute_f_moe_block_us" in self.overhead_profile.document["costs"]
        ):
            compute_submit_us = (
                0  # One native FusedMoE call owns router, experts and combine dispatch.
            )
        if (
            step.operation == "f_weighted_combine"
            and self.overhead_profile is not None
            and "compute_f_expert_group_us" in self.overhead_profile.document["costs"]
        ):
            compute_submit_us = (
                0  # Native grouped invocation includes combine dispatch.
            )
        combine_key = (block.block_id, layer_id, step.physical_workers)
        fused = (
            self._fused_combines.get(block.iteration_id, {}).get(combine_key)
            if step.stage == "weighted_combine"
            else None
        )
        if fused is not None:
            compute_submit_us = 0
        if timing.enabled and compute_submit_us:
            submit_id = f"{step.node_id}:compute_submit"
            yield from self._run_communication_host_stage(
                block,
                layer_id,
                step,
                submit_id,
                tuple(dict.fromkeys((*step.predecessor_ids, *preparation_ids))),
                step.physical_workers,
                compute_submit_us / 1e6,
                "host_submit",
                details={
                    "cost_field": compute_cost_field,
                    "cost_scope": (
                        "per_native_call_group"
                        if compute_cost_field != "compute_submit_us"
                        else "per_compute_dag_node"
                    ),
                    "calibrated": False,
                },
            )
            step = replace(step, predecessor_ids=(submit_id,))
        elif preparation_ids and not step.predecessor_ids:
            step = replace(step, predecessor_ids=preparation_ids)
        if (
            control is not None
            and "a_device_ready" in control
            and control["a_node"] in preparation_ids
        ):
            yield control["a_device_ready"]
            step = replace(
                step,
                predecessor_ids=tuple(
                    dict.fromkeys(
                        (
                            *step.predecessor_ids,
                            f"i{block.iteration_id}:control:payload:transport",
                        )
                    )
                ),
            )
        if self.runtime_plan is not None:
            self.runtime_plan.validate_execution_node(step)
        resources = self._compute_resource_bindings.get(step.physical_workers)
        if resources is None:
            resources = self.resources.compute_resources_for(step.physical_workers)
            self._compute_resource_bindings[step.physical_workers] = resources
        worker_id = resources[0][0]
        worker = self.resources.worker(worker_id)
        local_work = step.local_work
        if local_work is None:
            raise SimulationStateError(f"compute node {step.node_id} has no local work")
        effective_dtypes = self._effective_dtypes(block.link_name)
        query = prepare_query(
            self.compute_provider,
            AFComputeQuery,
            iteration_id=block.iteration_id,
            block_id=block.block_id,
            layer_id=layer_id,
            phase=block.phase,
            stage=step.stage,
            token_count=local_work.token_count,
            context_lengths=block.context_lengths,
            model_id=self.model_id,
            dtype=effective_dtypes["activation_dtype"],
            device_type=worker.hardware,
            worker_id=worker_id,
            operation=step.operation,
            source_domain=step.source_domain,
            dp_rank=step.dp_rank,
            tp_rank=step.tp_rank or 0,
            ep_rank=step.ep_rank or 0,
            logical_participant=step.logical_participants[0],
            physical_worker=step.physical_workers[0],
            local_shape=self._query_mapping(local_work.tensor_shape),
            local_token_count=local_work.token_count,
            expert_ids=local_work.expert_ids,
            expert_workload=local_work.expert_workload,
            tp_group=step.tp_group,
            ep_group=step.ep_group,
            layer_type=step.layer_type,
            original_token_count=local_work.original_token_count,
            input_edge_count=local_work.input_edge_count,
            output_token_count=local_work.output_token_count,
            weight_dtype=effective_dtypes["weight_dtype"],
            activation_dtype=effective_dtypes["activation_dtype"],
            kv_cache_dtype=effective_dtypes["kv_cache_dtype"],
            a_to_f_dtype=effective_dtypes["a_to_f_dtype"],
            f_to_a_dtype=effective_dtypes["f_to_a_dtype"],
            logical_stage_id=step.logical_stage_id,
            tensor_slices=step.tensor_slices,
            owner_endpoint=local_work.owner_endpoint,
            routing_edges=local_work.routing_edges,
            source_dtype=local_work.source_dtype,
            target_dtype=local_work.target_dtype,
            assignment_count=local_work.assignment_count,
        )
        stage_name = step.stage
        ready_at, start_at, end_at, estimate = yield from self._run_resources_stage(
            block,
            layer_id,
            step.node_id,
            tuple(resource for _, resource in resources),
            lambda: (
                AFServiceEstimate(
                    0,
                    fused.source_id,
                    dataset_version=fused.dataset_version,
                    match_type=fused.match_type,
                    details={
                        "timing_scope": "device_service",
                        "covered_by": "routed_experts",
                        "semantic_completion_only": True,
                    },
                )
                if fused is not None
                else self._estimate(
                    self.compute_provider, query, block, layer_id, stage_name
                )
            ),
        )
        if (
            self.cluster_config.af.execution_protocol == "vllm_afd"
            and step.stage == "routed_expert"
            and "weighted_combine" in estimate.details.get("covered_operations", ())
        ):
            self._fused_combines.setdefault(block.iteration_id, {})[
                combine_key
            ] = estimate
        if self.trace.enabled:
            resource_ids = tuple(f"worker:{item[0]}" for item in resources)
            self.trace.record(
                AFTraceEvent(
                    iteration_id=block.iteration_id,
                    block_id=block.block_id,
                    layer_id=layer_id,
                    phase=block.phase,
                    event_type="compute",
                    stage=step.stage,
                    operation=step.operation,
                    resource_id=resource_ids[0],
                    resource_ids=resource_ids,
                    worker_id=worker_id,
                    ready_at=ready_at,
                    start_at=start_at,
                    end_at=end_at,
                    queue_duration=start_at - ready_at,
                    service_duration=end_at - start_at,
                    request_count=block.request_count,
                    token_count=block.token_count,
                    source_id=estimate.source_id,
                    dataset_version=estimate.dataset_version,
                    match_type=estimate.match_type,
                    details={
                        **estimate.details,
                        "execution_protocol": self.cluster_config.af.execution_protocol,
                        **(
                            {"routing_replay": self.routing_replay.describe()}
                            if self.routing_replay is not None
                            and step.stage == "router"
                            else {}
                        ),
                    },
                    source_domain=step.source_domain,
                    target_domain=step.target_domain,
                    logical_participants=step.logical_participants,
                    participant_ranks=step.domain_ranks,
                    physical_endpoints=step.physical_workers,
                    node_id=step.node_id,
                    predecessor_ids=step.predecessor_ids,
                    dp_rank=step.dp_rank,
                    tp_rank=step.tp_rank,
                    ep_rank=step.ep_rank,
                    tp_group=step.tp_group,
                    ep_group=step.ep_group,
                    local_work=local_work.as_dict(),
                    layer_type=step.layer_type,
                    original_token_count=local_work.original_token_count,
                    assignment_count=local_work.assignment_count,
                    input_edge_count=local_work.input_edge_count,
                    output_token_count=local_work.output_token_count,
                    logical_stage_id=step.logical_stage_id,
                    source_endpoint=step.source_endpoint,
                    target_endpoint=step.target_endpoint,
                    tensor_slices=step.tensor_slices,
                    source_layout=step.source_layout,
                    target_layout=step.target_layout,
                    routing_edges=step.routing_edges,
                    source_dtype=local_work.source_dtype,
                    target_dtype=local_work.target_dtype,
                )
            )
        self.active_stages.pop((block.iteration_id, block.block_id, step.node_id), None)

    def _run_planned_communication(
        self,
        block: AFBlock,
        layer_id: int,
        link: AFLinkConfig,
        step: ExecutionNode,
        transport_completed=None,
        input_readiness=None,
    ):
        is_control = step.metadata.get("message_class") == "control"
        binding_key = (
            link.name,
            step.direction,
            step.resource_group,
            step.resource_lane,
            (
                step.physical_workers
                if self.cluster_config.af.transfer_topology_mode == "physical"
                else ()
            ),
        )
        binding = self._communication_resource_bindings.get(binding_key)
        if binding is None:
            keyed_resources = self.resources.communication_resources_for_step(
                link,
                step.direction,
                step.resource_group,
                step.resource_lane,
                step.physical_workers,
            )
            resource_ids = tuple(
                self.resources.format_link_resource(key) for key, _ in keyed_resources
            )
            binding = (keyed_resources, resource_ids)
            self._communication_resource_bindings[binding_key] = binding
        keyed_resources, resource_ids = binding
        input_predecessors = (
            tuple(producer for producer, _ in input_readiness[0].values())
            if input_readiness
            else ()
        )
        if step.operation == "all_to_all" and (
            step.payload_bytes == 0 or len(step.physical_workers) == 1
        ):
            keyed_resources, resource_ids = (), ()
        endpoint_workers = self._endpoint_worker_bindings.get(step.physical_workers)
        if endpoint_workers is None:
            endpoint_workers = tuple(
                self.resources.worker(self.resources.worker_id_for_physical(identity))
                for identity in step.physical_workers
            )
            self._endpoint_worker_bindings[step.physical_workers] = endpoint_workers
        source_device = endpoint_workers[0].hardware
        target_device = endpoint_workers[-1].hardware
        effective_dtypes = self._effective_dtypes(block.link_name)
        transfer_dtype = (
            effective_dtypes["a_to_f_dtype"]
            if step.direction == "a_to_f"
            else (
                effective_dtypes["f_to_a_dtype"]
                if step.direction == "f_to_a"
                else effective_dtypes["activation_dtype"]
            )
        )
        if is_control:
            transfer_dtype = step.metadata["wire_dtype"]
        is_collective = step.direction == "collective"
        query = prepare_query(
            self.transfer_provider,
            AFTransferQuery,
            iteration_id=block.iteration_id,
            block_id=block.block_id,
            layer_id=layer_id,
            phase=block.phase,
            direction=step.direction,
            token_count=block.token_count,
            hidden_size=self.hidden_size,
            dtype=transfer_dtype,
            source_device=source_device,
            target_device=target_device,
            network=step.network or link.network,
            operation=step.operation,
            transport_backend=(
                step.metadata["transport_backend"]
                if is_control
                else (
                    "pynccl"
                    if self.cluster_config.af.execution_protocol == "vllm_afd"
                    else "unspecified"
                )
            ),
            link_group=link.resource_group,
            group_size=max(1, len(step.domain_ranks)),
            payload_bytes=step.payload_bytes,
            source_domain=step.source_domain,
            target_domain=step.target_domain,
            logical_participants=step.logical_participants,
            participant_ranks=step.domain_ranks,
            physical_endpoints=step.physical_workers,
            resource_identities=resource_ids,
            topology=self._query_mapping(
                (
                    ("participant_count", len(step.domain_ranks)),
                    ("physical_endpoint_count", len(step.physical_workers)),
                )
            ),
            node_id=step.node_id,
            communication_kind=("collective" if is_collective else "endpoint_leg"),
            layer_type=step.layer_type,
            original_token_count=step.metadata.get(
                "original_token_count", block.token_count
            ),
            assignment_count=step.metadata.get("assignment_count"),
            weight_dtype=effective_dtypes["weight_dtype"],
            activation_dtype=effective_dtypes["activation_dtype"],
            kv_cache_dtype=effective_dtypes["kv_cache_dtype"],
            a_to_f_dtype=effective_dtypes["a_to_f_dtype"],
            f_to_a_dtype=effective_dtypes["f_to_a_dtype"],
            logical_stage_id=step.logical_stage_id,
            leg_id=step.leg_id,
            source_endpoint=step.source_endpoint,
            target_endpoint=step.target_endpoint,
            participant_endpoints=step.participant_endpoints,
            tensor_slices=step.tensor_slices,
            source_layout=step.source_layout,
            target_layout=step.target_layout,
            fabric=step.resource_group,
            resource_group=step.resource_group,
            resource_lane=step.resource_lane,
            logical_input_bytes_per_rank=step.logical_input_bytes_per_rank,
            logical_output_bytes_per_rank=step.logical_output_bytes_per_rank,
            routing_edges=step.routing_edges,
        )
        timing = self.cluster_config.af.communication_timing
        estimate = self._estimate(
            self.transfer_provider, query, block, layer_id, step.operation
        )
        if (
            timing.enabled
            and estimate.details.get("timing_scope") != "transport_service"
        ):
            raise SimulationStateError(
                "Separated communication requires timing_scope='transport_service'; "
                "legacy or eager intervals may include host gaps and queueing"
            )

        communication_ready_at = self.env.now
        wire_node_id = step.node_id
        wire_predecessors = step.predecessor_ids
        if input_readiness is not None:
            input_ids = {producer for producer, _ in input_readiness[0].values()}
            wire_predecessors = tuple(
                key for key in wire_predecessors if key not in input_ids
            )
        submission_finished_at = self.env.now
        receiver_process = None
        native_state = self._native_controls.get(block.iteration_id, {})
        native_call = native_state.get("calls", {}).get(step.node_id)
        receiver_id = f"{step.node_id}:receive_submit"
        needs_receiver = (
            timing.enabled
            and step.direction in {"a_to_f", "f_to_a"}
            and (
                timing.host_receive_submit_us
                or timing.a_step_prepare_us
                or timing.f_step_prepare_us
                or timing.native_control_enabled
            )
        )
        if native_call is not None:
            receiver_process = native_call["receiver"]
        elif needs_receiver:
            receive_cost_field = self._native_call_cost_field(step, "receive")
            receiver_process = self.env.process(
                self._run_receiver_submit(
                    block,
                    layer_id,
                    step,
                    receiver_id,
                    self._host_cost_us(receive_cost_field, block) / 1e6,
                    receive_cost_field,
                )
            )
        if timing.enabled:
            submit_node_id = f"{step.node_id}:host_submit"
            is_collective = step.operation in {
                "tp_collective",
                "all_reduce",
                "all_gather",
                "reduce_scatter",
            }
            submitters = (
                step.physical_workers if is_collective else (step.physical_workers[0],)
            )
            try:
                send_cost_field = self._native_call_cost_field(step, "send")
                yield from self._run_communication_host_stage(
                    block,
                    layer_id,
                    step,
                    submit_node_id,
                    tuple(
                        key
                        for key in step.predecessor_ids
                        if input_readiness is None
                        or key
                        not in {producer for producer, _ in input_readiness[0].values()}
                    ),
                    submitters,
                    self._host_cost_us(send_cost_field, block) / 1e6,
                    "host_submit",
                    details={"cost_field": send_cost_field},
                )
            except BaseException:
                if receiver_process is not None and receiver_process.is_alive:
                    receiver_process.interrupt("AF sender submission cancelled")
                    yield receiver_process
                raise
            submission_finished_at = self.env.now
            if native_call is not None:
                native_call["submitted"].succeed()
                if step.operation == "af_control_payload":
                    native_state["a_ready"].succeed()
            wire_node_id = f"{step.node_id}:transport"
            wire_predecessors = (submit_node_id,)

        readiness_mode = "dag_dependencies_only"
        peer_wait_started_at = self.env.now
        if receiver_process is not None:
            try:
                receiver_result = yield receiver_process
                if isinstance(receiver_result, Exception):
                    raise receiver_result
            except BaseException:
                if receiver_process.is_alive:
                    receiver_process.interrupt("AF receive submission cancelled")
                    yield receiver_process
                raise
            wire_predecessors = (*wire_predecessors, receiver_id)
            readiness_mode = (
                "native_preposted_call_order"
                if native_call is not None
                else "fixed_host_preparation"
            )
        if self.transfer_readiness is not None:
            readiness = self.transfer_readiness(block, layer_id, step)
            if not isinstance(readiness, TransferReadiness):
                raise SimulationStateError(
                    "transfer_readiness must return TransferReadiness"
                )
            readiness_mode = "invocation_events"
            readiness_events = readiness.events(self.env)
            if readiness_events:
                yield self.env.all_of(readiness_events)
        peer_ready_at = self.env.now

        multiple_phases = False
        if self.cluster_config.af.transfer_topology_mode == "physical":
            if hasattr(self.transfer_provider, "communication_plan"):
                plan = self.transfer_provider.communication_plan(query)
                if self.resources.node_by_worker != dict(
                    self.transfer_provider.node_by_worker
                ):
                    raise ConfigurationError(
                        "provider and executor use different communication domain placement"
                    )
                multiple_phases = len(plan.phases) > 1
                ready_at, start_at, end_at = yield from self._run_domain_plan(
                    block,
                    layer_id,
                    step,
                    wire_node_id,
                    wire_predecessors,
                    plan,
                    input_readiness,
                )
                resource_ids = tuple(
                    sorted({key for phase in plan.phases for key in phase.resource_ids})
                )
            else:
                ready_at, start_at, end_at = (
                    yield from self.resources.domain_scheduler.run(
                        resource_ids,
                        estimate.service_duration,
                        (block.iteration_id, block.block_id, layer_id, wire_node_id),
                    )
                )
        else:
            ready_at, start_at, end_at, estimate = yield from self._run_resources_stage(
                block,
                layer_id,
                wire_node_id,
                tuple(resource for _, resource in keyed_resources),
                lambda: estimate,
            )
        if (
            estimate.payload_bytes is not None
            and step.payload_bytes is not None
            and estimate.payload_bytes != step.payload_bytes
        ):
            raise SimulationStateError(
                f"AF provider payload mismatch for node={step.node_id}: "
                f"planned={step.payload_bytes}, returned={estimate.payload_bytes}"
            )
        if self.trace.enabled:
            self.trace.record(
                AFTraceEvent(
                    iteration_id=block.iteration_id,
                    block_id=block.block_id,
                    layer_id=layer_id,
                    phase=block.phase,
                    event_type=(
                        "synchronize"
                        if multiple_phases
                        else (
                            "control_transfer"
                            if is_control
                            else (
                                "collective"
                                if step.operation
                                in {
                                    "tp_collective",
                                    "all_reduce",
                                    "all_gather",
                                    "reduce_scatter",
                                }
                                else "transfer"
                            )
                        )
                    ),
                    operation=step.operation,
                    direction=step.direction,
                    resource_id=(
                        f"completion:{wire_node_id}"
                        if multiple_phases or not resource_ids
                        else resource_ids[0]
                    ),
                    resource_ids=() if multiple_phases else resource_ids,
                    ready_at=end_at if multiple_phases else ready_at,
                    start_at=end_at if multiple_phases else start_at,
                    end_at=end_at,
                    queue_duration=0.0 if multiple_phases else start_at - ready_at,
                    service_duration=0.0 if multiple_phases else end_at - start_at,
                    request_count=block.request_count,
                    token_count=block.token_count,
                    source_id=estimate.source_id,
                    dataset_version=estimate.dataset_version,
                    match_type=estimate.match_type,
                    payload_bytes=step.payload_bytes,
                    details={
                        **estimate.details,
                        "communication_operation": step.operation,
                        "phase_completion_only": multiple_phases,
                        "message_class": "control" if is_control else "data",
                        "transport_backend": query.transport_backend,
                        "wire_dtype": transfer_dtype,
                        **(
                            {
                                "control_schema": step.metadata["control_schema"],
                                "payload_utf8": step.metadata.get("payload_utf8"),
                            }
                            if is_control
                            else {}
                        ),
                        "execution_protocol": self.cluster_config.af.execution_protocol,
                        "readiness_mode": readiness_mode,
                        "readiness_wait_started_at": peer_wait_started_at,
                        "peer_ready_at": peer_ready_at,
                        "readiness_wait_duration": peer_ready_at - peer_wait_started_at,
                        **(
                            {
                                "communication_ready_at": communication_ready_at,
                                "host_submit_finished_at": submission_finished_at,
                                "communication_timing_mode": "separated",
                            }
                            if timing.enabled
                            else {}
                        ),
                    },
                    source_domain=step.source_domain,
                    target_domain=step.target_domain,
                    logical_participants=step.logical_participants,
                    participant_ranks=step.domain_ranks,
                    physical_endpoints=step.physical_workers,
                    node_id=wire_node_id,
                    predecessor_ids=(
                        tuple(
                            f"{wire_node_id}:phase:{phase.phase_id}"
                            for phase in plan.phases
                        )
                        if multiple_phases
                        else tuple(
                            dict.fromkeys((*wire_predecessors, *input_predecessors))
                        )
                    ),
                    dp_rank=step.dp_rank,
                    tp_rank=step.tp_rank,
                    ep_rank=step.ep_rank,
                    tp_group=step.tp_group,
                    ep_group=step.ep_group,
                    logical_stage_id=step.logical_stage_id,
                    layer_type=step.layer_type,
                    tensor_slices=step.tensor_slices,
                    source_endpoint=step.source_endpoint,
                    target_endpoint=step.target_endpoint,
                    source_layout=step.source_layout,
                    target_layout=step.target_layout,
                    routing_edges=step.routing_edges,
                )
            )
        self.active_stages.pop((block.iteration_id, block.block_id, wire_node_id), None)
        if transport_completed is not None and not transport_completed.triggered:
            transport_completed.succeed()
        if timing.enabled:
            is_collective = step.operation in {
                "tp_collective",
                "all_reduce",
                "all_gather",
                "reduce_scatter",
            }
            completers = (
                step.physical_workers if is_collective else (step.physical_workers[-1],)
            )
            yield from self._run_communication_host_stage(
                block,
                layer_id,
                step,
                step.node_id,
                (wire_node_id,),
                completers,
                (
                    0.0
                    if is_control
                    else self._host_cost_us("host_completion_us", block) / 1e6
                ),
                "synchronize",
                details=(
                    {
                        "cost_field": None if is_control else "host_completion_us",
                        "communication_ready_at": communication_ready_at,
                        "submission_finished_at": submission_finished_at,
                        "transfer_finished_at": end_at,
                        "dependency_wait_duration": end_at - submission_finished_at,
                        "wait_is_dependency_not_additional_service": True,
                    }
                    if self.trace.enabled
                    else None
                ),
            )
        self.active_stages.pop((block.iteration_id, block.block_id, step.node_id), None)

    def _run_domain_plan(
        self,
        block,
        layer_id,
        step,
        wire_node_id,
        predecessors,
        plan,
        input_readiness=None,
    ):
        """Instantiate cached phases with fresh dependency events and FIFO waits."""
        ready = self.env.now
        single_phase = len(plan.phases) == 1 and not plan.phases[0].local_work_bytes
        completed = (
            {}
            if single_phase
            else {phase.phase_id: self.env.event() for phase in plan.phases}
        )
        intervals = {}
        children = []

        def run_phase(phase):
            dependencies = [completed[key] for key in phase.predecessor_ids]
            input_ids = ()
            if input_readiness is not None and not phase.predecessor_ids:
                bindings, failure = input_readiness
                identities = tuple(
                    self.resources.worker(rank).physical_worker_ref.identity
                    for rank in phase.participants
                )
                input_ids = tuple(
                    dict.fromkeys(bindings[identity][0] for identity in identities)
                )
                dependencies.extend(bindings[identity][1] for identity in identities)
            if dependencies:
                yield self.env.all_of(dependencies)
            if input_readiness is not None and input_readiness[1].triggered:
                raise SimulationStateError("collective input producer failed")
            phase_predecessors = tuple(
                dict.fromkeys(
                    (
                        *input_ids,
                        *(
                            tuple(
                                f"{wire_node_id}:phase:{key}"
                                for key in phase.predecessor_ids
                            )
                            or predecessors
                        ),
                    )
                )
            )
            estimate = self.transfer_provider.phase_estimate(phase)
            identity = (
                block.iteration_id,
                block.block_id,
                layer_id,
                step.node_id,
                phase.phase_id,
            )
            node_id = f"{wire_node_id}:phase:{phase.phase_id}"
            self._set_active_stage(block, layer_id, node_id, self.env.now)
            try:
                local_children = ()
                if phase.local_work_bytes:
                    phase_ready = self.env.now
                    local_children = tuple(
                        f"{node_id}:worker:{rank}" for rank in phase.participants
                    )
                    processes = [
                        self.env.process(
                            self._run_communication_reorder(
                                block,
                                layer_id,
                                step,
                                phase,
                                child_id,
                                phase_predecessors,
                                rank,
                            )
                        )
                        for rank, child_id in zip(phase.participants, local_children)
                    ]
                    for process in processes:
                        process.defused = True
                    joined = self.env.all_of(processes)
                    joined.defused = True
                    try:
                        yield joined
                    except BaseException:
                        for process in processes:
                            if process.is_alive:
                                process.interrupt(
                                    "local communication reorder cancelled"
                                )
                        for process in processes:
                            if not process.triggered:
                                try:
                                    yield process
                                except BaseException:
                                    pass
                        raise
                    times = (phase_ready, phase_ready, self.env.now)
                elif phase.resource_ids:
                    times = yield from self.resources.domain_scheduler.run(
                        phase.resource_ids, estimate.service_duration, identity
                    )
                else:
                    times = (self.env.now, self.env.now, self.env.now)
                intervals[phase.phase_id] = times
                if len(plan.phases) > 1 and self.trace.enabled:
                    phase_ready, start, end = times
                    if local_children:
                        phase_ready = start = end
                    self.trace.record(
                        AFTraceEvent(
                            iteration_id=block.iteration_id,
                            block_id=block.block_id,
                            layer_id=layer_id,
                            phase=block.phase,
                            event_type="collective",
                            operation=phase.operation,
                            direction=step.direction,
                            resource_id=(
                                phase.resource_ids[0]
                                if phase.resource_ids
                                else f"noop:{node_id}"
                            ),
                            resource_ids=phase.resource_ids,
                            ready_at=phase_ready,
                            start_at=start,
                            end_at=end,
                            queue_duration=start - phase_ready,
                            service_duration=end - start,
                            request_count=block.request_count,
                            token_count=block.token_count,
                            source_id=estimate.source_id,
                            dataset_version=estimate.dataset_version,
                            match_type=estimate.match_type,
                            payload_bytes=phase.logical_input_bytes,
                            details={
                                **estimate.details,
                                "routing_provenance": plan.provenance,
                                "dependency_wait_duration": phase_ready - ready,
                            },
                            node_id=node_id,
                            predecessor_ids=local_children or phase_predecessors,
                            logical_stage_id=step.logical_stage_id,
                            source_domain=step.source_domain,
                            target_domain=step.target_domain,
                            layer_type=step.layer_type,
                            original_token_count=step.metadata.get(
                                "original_token_count", block.token_count
                            ),
                        )
                    )
                if not single_phase:
                    completed[phase.phase_id].succeed()
            finally:
                self.active_stages.pop(
                    (block.iteration_id, block.block_id, node_id), None
                )

        try:
            # One transport phase needs no child Process or ConditionValue.
            # Retain the urgent start and two normal completion turns of the
            # old Initialize -> Process -> AllOf path. Removing those turns
            # can change downstream compute/host FIFO ties at the same time.
            # The old phase-done event had no consumers and can be omitted.
            if single_phase:
                start_event = self.env.event()
                start_event._ok, start_event._value = True, None
                self.env.schedule(start_event, priority=simpy.events.URGENT)
                yield start_event
                error = None
                try:
                    yield from run_phase(plan.phases[0])
                except simpy.Interrupt:
                    raise
                except BaseException as exc:
                    error = exc
                yield self.env.timeout(0)
                yield self.env.timeout(0)
                if error is not None:
                    raise error
            else:
                children = [self.env.process(run_phase(phase)) for phase in plan.phases]
            if children:
                # Parent cancellation detaches its wait callback. Defuse the
                # abandoned join and each child, while yielding still propagates
                # failures to the parent and runs complete resource cleanup.
                for child in children:
                    child.defused = True
                joined = self.env.all_of(children)
                joined.defused = True
                yield joined
        except BaseException:
            for child in children:
                if child.is_alive:
                    child.interrupt("communication plan cancelled")
            # Defuse child interrupts during cleanup; the parent failure remains.
            for child in children:
                if not child.triggered:
                    try:
                        yield child
                    except BaseException:
                        pass
            raise
        start = min((times[1] for times in intervals.values()), default=ready)
        end = max((times[2] for times in intervals.values()), default=ready)
        phase_ready = min((times[0] for times in intervals.values()), default=ready)
        return phase_ready, start, end

    def _run_communication_reorder(
        self, block, layer_id, step, phase, node_id, predecessors, worker_id
    ):
        worker = self.resources.worker(worker_id)
        query = AFComputeQuery(
            iteration_id=block.iteration_id,
            block_id=block.block_id,
            layer_id=layer_id,
            phase=block.phase,
            stage="layout",
            operation="data_reorder",
            token_count=1,
            local_token_count=1,
            context_lengths=block.context_lengths,
            model_id=self.model_id,
            dtype=phase.dtype,
            device_type=worker.hardware,
            worker_id=worker_id,
            source_domain=step.source_domain,
            physical_worker=worker.physical_worker_ref.identity,
            local_shape={
                "tokens": 1,
                "hidden_size": self.hidden_size,
                "global_hidden_size": self.hidden_size,
                "tp_size": 1,
                "reorder_bytes": phase.local_work_bytes,
            },
        )
        estimate = self._estimate(
            self.compute_provider, query, block, layer_id, node_id
        )
        ready, start, end, _ = yield from self._run_resources_stage(
            block,
            layer_id,
            node_id,
            (self.resources.compute_slots[worker_id],),
            lambda: estimate,
        )
        if self.trace.enabled:
            self.trace.record(
                AFTraceEvent(
                    iteration_id=block.iteration_id,
                    block_id=block.block_id,
                    layer_id=layer_id,
                    phase=block.phase,
                    event_type="compute",
                    operation="data_reorder",
                    direction=None,
                    resource_id=f"worker:{worker_id}",
                    resource_ids=(f"worker:{worker_id}",),
                    ready_at=ready,
                    start_at=start,
                    end_at=end,
                    queue_duration=start - ready,
                    service_duration=end - start,
                    request_count=block.request_count,
                    token_count=block.token_count,
                    source_id=estimate.source_id,
                    dataset_version=estimate.dataset_version,
                    match_type=estimate.match_type,
                    node_id=node_id,
                    predecessor_ids=predecessors,
                    source_domain=step.source_domain,
                    target_domain=step.target_domain,
                    layer_type=step.layer_type,
                    stage="layout",
                    logical_stage_id=step.logical_stage_id,
                    details={
                        **estimate.details,
                        "local_work_bytes": phase.local_work_bytes,
                        "layout_permutation": phase.layout_permutation,
                        "wire_bytes": 0,
                    },
                )
            )
        self.active_stages.pop((block.iteration_id, block.block_id, node_id), None)

    def _run_communication_host_stage(
        self,
        block,
        layer_id,
        step,
        node_id,
        predecessors,
        participants,
        duration,
        event_type,
        details=None,
    ):
        """Run host service without holding the communication fabric."""
        keyed = self.resources.host_resources_for(participants)
        if len(keyed) > 1:
            # Each participating host submits independently. Acquiring all hosts
            # before starting service would hold an idle host behind a busy peer.
            child_ids = tuple(f"{node_id}:host:{identity}" for identity in participants)
            children = [
                self.env.process(
                    self._run_communication_host_stage(
                        block,
                        layer_id,
                        step,
                        child_id,
                        predecessors,
                        (identity,),
                        duration,
                        event_type,
                        details,
                    )
                )
                for identity, child_id in zip(participants, child_ids)
            ]
            try:
                yield self.env.all_of(children)
            except simpy.Interrupt:
                for child in children:
                    if child.is_alive:
                        child.interrupt("AF collective host stage cancelled")
                raise
            # Retain the original DAG completion identity as a zero-cost barrier.
            predecessors = child_ids
            duration = 0.0
            event_type = "synchronize"
        estimate = AFServiceEstimate(
            duration,
            (
                f"profile:{self.overhead_profile.fingerprint}:{details['cost_field']}"
                if self.overhead_profile is not None
                and details
                and details.get("cost_field") in self.overhead_profile.document["costs"]
                else "assumption:communication-host-cost"
            ),
            dataset_version="communication-timing-config",
            details=(
                {
                    "communication_operation": step.operation,
                    "timing_scope": (
                        self.overhead_profile.document["timing_scope"]
                        if self.overhead_profile is not None
                        else "host_service"
                    ),
                    "fixed_cost_us": duration * 1e6,
                    "host_resource_scope": (
                        self.cluster_config.af.communication_timing.host_resource_scope
                    ),
                    **(details or {}),
                    **(
                        {"host_profile": self.overhead_profile.describe()}
                        if self.overhead_profile is not None
                        else {}
                    ),
                    **(
                        {
                            "effective_cost": self.host_cost_provenance(
                                details["cost_field"], tokens=block.token_count
                            ),
                            "calibrated": self.host_cost_provenance(
                                details["cost_field"], tokens=block.token_count
                            )["calibrated"],
                        }
                        if details and details.get("cost_field")
                        else {}
                    ),
                }
                if self.trace.enabled
                else {}
            ),
        )
        ready, start, end, _ = yield from self._run_resources_stage(
            block,
            layer_id,
            node_id,
            tuple(resource for _, resource in keyed),
            lambda: estimate,
        )
        if self.trace.enabled:
            self.trace.record(
                AFTraceEvent(
                    iteration_id=block.iteration_id,
                    block_id=block.block_id,
                    layer_id=layer_id,
                    phase=block.phase,
                    event_type=event_type,
                    operation=event_type,
                    direction=step.direction,
                    resource_id=keyed[0][0],
                    resource_ids=tuple(key for key, _ in keyed),
                    ready_at=ready,
                    start_at=start,
                    end_at=end,
                    queue_duration=start - ready,
                    service_duration=end - start,
                    request_count=block.request_count,
                    token_count=block.token_count,
                    source_id=estimate.source_id,
                    dataset_version=estimate.dataset_version,
                    details=estimate.details,
                    node_id=node_id,
                    predecessor_ids=predecessors,
                    logical_stage_id=step.logical_stage_id,
                    leg_id=step.leg_id,
                    source_domain=step.source_domain,
                    target_domain=step.target_domain,
                    layer_type=step.layer_type,
                    physical_endpoints=participants,
                )
            )
        self.active_stages.pop((block.iteration_id, block.block_id, node_id), None)

    def _run_resources_stage(
        self,
        block: AFBlock,
        layer_id: int,
        stage: str,
        resources: tuple[simpy.Resource, ...],
        estimate_factory,
    ):
        ready_at = self.env.now
        self._set_active_stage(block, layer_id, stage, ready_at)
        estimate = estimate_factory()
        requests = (
            [resource.request() for resource in resources]
            if estimate.service_duration > 0
            else []
        )
        try:
            if requests:
                yield self.env.all_of(requests)
            start_at = self.env.now
            self._set_active_stage(block, layer_id, stage, ready_at, start_at)
            yield self.env.timeout(estimate.service_duration)
            end_at = self.env.now
        finally:
            for resource, request in zip(resources, requests):
                if request.triggered:
                    resource.release(request)
                else:
                    request.cancel()
            self.active_stages.pop((block.iteration_id, block.block_id, stage), None)
        return ready_at, start_at, end_at, estimate

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
