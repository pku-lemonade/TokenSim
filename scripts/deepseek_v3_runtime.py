"""Shared DeepSeek-V3 runtime factory for the operator-table AF backend."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import simpy

from TokenSim.af import AFExecutor, AFTraceRecorder, build_operator_table_af_providers, compile_af_runtime_plan
from TokenSim.af.latency import AFEndpoint, AFTransferQuery, AFServiceEstimate
from TokenSim.af.trace import AFTraceEvent
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.hardware import HardwareContext
from TokenSim.hardware.device.sourced_value import SourcedValue
from TokenSim.hardware.placement import build_topology_placement

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "deepseek-v3-671b-mla"
_ASSUMED_DEVICE = "a100_sxm_80g"


def load_assumptions(path: str | Path | None = None) -> dict[str, Any]:
    filename = Path(path) if path else ROOT / "data/operator_data/deepseek-v3-theoretical.json"
    return json.loads(filename.read_text(encoding="utf-8"))


def configuration(*, full_model: bool = False, a_tp: int = 1, f_tp: int | None = None,
                  af_batch_size: int = 32, data_root: str | Path | None = None,
                  device_id: str = _ASSUMED_DEVICE, network: str = "nvlink3",
                  hypothetical_f_memory_gib: int | None = None):
    """Build a V3 ModelSpec, AF cluster and compiled runtime plan."""
    if a_tp < 1 or (f_tp is not None and f_tp < 1) or af_batch_size < 1:
        raise ValueError("parallel sizes and af_batch_size must be positive")
    hardware = HardwareContext.load(data_root or ROOT / "data")
    model = hardware.model(MODEL_ID)
    if not full_model:
        model = replace(model, num_layers=1,
                        moe=replace(model.moe, num_moe_layers=1, first_k_dense_replace=0))
    f_tp = f_tp or (32 if full_model else 1)
    device = hardware.device(device_id).device_id
    f_device = device
    if hypothetical_f_memory_gib is not None:
        if type(hypothetical_f_memory_gib) is not int or hypothetical_f_memory_gib < 1:
            raise ValueError("hypothetical_f_memory_gib must be a positive integer")
        base = hardware.device(device)
        f_device = f"{device}_virtual_f_{hypothetical_f_memory_gib}gib"
        hardware.devices.add(replace(
            base, device_id=f_device, display_name=f_device, aliases=(),
            memory_capacity_bytes=SourcedValue(
                hypothetical_f_memory_gib * 2**30, "assumption:virtual-f-capacity", "D"),
        ))
    phases = ["prefill", "decode"]
    groups = [
        dict(name="a-pool", capabilities=["prefill", "af_attention"], hardware=device,
             num_workers=a_tp, network="af-net", operator_backend="vllm"),
        dict(name="f-pool", capabilities=["prefill", "af_ffn"], hardware=f_device,
             num_workers=f_tp, network="af-net", operator_backend="vllm"),
    ]
    a_mappings = [dict(domain_rank=i, tp_rank=i, pp_rank=0, dp_rank=0, ep_rank=0,
                        physical_worker=dict(group_name="a-pool", worker_index=i))
                  for i in range(a_tp)]
    f_mappings = [dict(domain_rank=i, tp_rank=0, pp_rank=0, dp_rank=0, ep_rank=i,
                        physical_worker=dict(group_name="f-pool", worker_index=i))
                  for i in range(f_tp)]
    domains = [
        dict(name="a-domain", capability="af_attention", phases=phases,
             worker_group="a-pool",
             parallel=dict(tensor_parallel_size=a_tp, pipeline_parallel_size=1,
                           data_parallel_size=1, expert_parallel_size=1),
             logical_instances=[dict(name="a0", rank_mapping=a_mappings)],
             communication_plan=[dict(operation=op, after_stage=stage, participants=list(range(a_tp)),
                 network="af-net", resource_group="af-fabric", resource_lane=lane)
                 for op,stage,lane in (("all_reduce","a_compute","a_to_f"),("all_gather","residual","f_to_a"))] if a_tp > 1 else []),
        dict(name="f-domain", capability="af_ffn", phases=phases,
             worker_group="f-pool",
             parallel=dict(tensor_parallel_size=1, pipeline_parallel_size=1,
                           data_parallel_size=1, expert_parallel_size=f_tp,
                           expert_parallel_mode="rank_reuse"),
             logical_instances=[dict(name="f0", rank_mapping=f_mappings)],
             communication_plan=[]),
    ]
    doc = dict(
        num_workers=a_tp + f_tp,
        networks={"af-net": network},
        worker_groups=groups,
        af=dict(
            enabled=True, phases=phases, af_block_size=af_batch_size,
            trace_enabled=True, router_location="a", weight_dtype="bf16",
            activation_dtype="bf16", kv_cache_dtype="bf16",
            execution_domains=domains,
            links=[dict(name="af-link", phases=phases,
                attention_group="a-pool", ffn_group="f-pool",
                attention_domain="a-domain", ffn_domain="f-domain",
                network="af-net", resource_group="af-fabric", duplex="full",
                a_to_f_capacity=1, f_to_a_capacity=1)],
        ),
    )
    cluster = ClusterConfig(**doc)
    workers = cluster.workers()
    placement = build_topology_placement(cluster, workers, hardware)
    plan = compile_af_runtime_plan(
        cluster_config=cluster, worker_configs=workers, model=model,
        hardware=hardware, block_size=16, dtype="bf16",
    )
    return SimpleNamespace(doc=doc, hardware=hardware, model=model, cluster=cluster,
                           workers=workers, placement=placement, plan=plan)


def simulate(config, batch: int, context: int, *, steps: int = 1, trace: bool = True,
             fallback: str = "table_first"):
    if batch < 1 or context < 1 or steps < 1:
        raise ValueError("batch, context and steps must be positive")
    env = simpy.Environment()
    compute, transfer = build_operator_table_af_providers(
        hardware=config.hardware, model=config.model,
        cluster_config=config.cluster, worker_configs=config.workers,
        placement=config.placement, dtype="bf16", fallback=fallback,
    )
    executor = V3Executor(
        env, config.cluster, config.workers, compute, transfer,
        config.model.model_id, "bf16", config.plan.total_layers,
        config.model.hidden_size, moe_config=config.model.moe, random_seed=42,
        runtime_plan=config.plan, trace_recorder=AFTraceRecorder(trace),
    )
    rows = []
    for step in range(steps):
        requests = [SimpleNamespace(
            id=i, is_prefill=False, prefill_len=context,
            context_len=context + step, generation_idx=step,
        ) for i in range(batch)]
        iteration = executor.submit(0, requests, "decode")
        env.process(executor.run_iteration(iteration))
        env.run()
        rows.append(dict(iteration_id=iteration.iteration_id,
                         batch_size=batch, context_len=context + step,
                         makespan_us=iteration.makespan * 1e6))
    return executor, rows, compute, transfer


class V3Executor(AFExecutor):
    """Whole-model boundaries use the same A and fabric resources as layers."""

    def _run_block(self, block):
        bindings = self.runtime_plan.bindings_for_instance(block.placement.a_domain, block.placement.a_instance)
        link = self._links[block.link_name]
        previous = yield from self._boundary_stage(block, bindings, "embedding_lookup", ())
        if len(bindings) > 1:
            previous = yield from self._boundary_collective(block, bindings, link, "embedding_reduce", "all_reduce", block.token_count * self.hidden_size * 2, previous)
        for layer_id in range(self.total_layers):
            dag = self.plan_builder.build_layer(block, layer_id, link)
            yield from self._run_dag(block, layer_id, link, dag)
            # The executor serializes layers by yielding their DAGs. Record
            # these cross-DAG edges too so the exported trace is reconstructible.
            roots = {node.node_id for node in dag.nodes if not node.predecessor_ids}
            if self.trace.enabled:
                for index, event in enumerate(self.trace.events):
                    if event.node_id in roots:
                        self.trace.events[index] = replace(event, predecessor_ids=previous)
            # TP execution has one terminal combine node per A rank.  The next
            # transformer layer must wait for the complete prior layer.
            previous = tuple(dag.terminal_node_ids)
        previous = yield from self._boundary_stage(block, bindings, "final_norm", previous)
        previous = yield from self._boundary_stage(block, bindings, "lm_head", previous)
        if len(bindings) > 1:
            previous = yield from self._boundary_collective(block, bindings, link, "logits_gather", "all_gather", block.token_count * (self.compute_provider.model.vocab_size // len(bindings)) * 2, previous)
        yield from self._boundary_stage(block, bindings, "token_selection", previous)
        self.active_iterations[block.iteration_id].block_completed_at[block.block_id] = self.env.now

    def _boundary_stage(self, block, bindings, operation, predecessors):
        processes = [self.env.process(self._boundary_compute(block, binding, operation, predecessors)) for binding in bindings]
        yield self.env.all_of(processes)
        return tuple(process.value for process in processes)

    def _boundary_compute(self, block, binding, operation, predecessors):
        model = self.compute_provider.model
        tp = self.runtime_plan.domain_runtime(binding.domain).tensor_parallel_size
        if model.vocab_size % tp:
            raise ValueError(f"model={model.model_id} vocab_size={model.vocab_size} must divide A TP={tp}")
        tokens, hidden = block.token_count, model.hidden_size
        if operation == "lm_head":
            table, key = "gemm", dict(dtype=model.dtype, m=tokens, n=model.vocab_size // tp, k=hidden)
        else:
            op = {"embedding_lookup": "embedding_lookup", "final_norm": "rmsnorm", "token_selection": "argmax"}[operation]
            table, key = "elementwise", dict(op_name=op, dtype=model.dtype, num_tokens=tokens, hidden_size=model.vocab_size if op == "argmax" else hidden)
        resolver = self.compute_provider.resolver_for(binding.hardware)
        resources = self.resources.compute_resources_for((binding.physical_identity,))
        worker_id = resources[0][0]
        node_id = f"i{block.iteration_id}:b{block.block_id}:boundary:{operation}:r{binding.domain_rank}"
        def estimate():
            value = resolver.query(table, key)
            dataset_version = (
                f"analytical:{resolver.device.device_id}"
                if value.match_type == "analytical"
                else resolver.dataset_version or f"unknown:{resolver.device.device_id}"
            )
            component = dict(name=operation, table=table, key=key, latency_us=value.latency_us, match_type=value.match_type, source_id=value.source_id, dataset_version=dataset_version)
            return AFServiceEstimate(value.latency_us / 1e6, value.source_id, dataset_version=dataset_version, match_type=value.match_type, details=dict(provider="operator_table", boundary=True, components=[component], total_latency_us=value.latency_us))
        ready, start, end, value = yield from self._run_resources_stage(block, -1, node_id, tuple(r for _, r in resources), estimate)
        self.trace.record(AFTraceEvent(
            iteration_id=block.iteration_id, block_id=block.block_id, layer_id=-1, phase=block.phase,
            event_type="compute", resource_id=f"worker:{worker_id}", resource_ids=(f"worker:{worker_id}",),
            ready_at=ready, start_at=start, end_at=end, queue_duration=start-ready, service_duration=end-start,
            request_count=block.request_count, token_count=tokens, source_id=value.source_id,
            dataset_version=value.dataset_version, match_type=value.match_type, details=value.details,
            operation=operation, stage="layout", worker_id=worker_id, node_id=node_id, predecessor_ids=predecessors,
            source_domain=binding.domain, target_domain=binding.domain, participant_ranks=(binding.domain_rank,),
            physical_endpoints=(binding.physical_identity,), tp_rank=binding.tp_rank, dp_rank=binding.dp_rank,
        ))
        self.active_stages.pop((block.iteration_id, block.block_id, node_id), None)
        return node_id

    def _boundary_collective(self, block, bindings, link, name, operation, payload, predecessors):
        resources = self.resources.communication_resources_for_step(link, "tp_collective", link.resource_group, "a_to_f")
        identities = tuple(self.resources.format_link_resource(key) for key, _ in resources)
        node_id = f"i{block.iteration_id}:b{block.block_id}:boundary:{name}"
        query = AFTransferQuery(iteration_id=block.iteration_id, block_id=block.block_id, layer_id=-1,
            phase=block.phase, direction="tp_collective", token_count=block.token_count, hidden_size=self.hidden_size,
            dtype=self.dtype, source_device=bindings[0].hardware, target_device=bindings[-1].hardware,
            network=link.network, operation=operation, payload_bytes=payload, group_size=len(bindings),
            participant_ranks=tuple(b.domain_rank for b in bindings), physical_endpoints=tuple(b.physical_identity for b in bindings))
        ready, start, end, value = yield from self._run_resources_stage(block, -1, node_id, tuple(r for _,r in resources), lambda: self.transfer_provider.estimate(query))
        self.trace.record(AFTraceEvent(iteration_id=block.iteration_id, block_id=block.block_id, layer_id=-1,
            phase=block.phase, event_type="collective", resource_id=identities[0], resource_ids=identities,
            ready_at=ready, start_at=start, end_at=end, queue_duration=start-ready, service_duration=end-start,
            request_count=block.request_count, token_count=block.token_count, source_id=value.source_id,
            dataset_version=value.dataset_version, match_type=value.match_type, details=value.details,
            operation=operation, direction="tp_collective", node_id=node_id, predecessor_ids=predecessors,
            payload_bytes=payload, participant_ranks=query.participant_ranks, physical_endpoints=query.physical_endpoints))
        self.active_stages.pop((block.iteration_id, block.block_id, node_id), None)
        return (node_id,)


def write_profile(path: str | Path, executor, metadata: dict[str, Any] | None = None):
    events = []
    for event in executor.trace.events:
        row = event.as_dict()
        events.append(dict(
            name=row.get("stage") or row.get("operation") or row.get("event_type"),
            cat=row.get("event_type"), ph="X", pid=1, tid=row.get("resource_id"),
            ts=row["start_at"] * 1e6, dur=row["service_duration"] * 1e6,
            args=row,
        ))
    payload = {"displayTimeUnit": "us", "traceEvents": events,
               "metadata": metadata or {"timing": "simulated operator-table service time"}}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
