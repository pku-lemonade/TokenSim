"""Offline Roofline tables followed by real AF scheduling; CPU only, no weights.

Default: one V3-shaped MoE layer on virtual 1A1F. --full-model: all 61
layers on virtual 1A32F (BF16 weights do not fit four A100s).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, fields
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import simpy
from TokenSim.af import AFComputeQuery, AFExecutor, compile_af_runtime_plan
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.constants import _GB
from TokenSim.moe.config import MoEModelConfig
from TokenSim.operator_data import (
    Calibration, EvidenceGrade, HardwareProfile, LookupResult,
    ModelOperatorManifest, Observation, OperatorDataComputeLatencyProvider,
    OperatorDataGenerator, OperatorDataLookup, OperatorDataPackage,
    OperatorDataTransferLatencyProvider, SourceRecord,
)
from TokenSim.operator_data.providers import ModelShape
from TokenSim.operator_data.schema import TABLE_SPECS


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class ShapeDiscovery:
    """Dummy times enumerate keys only; these timings are never reported."""
    def __init__(self, assumptions):
        self.package = SimpleNamespace(dataset_version="discovery-not-results")
        self.shapes = {}
        self.assumptions = assumptions

    def lookup(self, table, key):
        self.shapes[(table, json.dumps(key, sort_keys=True))] = dict(table=table, key=dict(key))
        return LookupResult(1.0, "exact", "discovery-not-results", key)

    def link_profile(self, key):
        self.lookup("communication", key)
        return LookupResult(1.0, "exact", "assumed-link", key, values=dict(
            fixed_latency_us=self.assumptions["link_fixed_latency_us"],
            effective_bandwidth_bytes_per_s=self.assumptions["interconnect_bandwidth_bytes_per_s"]))


def configuration(full_model=False, a_tp=1, f_tp=None, af_batch_size=32, assumptions=None, router_location="a", communication_timing=None, hypothetical_f_memory_bytes=None):
    assumptions = assumptions or json.loads((ROOT / "data/operator_data/deepseek-v3-theoretical.json").read_text())
    raw = json.loads((ROOT / "data/models/deepseek-v3-671b-mla.json").read_text())
    layers = raw["num_hidden_layers"] if full_model else 1
    f_tp = f_tp or (32 if full_model else 1)
    device = assumptions["device"]
    if hypothetical_f_memory_bytes is not None and hypothetical_f_memory_bytes <= 0:
        raise ValueError("Hypothetical F memory capacity must be positive")
    f_device = device if hypothetical_f_memory_bytes is None else device + "-hypothetical-f-memory"
    phases = ["prefill", "decode"]
    domains, groups = [], []
    for role, size in (("a", a_tp), ("f", f_tp)):
        capability = "af_attention" if role == "a" else "af_ffn"
        groups.append(dict(name=f"{role}-pool", capabilities=[capability], hardware=device if role == "a" else f_device,
                           num_workers=size, network="af-net"))
        parallel = dict(tensor_parallel_size=size, pipeline_parallel_size=1, data_parallel_size=1)
        if role == "f":
            parallel.update(expert_parallel_size=size, expert_parallel_mode="rank_reuse")
        mappings = [dict(domain_rank=i, tp_rank=i, pp_rank=0, dp_rank=0,
                        ep_rank=i if role == "f" else 0,
                        physical_worker=dict(group_name=f"{role}-pool", worker_index=i))
                    for i in range(size)]
        stages = [("all_reduce", "a_compute", "a_to_f"), ("all_gather", "residual", "f_to_a")] if role == "a" else [("all_reduce", "expert_compute", "f_to_a")]
        collectives = [dict(operation=op, after_stage=stage, participants=list(range(size)),
                            network="af-net", resource_group="af-fabric", resource_lane=lane)
                       for op, stage, lane in stages] if size > 1 else []
        domains.append(dict(name=f"{role}-domain", capability=capability, phases=phases,
                            worker_group=f"{role}-pool", parallel=parallel,
                            logical_instances=[dict(name=f"{role}0", rank_mapping=mappings)],
                            communication_plan=collectives))
    doc = dict(num_workers=a_tp+f_tp, networks={"af-net": "assumed-shared-fabric"},
        worker_groups=groups, af=dict(enabled=True, phases=phases,
        af_block_size=af_batch_size, trace_enabled=True, router_location=router_location,
        weight_dtype="bf16", activation_dtype="bf16", kv_cache_dtype="bf16",
        execution_domains=domains, links=[dict(name="af-link", phases=phases,
            attention_group="a-pool", ffn_group="f-pool", attention_domain="a-domain",
            ffn_domain="f-domain", network="af-net", resource_group="af-fabric",
            duplex="full", a_to_f_capacity=1, f_to_a_capacity=1)]))
    if communication_timing is not None:
        doc['af']['communication_timing'] = communication_timing
    cluster = ClusterConfig(**doc)
    shape = ModelShape(**{f.name: raw[f.name] for f in fields(ModelShape) if f.name in raw},
                       tp_size=a_tp, ep_size=f_tp, f_elementwise=("rmsnorm",))
    first_dense = raw["first_k_dense_replace"] if full_model else 0
    moe = MoEModelConfig(is_moe_model=True, num_experts=shape.num_experts,
        num_experts_per_tok=shape.top_k, moe_intermediate_size=shape.routed_intermediate_size,
        num_shared_experts=shape.num_shared_experts, num_moe_layers=layers-first_dense,
        first_k_dense_replace=first_dense, hidden_size=shape.hidden_size,
        intermediate_size=shape.dense_intermediate_size,
        num_attention_heads=shape.num_attention_heads, num_key_value_heads=1)
    metadata = SimpleNamespace(models={shape.model_id: SimpleNamespace(
        Dmodel=shape.hidden_size, Nlayer=layers, Nhead=shape.num_attention_heads,
        Nkvhead=1, FFN_Hidden=shape.dense_intermediate_size, Vocab=raw["vocab_size"],
        attention_type="mla", tie_word_embeddings=raw["tie_word_embeddings"],
        **asdict(shape.mla_dimensions))}, hardwares={device: SimpleNamespace(
        MM_Card_Num=1, Capacity=assumptions["memory_capacity_bytes"]/_GB)})
    if hypothetical_f_memory_bytes is not None:
        metadata.hardwares[f_device] = SimpleNamespace(
            MM_Card_Num=1, Capacity=hypothetical_f_memory_bytes/_GB)
    plan = compile_af_runtime_plan(cluster_config=cluster, worker_configs=cluster.workers(),
        roofline=metadata, model=shape.model_id, moe_config=moe, block_size=16, dtype="bf16")
    return doc, cluster, shape, moe, plan


def simulate(lookup, config, batch, context, steps=1, phase="decode"):
    _, cluster, shape, moe, plan = config
    if phase != "decode":
        raise ValueError("Full V3 AF scheduling supports decode only; use prefill_attention_smoke for MLA prefill")
    # Fixed-batch decode starts with populated caches; no admission/swap simulator.
    required_blocks = batch * ((context+steps-1+plan.block_size-1)//plan.block_size)
    if any(required_blocks > r.logical_gpu_blocks for r in plan.replicas if r.is_attention):
        raise ValueError("MLA workload KV cache exceeds A replica capacity")
    env = simpy.Environment()
    executor = AFExecutor(env, cluster, cluster.workers(),
        OperatorDataComputeLatencyProvider(lookup, ModelOperatorManifest({shape.model_id: shape}), "analytical"),
        OperatorDataTransferLatencyProvider(lookup, "analytical"), shape.model_id, "bf16",
        total_layers=plan.total_layers, hidden_size=shape.hidden_size,
        moe_config=moe, random_seed=42, runtime_plan=plan)
    rows = []
    for step in range(steps):
        requests = [SimpleNamespace(id=i, is_prefill=phase == "prefill", prefill_len=context,
            prefill_compute_len=context, context_len=context+step, generation_idx=step)
            for i in range(batch)]
        iteration = executor.submit(0, requests, phase)
        env.process(executor.run_iteration(iteration))
        env.run()
        if not isinstance(lookup, ShapeDiscovery):
            executor.trace.validate_invariants(iteration.iteration_id)
        rows.append(dict(iteration_id=iteration.iteration_id, phase=phase, batch_size=batch,
                         context=context+step, makespan_us=iteration.makespan*1e6))
    return executor, rows


def legacy_coverage_shapes(device):
    """Schema v2 requires all six base tables, even if MLA never queries them."""
    common = dict(device=device, backend="analytical", activation_dtype="bf16")
    attention = dict(common, attention_type="mha", kv_cache_dtype="bf16", batch_size=1,
                     num_heads=1, num_kv_heads=1, head_dim=2, window_size=0)
    return [dict(table="context_attention", key=dict(attention, input_seq_len=1)),
            dict(table="generation_attention", key=dict(attention, context_len=1)),
            dict(table="moe", key=dict(common, weight_dtype="bf16", operation="routed_expert",
                phase="decode", distribution="balanced", num_tokens=1, hidden_size=2,
                inter_size=2, top_k=1, num_experts=1, tp_size=1, ep_size=1))]


def prefill_attention_smoke(lookup, config, length=8):
    """One A-rank MLA operator composition, not a full-stack prefill benchmark.

    The existing AF planner's formal per-layer runtime graph is decode-only;
    avoid sending V3's dense/shared layout through its legacy prefill graph.
    """
    _, cluster, shape, _, _ = config
    local = dict(tokens=length, hidden_size=shape.hidden_size, global_hidden_size=shape.hidden_size,
        **shape.mla_dimensions.local_shape(shape.num_attention_heads, shape.tp_size))
    query = AFComputeQuery(iteration_id=0, block_id=0, layer_id=0, phase="prefill", stage="attention",
        token_count=length, local_token_count=length, original_token_count=length,
        context_lengths=(length,), model_id=shape.model_id, dtype="bf16",
        device_type=cluster.workers()[0].hardware, worker_id=0, layer_type="moe", local_shape=local)
    provider = OperatorDataComputeLatencyProvider(lookup, ModelOperatorManifest({shape.model_id: shape}), "analytical")
    return provider.estimate(query)


def generate_package(output, discovery, assumptions, observations=(), sources=()):
    device = assumptions["device"]
    hardware = HardwareProfile(**{f.name: assumptions[f.name] for f in fields(HardwareProfile)
                                  if f.name in assumptions})
    source = SourceRecord("assumption:uniform-fabric", EvidenceGrade.D, "assumption", device,
        "analytical", assumptions["notes"], "unmeasured-virtual-cluster")
    shapes = list(discovery.shapes.values())
    links = [Observation("communication", dict(s["key"],
        fixed_latency_us=assumptions["link_fixed_latency_us"],
        effective_bandwidth_bytes_per_s=assumptions["interconnect_bandwidth_bytes_per_s"]),
        0.0, source.source_id) for s in shapes if s["table"] == "communication"]
    seeds = legacy_coverage_shapes(device)
    calibration = Calibration(offset_us=assumptions["launch_overhead_us"],
                              calibration_id="ASSUMED-launch-not-measured")
    generator = OperatorDataGenerator("deepseek-mla-theoretical-v1", {device: hardware},
        [source, *sources], calibrations={t: calibration for t in TABLE_SPECS if t != "communication"})
    generator.generate(output, "deepseek-mla", shapes+seeds, links+list(observations))
    write_json(Path(output) / "unused-schema-coverage.json", seeds)
    return OperatorDataPackage.load(output)


def save_trace(output, executor):
    trace, components = [], defaultdict(float)
    for event in executor.trace.events:
        e = event.as_dict()
        trace.append(dict(name=e["stage"] or e["operation"] or e["direction"], cat=e["event_type"],
            ph="X", pid=1, tid=e["resource_id"], ts=e["start_at"]*1e6,
            dur=e["service_duration"]*1e6, args=e))
        start = e["start_at"]*1e6
        for item in e.get("details", {}).get("components", []):
            components[item["name"]] += item["latency_us"]
            trace.append(dict(name=item["name"], cat="analytical-operator", ph="X", pid=2,
                tid=e["resource_id"], ts=start, dur=item["latency_us"], args=item))
            start += item["latency_us"]
    write_json(output, dict(displayTimeUnit="ms", traceEvents=trace,
                           metadata={"timing": "SIMULATED, not GPU measurements"}))
    return dict(components)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/deepseek-mla")
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--a-tp", type=int, default=1)
    parser.add_argument("--f-tp", type=int)
    parser.add_argument("--af-batch-size", type=int, default=32)
    parser.add_argument("--assumptions", type=Path, default=ROOT / "data/operator_data/deepseek-v3-theoretical.json")
    args = parser.parse_args()
    if min(*args.batches, args.context, args.steps, args.a_tp, args.f_tp or 1, args.af_batch_size) < 1:
        parser.error("batch, context, steps and parallel sizes must be positive")
    assumptions = json.loads(args.assumptions.read_text())
    config = configuration(args.full_model, args.a_tp, args.f_tp, args.af_batch_size, assumptions)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    doc, _, shape, _, plan = config
    model = asdict(shape)
    model.pop("model_id")
    write_json(output / "model-manifest.json", dict(schema_version=2, models={shape.model_id: model}))
    ModelOperatorManifest.load(output / "model-manifest.json")
    write_json(output / "cluster.json", doc)
    write_json(output / "runtime-plan.json", plan.summary())
    write_json(output / "hardware-assumptions.json", assumptions)
    discovery = ShapeDiscovery(assumptions)
    for batch in args.batches:
        print(f"Discovering offline shapes: layers={plan.total_layers}, batch={batch}", flush=True)
        simulate(discovery, config, batch, args.context, args.steps)
    # Small full-sequence prefill covers the expanded path; not a TTFT benchmark.
    prefill_attention_smoke(discovery, config)
    write_json(output / "requested-shapes.json", list(discovery.shapes.values()))
    package = generate_package(output / "operator-data", discovery, assumptions)
    lookup = OperatorDataLookup(package)
    rows = []
    for batch in args.batches:
        executor, iterations = simulate(lookup, config, batch, args.context, args.steps)
        duration_us = sum(r["makespan_us"] for r in iterations)
        throughput = batch*args.steps / (duration_us/1e6)
        component_us = save_trace(output / f"decode-b{batch}.trace.json", executor)
        rows.append(dict(batch_size=batch, avg_step_us=duration_us/args.steps,
            output_tokens_per_s=throughput, total_power_w=doc["num_workers"]*assumptions["device_power_w"],
            component_service_us_sum=component_us, iterations=iterations))
        print(f"batch={batch}: {duration_us/args.steps/1000:.3f} ms/step, {throughput:.2f} tokens/s (theoretical)", flush=True)
    prefill = prefill_attention_smoke(lookup, config)
    prefill_trace, start = [], 0.0
    for item in prefill.details["components"]:
        prefill_trace.append(dict(name=item["name"], ph="X", pid=1, tid=0,
                                 ts=start, dur=item["latency_us"], args=item))
        start += item["latency_us"]
    write_json(output / "prefill-smoke.trace.json", dict(traceEvents=prefill_trace,
        metadata={"scope": "Single MLA attention layer per A rank; simulated, not measured"}))
    write_json(output / "summary.json", dict(status="analytical_uncalibrated", virtual_devices=doc["num_workers"],
        total_layers=plan.total_layers, af_batch_size=args.af_batch_size, decode=rows,
        prefill_attention_smoke=dict(service_us=prefill.service_duration*1e6, **prefill.details),
        limitations=["Synthetic uniform top-8, not V3 noaux_tc routing", "BF16, not official FP8 checkpoint",
            "No embedding, lm_head or sampling latency; transformer-layer stack only",
            "Peak roofline and assumed launches/fabric; physical accuracy pending measurement",
            "Prefill supports equal full sequences only", "Service-time sums across GPUs are not wall time"]))
    write_json(output / "measurement-backlog.json", dict(status="pending, user authorized later measurements",
        shapes=list(discovery.shapes.values()), contract=str(ROOT / "docs/deepseek-v3-mla.md")))


if __name__ == "__main__":
    main()
