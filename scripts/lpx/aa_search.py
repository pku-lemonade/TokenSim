"""Search capacity-valid inventories and replay AA-shaped serial requests."""

from dataclasses import asdict
import math
from statistics import median

import numpy as np
import simpy

from TokenSim.errors import ConfigurationError
from scripts.lpx.aa_model import Capacity


ATTENTION_TPS = (1, 2, 4, 8, 16, 32)


def inventory(placements):
    result = {"rubin": 0, "groq_lp30": 0}
    for placement in placements:
        result[placement.device] += placement.cards
    return result


def request_cost(costs, case, workload):
    inputs = math.ceil(workload["input_tokens"] * costs.rho_in)
    outputs = max(2, math.ceil(workload["output_tokens"] * costs.rho_out))
    parts = []
    for placement in case["decode"]:
        parts.append(
            costs.decode(
                placement.device, placement.tp, placement.part, inputs, outputs
            )
        )
    components = {key: sum(part[key] for part in parts) for key in parts[0]}
    components["pipeline_us"] = sum(
        costs.pipeline(placement)
        for placement in case["decode"]
        if placement.part != "ffn"
    )
    components["cross_us"] = 0.0
    if case["mode"] in {"af", "af-global", "af-reverse"}:
        devices = {placement.device: placement for placement in case["decode"]}
        components["cross_us"] = costs.af_boundary(
            devices["rubin"].tp,
            devices["groq_lp30"].tp,
            layers=(
                costs.model.num_layers // costs.model.full_attention_interval
                + 1  # Embedding dispatch and final hidden-state return.
                if case["mode"] == "af-global"
                else None
            ),
        )
    durations = sum(components.values())
    first = case["prefill"] or case["decode"][0]
    ttft = costs.prefill(first.device, first.tp, inputs)
    chunks = math.ceil(inputs / costs.chunk)
    ttft += costs.pipeline(first, costs.chunk) * chunks
    handoff_us, handoff_bytes = 0.0, 0
    if case["mode"] in {"pd", "pd-reverse", "af-global", "af-reverse"}:
        target = case["decode"][1 if case["mode"] == "af-global" else 0]
        # Pack each target PP stage, send its complete needed cache from the
        # source rank holding that head, then replicate inside the target TP.
        # Source-rank loads enforce the aggregate per-endpoint Ethernet budget.
        source_loads = [0.0] * first.cards
        target_loads = [0.0] * target.cards

        def owner(placement, layer, head, heads):
            stage = next(
                i
                for i, (start, end) in enumerate(placement.stages)
                if start <= layer < end
            )
            return stage * placement.tp + min(
                placement.tp - 1, head * placement.tp // heads
            )

        for layer in range(costs.model.num_layers):
            full = costs.model.full(layer)
            if case["mode"] == "af-global" and full:
                continue
            heads = (
                costs.model.num_global_key_value_heads
                if full
                else costs.model.num_key_value_heads
            )
            one_head = costs.model.kv_bytes(full, heads, inputs, costs.kv_dtype)
            for head in range(heads):
                source_loads[owner(first, layer, head, heads)] += one_head
                target_loads[owner(target, layer, head, heads)] += one_head
                handoff_bytes += one_head
        handoff_us = costs.cross_link.analytical_transfer_us(
            max(max(source_loads), max(target_loads))
        )
        # Tree fanout is a modeled pack/replication approximation, not a
        # zero-cost transfer when the target TP replicates KV heads.
        for full in (False, True):
            if case["mode"] == "af-global" and full:
                continue
            heads = (
                costs.model.num_global_key_value_heads
                if full
                else costs.model.num_key_value_heads
            )
            replicas = max(1, target.tp // heads)
            if replicas > 1:
                local = costs.hardware.link(costs.devices[target.device].scale_up_link)
                payload = sum(
                    costs.model.kv_bytes(full, target.tp, inputs, costs.kv_dtype)
                    for layer in range(costs.model.num_layers)
                    if costs.model.full(layer) == full
                )
                handoff_us += math.ceil(
                    math.log2(replicas)
                ) * local.analytical_transfer_us(payload)
        ttft += handoff_us
        # The prefill pool produced the first token; forwarding its ID is
        # included before the decoder can consume it.
        ttft += costs.cross_link.analytical_transfer_us(4)
    return dict(
        durations_us=np.broadcast_to(durations, (outputs - 1,)),
        components=components,
        native_input_tokens=inputs,
        native_output_tokens=outputs,
        ttft_us=ttft,
        handoff_us=handoff_us,
        handoff_bytes=handoff_bytes,
    )


def evaluate_case(costs, case, workloads):
    placements = list(case["decode"])
    if case["mode"] in {"pd", "pd-reverse"}:
        placements.append(case["prefill"])
    cards = inventory(placements)
    benchmarks = []
    for workload in workloads:
        request = request_cost(costs, case, workload)
        durations = request["durations_us"]
        components = {
            key: float(np.mean(value)) for key, value in request["components"].items()
        }
        tail = durations[int(len(durations) * 0.2) :]
        primary = tail if costs.timing_window == "tail80" else durations
        benchmarks.append(
            dict(
                workload=workload["name"],
                target_tps=workload["target_tps"],
                predicted_tps=1e6 / (costs.rho_out * float(np.mean(primary))),
                predicted_full_tps=1e6 / (costs.rho_out * float(np.mean(durations))),
                predicted_tail80_tps=1e6 / (costs.rho_out * float(np.mean(tail))),
                timing_window=costs.timing_window,
                ttft_s=request["ttft_us"] / 1e6,
                request_latency_s=(request["ttft_us"] + float(np.sum(durations))) / 1e6,
                decode_component_mean_us=components,
                native_input_tokens=request["native_input_tokens"],
                native_output_tokens=request["native_output_tokens"],
                handoff_us=request["handoff_us"],
                handoff_bytes=request["handoff_bytes"],
            )
        )
    return dict(
        name=case["name"],
        mode=case["mode"],
        inventory=cards,
        total_cards=sum(cards.values()),
        racks=dict(
            rubin=math.ceil(cards["rubin"] / 72),
            lpx=math.ceil(cards["groq_lp30"] / 256),
        ),
        decode=[asdict(placement) for placement in case["decode"]],
        prefill=asdict(case["prefill"]) if case["prefill"] else None,
        benchmarks=benchmarks,
        score=min(row["predicted_tps"] / row["target_tps"] for row in benchmarks),
        meets_all_targets=all(
            row["predicted_tps"] >= row["target_tps"] for row in benchmarks
        ),
    )


def search(costs, workloads, *, max_lpu=1024, max_gpu=144, reserve=0.05):
    peak = max(
        math.ceil(w["input_tokens"] * costs.rho_in)
        + math.ceil(w["output_tokens"] * costs.rho_out)
        for w in workloads
    )
    if peak > costs.model.max_position_embeddings:
        raise ConfigurationError("AA workload exceeds the Gemma context limit")
    capacity = Capacity(
        costs.hardware,
        costs.model,
        weight_dtype=costs.weight_dtype,
        kv_dtype=costs.kv_dtype,
        reserve=reserve,
    )
    cases, rejected = [], []

    def place(device, tp, part):
        result = capacity.place(device, tp, part, peak, costs.chunk)
        if result is None or result.cards > (max_gpu if device == "rubin" else max_lpu):
            rejected.append(
                dict(
                    device=device,
                    tp=tp,
                    part=part,
                    reason="capacity or search inventory limit",
                )
            )
            return None
        return result

    for device, mode in (("rubin", "rubin"), ("groq_lp30", "lpx")):
        for tp in ATTENTION_TPS:
            placement = place(device, tp, "full")
            if placement is not None:
                cases.append(
                    dict(
                        name=f"{mode}-tp{tp}-pp{len(placement.stages)}",
                        mode=mode,
                        decode=(placement,),
                        prefill=None,
                    )
                )
                pd_mode = "pd" if mode == "lpx" else "pd-reverse"
                for pre_tp in ATTENTION_TPS:
                    prefill = place(
                        "rubin" if mode == "lpx" else "groq_lp30", pre_tp, "full"
                    )
                    if prefill is not None:
                        cases.append(
                            dict(
                                name=f"{pd_mode}-p{pre_tp}-d{tp}-pp{len(placement.stages)}",
                                mode=pd_mode,
                                decode=(placement,),
                                prefill=prefill,
                            )
                        )
    for a_tp in ATTENTION_TPS:
        reverse_a = place("groq_lp30", a_tp, "attention")
        if reverse_a is not None:
            for f_tp in ATTENTION_TPS:
                reverse_f = place("rubin", f_tp, "ffn")
                reverse_prefill = place("rubin", f_tp, "full")
                if reverse_f is not None and reverse_prefill is not None:
                    cases.append(
                        dict(
                            name=f"af-reverse-a{a_tp}-f{f_tp}-pp{len(reverse_a.stages)}",
                            mode="af-reverse",
                            decode=(reverse_a, reverse_f),
                            prefill=reverse_prefill,
                        )
                    )
        a = place("rubin", a_tp, "attention")
        prefill = place("rubin", a_tp, "full")
        if a is None or prefill is None or a.stages != prefill.stages:
            continue
        # A single client alternates prefill and decode, so it can share the
        # Rubin pool. The full model's memory was checked above.
        for f_tp in range(1, 257):
            if costs.model.intermediate_size % f_tp:
                continue
            f = place("groq_lp30", f_tp, "ffn")
            if f is not None:
                cases.append(
                    dict(
                        name=f"af-a{a_tp}-f{f_tp}-pp{len(f.stages)}",
                        mode="af",
                        decode=(a, f),
                        prefill=prefill,
                    )
                )
        global_a = place("rubin", a_tp, "global_attention")
        for local_tp in ATTENTION_TPS:
            local_f = place("groq_lp30", local_tp, "local_ffn")
            if global_a is not None and local_f is not None:
                cases.append(
                    dict(
                        name=f"af-global-a{a_tp}-l{local_tp}-pp{len(local_f.stages)}",
                        mode="af-global",
                        decode=(global_a, local_f),
                        prefill=prefill,
                    )
                )
    points = [evaluate_case(costs, case, workloads) for case in cases]
    selections, selected_cases = {}, {}
    for mode in ("rubin", "lpx", "af", "af-global", "pd", "af-reverse", "pd-reverse"):
        subset = [p for p in points if p["mode"] == mode]
        if not subset:
            selections[mode] = dict(best=None, minimum_cards=None)
            continue
        best = max(subset, key=lambda p: (p["score"], -p["total_cards"]))
        passing = [p for p in subset if p["meets_all_targets"]]
        minimum = (
            min(passing, key=lambda p: (p["total_cards"], -p["score"]))
            if passing
            else None
        )
        selections[mode] = dict(best=best, minimum_cards=minimum)
        for point in (best, minimum):
            if point:
                selected_cases[point["name"]] = next(
                    c for c in cases if c["name"] == point["name"]
                )
    return dict(points=points, rejected=rejected, selections=selections), selected_cases


def replay(costs, case, workloads, *, requests=50, warmup=5):
    """SimPy serial replay; deterministic repeats have no statistical CI."""
    results = []
    for workload in workloads:
        request = request_cost(costs, case, workload)
        env = simpy.Environment()
        samples = []

        def client():
            for index in range(warmup + requests):
                submitted = env.now
                yield env.timeout(request["ttft_us"] / 1e6)
                first = env.now
                tail_start = int(len(request["durations_us"]) * 0.2)
                tail_first = first
                for token_index, duration in enumerate(request["durations_us"]):
                    if token_index == tail_start:
                        tail_first = env.now
                    yield env.timeout(float(duration) / 1e6)
                last = env.now
                native = request["native_output_tokens"]
                if index >= warmup:
                    full_tps = (native - 1) / costs.rho_out / (last - first)
                    tail_tps = (
                        (native - 1 - tail_start) / costs.rho_out / (last - tail_first)
                    )
                    samples.append(
                        dict(
                            request=index - warmup,
                            submitted_s=submitted,
                            first_output_s=first,
                            last_output_s=last,
                            native_output_tokens=native,
                            tps=(
                                tail_tps
                                if costs.timing_window == "tail80"
                                else full_tps
                            ),
                            full_tps=full_tps,
                            tail80_tps=tail_tps,
                            ttft_s=first - submitted,
                        )
                    )

        env.process(client())
        env.run()
        results.append(
            dict(
                workload=workload["name"],
                samples=samples,
                median_tps=median(s["tps"] for s in samples),
                concurrency=1,
                timing_window=costs.timing_window,
                warmup=warmup,
                independent_text_samples=False,
            )
        )
    return results
