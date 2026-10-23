"""Capacity admission and card-count search using native LPX decode graphs."""

from dataclasses import asdict

from TokenSim.af.runtime.admission import ResidentKV
from TokenSim.comm.streaming import StreamingConfig
from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError
from scripts.lpx.config import Layout
from scripts.lpx.execution import ExecutionAssumptions, with_execution
from scripts.lpx.native_config import configuration
from scripts.lpx.native_decode import curve_for, interval_summary, serial_replay
from scripts.lpx.token_units import TokenUnits

WORKLOADS = ((1000, 1000, 3400), (10000, 1500, 3382), (100000, 2000, 3431))


def layouts(model, tp_values, max_chips, pp_values=None, kvp_values=(1,)):
    """Yield stage TP ``N`` layouts with KVP factors that divide ``N``."""
    integer(max_chips, "maximum LP30 chips", minimum=1)
    for kvp in kvp_values:
        integer(kvp, "KVP", minimum=1)
    for tp in sorted(set(tp_values)):
        integer(tp, "TP", minimum=1)
        if model.num_attention_heads % tp or model.intermediate_size % tp:
            raise ConfigurationError(
                "search TP must divide attention heads and FFN width"
            )
        stages = range(1, min(model.num_layers, max_chips // tp) + 1)
        if pp_values is not None:
            for pp in pp_values:
                integer(pp, "PP", minimum=1)
            stages = sorted(set(pp_values) & set(stages))
        for kvp in sorted(k for k in set(kvp_values) if tp % k == 0):
            for pp in stages:
                yield Layout(
                    a_device="groq_lp30",
                    f_device="groq_lp30",
                    a_tp=tp // kvp,
                    a_cp=kvp,
                    f_tp=tp,
                    a_pp=pp,
                    f_pp=pp,
                    f_ep=1,
                )


def with_startup(hardware, kernel_launch_us):
    return with_execution(
        hardware, ExecutionAssumptions(kernel_launch_us=kernel_launch_us)
    )


def build_runtime(
    hardware, model, layout, *, reuse=True, layer_counts=None, kv_settings=None
):
    return configuration(
        hardware,
        model,
        layout,
        namespace="search",
        dag_template_cache=True,
        colocated_dense_reuse=reuse,
        pipeline_layer_counts=layer_counts,
        **(kv_settings or {}),
    )


def memory_report(runtime, maximum_context):
    if maximum_context > runtime.model.max_position_embeddings:
        raise ConfigurationError("search exceeds the model context limit")
    memory = ResidentKV(runtime.plan)
    owner = next(iter(runtime.plan.request_owners))
    memory.admit(
        0,
        owner.domain,
        owner.instance,
        maximum_context=maximum_context,
        current_context=0,
    )
    rows = [
        dict(
            physical=state.physical_identity,
            capacity_bytes=state.capacity_bytes,
            weights_bytes=state.model_bytes,
            buffers_bytes=state.protocol_buffer_bytes,
            buffer_breakdown=dict(state.protocol_buffers),
            kv_bytes=memory.reserved[state.physical_identity],
            total_bytes=state.model_bytes
            + state.protocol_buffer_bytes
            + memory.reserved[state.physical_identity],
        )
        for state in runtime.plan.physical_states
    ]
    return dict(
        maximum_context=maximum_context,
        peak_rank=max(rows, key=lambda r: r["total_bytes"]),
        minimum_headroom_bytes=min(
            r["capacity_bytes"] - r["total_bytes"] for r in rows
        ),
        ranks=len(rows),
        scope=(
            "Weights, protocol buffers, LP30 live scratch/reserve and one resident KV lease."
            if runtime.hardware.device("groq_lp30").analytical.get("lpx_memory")
            is not None
            else "Native decode weights, protocol buffers and one resident KV lease; prefill workspace excluded."
        ),
    )


def streaming_settings(layout, launch_us=0.5):
    # Multi-tray overlap is unsupported by the existing streaming refinement.
    # Retain its complete catalog collective cost and disclose this limitation.
    return asdict(
        StreamingConfig(
            overlap=layout.a_stage_ranks <= 16, collective_launch_us=launch_us
        )
    )


def acceptance(rows):
    if [(r["input_tokens"], r["output_tokens"]) for r in rows] != [
        (i, o) for i, o, _ in WORKLOADS
    ]:
        raise ConfigurationError(
            "acceptance requires every configured workload in order"
        )
    for row in rows:
        number(row["generation_tps"], "generation TPS", positive=True)
    return dict(
        common_3400=all(row["generation_tps"] >= 3400 for row in rows),
        published_targets=all(
            row["generation_tps"] >= target
            for row, (_, _, target) in zip(rows, WORKLOADS)
        ),
        common_margin=min(row["generation_tps"] / 3400 - 1 for row in rows),
        published_margin=min(
            row["generation_tps"] / target - 1
            for row, (_, _, target) in zip(rows, WORKLOADS)
        ),
    )


def screen(
    hardware,
    model,
    layout,
    *,
    launch_us=0.5,
    reuse=True,
    verify=False,
    requests=50,
    warmup=5,
    invocation_us=0.0,
    layer_counts=None,
    token_units=None,
    kv_settings=None,
):
    units = TokenUnits(**(token_units or {}))
    workloads = [
        (i, o, target, *units.native_lengths(i, o)) for i, o, target in WORKLOADS
    ]
    row = dict(layout=asdict(layout), lp30=layout.a_ranks)
    try:
        runtime = build_runtime(
            hardware,
            model,
            layout,
            reuse=reuse,
            layer_counts=layer_counts,
            kv_settings=kv_settings,
        )
        memory = memory_report(runtime, max(i + o for _, _, _, i, o in workloads))
    except ConfigurationError as exc:
        return dict(row, status="infeasible", reason=str(exc))
    streaming = streaming_settings(layout, launch_us)
    rows = []
    for aa_inputs, aa_outputs, target, inputs, outputs in workloads:
        curve = curve_for(runtime, inputs, outputs, streaming, invocation_us)
        result = interval_summary(curve, inputs, outputs)
        native_tps = result["generation_tps"]
        result.update(
            input_tokens=aa_inputs,
            output_tokens=aa_outputs,
            native_input_tokens=inputs,
            native_output_tokens=outputs,
            native_generation_tps=native_tps,
            generation_tps=units.aa_intervals(outputs) / result["generation_seconds"],
            token_units=units.report(),
        )
        result["published_target_tps"] = target
        if verify:
            contexts = {
                segment["low"] + (segment["high"] - segment["low"]) * k // 4
                for segment in curve.segments
                for k in (1, 3)
            }
            result["withheld_probes"] = curve.check(contexts)
            result["serial_replay"] = serial_replay(
                runtime,
                curve,
                inputs=inputs,
                outputs=outputs,
                requests=requests,
                warmup=warmup,
            )
        rows.append(result)
    return dict(
        row,
        status="ok",
        memory=memory,
        workloads=rows,
        acceptance=acceptance(rows),
        streaming=streaming,
        overlap_scope=(
            "single-tray"
            if layout.a_stage_ranks <= 16
            else "catalog-only; multi-tray overlap unmodeled"
        ),
        colocated_dense_reuse=reuse,
        invocation_us=invocation_us,
        token_units=units.report(),
        pipeline_layer_counts=layer_counts,
        **({"kv_settings": kv_settings} if kv_settings else {}),
    )


def summarize(rows):
    feasible = [r for r in rows if r["status"] == "ok"]
    result = dict(
        points=len(rows), feasible=len(feasible), infeasible=len(rows) - len(feasible)
    )
    for target in ("common_3400", "published_targets"):
        candidates = [r for r in feasible if r["acceptance"][target]]
        best = (
            min(
                candidates, key=lambda r: (r["lp30"], -r["acceptance"]["common_margin"])
            )
            if candidates
            else None
        )
        result[target] = best
    result["fastest"] = max(
        feasible, key=lambda r: r["acceptance"]["common_margin"], default=None
    )
    return result
