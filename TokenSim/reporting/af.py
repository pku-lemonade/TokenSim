"""AF iteration summaries, phase fidelity and provider provenance."""

from typing import Any


def get_af_stats(engine) -> dict[str, Any]:
    executor = getattr(engine, "af_executor", None)
    summaries = getattr(engine, "af_iteration_summaries", [])
    runtime_summary = getattr(engine, "af_runtime_summary", None)
    cluster_config = getattr(engine, "cluster_config", None)
    af_config = getattr(cluster_config, "af", None)
    phases = set(getattr(af_config, "phases", ())) if executor is not None else set()
    stats: dict[str, Any] = {
        "af_enabled": executor is not None,
        "af_iteration_count": len(summaries),
        "af_trace": (
            [executor.trace.export_summary(summary) for summary in summaries]
            if executor is not None and executor.trace.enabled
            else None
        ),
        "af_runtime_state": (
            runtime_summary()
            if executor is not None and callable(runtime_summary)
            else None
        ),
        "af_phase_fidelity": (
            {
                "prefill": (
                    (
                        "prefill_af_rank_local"
                        if getattr(executor, "runtime_plan", None) is not None
                        else "af_prefill_compatibility"
                    )
                    if "prefill" in phases
                    else "prefill_aggregate"
                ),
                "decode": (
                    "decode_af_rank_local" if "decode" in phases else "decode_aggregate"
                ),
            }
            if executor is not None
            else None
        ),
        "af_latency_providers": None,
        "af_operator_stats": None,
    }
    if executor is None:
        return stats
    # Provenance of the AF service times: which provider answered and, for the
    # operator-table provider, the same match-type / missing-shape accounting
    # the step-level backend reports.
    providers: dict[str, Any] = {}
    for name in ("compute", "transfer"):
        provider = getattr(engine, f"af_{name}_provider", None)
        describe = getattr(provider, "describe", None)
        if callable(describe):
            providers[name] = describe()
    stats["af_latency_providers"] = providers or None
    stats_dict = getattr(
        getattr(engine, "af_compute_provider", None), "stats_dict", None
    )
    stats["af_operator_stats"] = stats_dict() if callable(stats_dict) else None
    return stats
