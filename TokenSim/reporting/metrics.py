"""Run metadata, AgentX metrics and workload latency target attainment."""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
from TokenSim.results import MetricData
from TokenSim.llm.llm_request import Request


def _is_root_stream(stream_id: str | None) -> bool:
    """True if stream_id belongs to a root stream (single or multi-stream)."""
    return stream_id is not None and (
        stream_id == "root" or stream_id.startswith("root:")
    )


@dataclass
class RunReport:
    """How a run ended and where its results go, as decided by the runner."""

    results_path: Path
    qps: float
    batching: str
    termination_reason: str = "completed"
    failure: str | None = None
    # AgentX replay: concurrency and the replay's runtime metadata.
    agentx_concurrency: int | None = None
    agentx_metadata: dict[str, Any] | None = None

    @property
    def result_name(self) -> str:
        if self.agentx_concurrency is not None:
            return f"agentx_c{self.agentx_concurrency}.json"
        return f"result_{self.qps}.json"


def get_agentx_metrics(
    run: RunReport,
    requests: list[Request],
    duration: float,
    num_gpus: int = 1,
) -> dict[str, Any] | None:
    if run.agentx_concurrency is None:
        return None

    requests = [r for r in requests if r.is_done and not r.cancel_requested]
    root_requests = [
        r for r in requests if _is_root_stream(getattr(r.source, "stream_id", None))
    ]
    subagent_requests = [
        r for r in requests if not _is_root_stream(getattr(r.source, "stream_id", None))
    ]

    ttft = [request.prefill_latency for request in requests]
    e2e = [request.total_time for request in requests]
    # TPOT/ITL: mean inter-token latency excluding first token (matches vLLM).
    tpot = [
        request.decode_time_sum / (request.generation_idx - 1)
        for request in requests
        if request.generation_idx > 1
    ]
    normalized_latency = [
        request.total_time / request.generation_idx
        for request in requests
        if request.total_time > 0 and request.generation_idx > 0
    ]

    input_tokens = sum(request.prefill_len for request in requests)
    output_tokens = sum(request.generation_idx for request in requests)
    cached_input_tokens = sum(request.cached_prefill_tokens for request in requests)
    external_input_tokens = sum(
        request.external_cached_prefill_tokens for request in requests
    )
    computed_input_tokens = sum(
        (request.effective_prefill_tokens or request.prefill_len)
        for request in requests
    )
    metadata = dict(run.agentx_metadata or {})
    if metadata.get("replay_protocol"):
        sources = [r.first_prefill_source for r in requests]
        if any(source is None for source in sources):
            raise ValueError(
                "successful native requests need first-admission source observations"
            )
        gpu_input_tokens = sum(s.gpu_hit_tokens for s in sources)
        external_input_tokens = sum(s.cpu_hit_tokens for s in sources)
        cached_input_tokens = gpu_input_tokens + external_input_tokens
        computed_input_tokens = sum(s.computed_tokens for s in sources)

    def summary(values):
        if not values and metadata.get("replay_protocol"):
            return {
                key: 0 if key == "count" else None
                for key in ("count", "mean", "p50", "p75", "p90", "p95", "p99", "max")
            }
        return _percentile_summary(values)

    def interactivity_summary(latencies):
        values = [value for value in latencies if value > 0]
        result = summary(values)
        for name, value in result.items():
            if name not in ("count", "max") and value:
                result[name] = 1.0 / value
        if values:
            result["max"] = 1.0 / min(values)
        return result

    # In-flight request count at profile end (set by benchmark if available).
    in_flight_at_cutoff = metadata.get("in_flight_at_cutoff", 0)

    result: dict[str, Any] = {
        "methodology": "agentx-closed-loop-simulation",
        "official_submission_compatible": False,
        "concurrency": run.agentx_concurrency,
        "trace_count": metadata.get("trace_count", 0),
        "play_count": metadata.get("play_count", 0),
        "warmup_request_count": metadata.get("warmup_request_count", 0),
        "warmup_duration_s": metadata.get("warmup_duration_s", 0.0),
        "random_seed": metadata.get("random_seed"),
        "workload_seed": metadata.get("workload_seed"),
        "initial_trace_ids": metadata.get("initial_trace_ids", []),
        "trace_sampling_strategy": (
            "shared-sequential-pool"
            if metadata.get("replay_protocol")
            else "seed-shuffled-pool"
        ),
        "profile_duration_s": duration,
        "metric_observation_defined": bool(requests),
        "throughput_denominator_kind": (
            "successful-request-boundaries"
            if metadata.get("replay_protocol") and requests
            else (
                "phase-elapsed-no-success"
                if metadata.get("replay_protocol")
                else "legacy-profile-window"
            )
        ),
        "replay_protocol": metadata.get("replay_protocol"),
        "phase_request_counts": metadata.get("phase_request_counts"),
        "phase_trace_counts": metadata.get("phase_trace_counts"),
        "phase_unique_original_requests": metadata.get(
            "phase_unique_original_requests"
        ),
        "issue_window_s": metadata.get("issue_window_s"),
        "grace_period_s": metadata.get("grace_period_s"),
        "cancellation_timeout_s": metadata.get("cancellation_timeout_s"),
        "cancel_unreturned": metadata.get("cancel_unreturned"),
        "tree_idle_jumps": metadata.get("tree_idle_jumps"),
        "system_idle_jumps": metadata.get("system_idle_jumps"),
        "in_flight_at_cutoff": in_flight_at_cutoff,
        # --- Raw counts (always present for reconciliation) ---
        "request_count": len(requests),
        "root_request_count": len(root_requests),
        "subagent_request_count": len(subagent_requests),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "computed_input_tokens": computed_input_tokens,
        # --- Derived throughput ---
        "request_throughput_rps": len(requests) / duration if duration else 0.0,
        "input_token_throughput_tps": input_tokens / duration if duration else 0.0,
        "input_token_throughput_per_gpu_tps": (
            input_tokens / duration / max(1, num_gpus) if duration else 0.0
        ),
        "output_token_throughput_tps": output_tokens / duration if duration else 0.0,
        "output_token_throughput_per_gpu_tps": (
            output_tokens / duration / max(1, num_gpus) if duration else 0.0
        ),
        # --- Latency distributions (all requests) ---
        "ttft_s": summary(ttft),
        "tpot_s": summary(tpot),
        "e2e_s": summary(e2e),
        "interactivity_tps": interactivity_summary(tpot),
        "e2e_normalized_interactivity_tps": interactivity_summary(normalized_latency),
        "interactivity_percentile_semantics": "inverse_of_matching_latency_percentile",
    }
    # Native replay records the input-token decomposition per request. Older
    # Mooncake paths expose their own counters; absent native observations
    # must not silently become a zero CPU hit rate.
    if metadata.get("replay_protocol"):
        result.update(
            cache_source_observation="first-admission",
            gpu_cached_input_tokens=cached_input_tokens - external_input_tokens,
            cpu_cached_input_tokens=external_input_tokens,
            gpu_cache_hit_rate=(
                (cached_input_tokens - external_input_tokens) / input_tokens
                if input_tokens
                else None
            ),
            cpu_cache_hit_rate=(
                external_input_tokens / input_tokens if input_tokens else None
            ),
            readmission_count=sum(
                max(0, r.prefill_admission_count - 1) for r in requests
            ),
            reloaded_gpu_tokens=sum(r.reloaded_gpu_tokens for r in requests),
            reloaded_cpu_tokens=sum(r.reloaded_cpu_tokens for r in requests),
            reloaded_generated_gpu_tokens=sum(
                r.reloaded_generated_gpu_tokens for r in requests
            ),
            reloaded_generated_cpu_tokens=sum(
                r.reloaded_generated_cpu_tokens for r in requests
            ),
            recomputed_tokens=sum(r.recomputed_tokens_total for r in requests),
            latest_gpu_hit_tokens=sum(
                r.cached_prefill_tokens - r.external_cached_prefill_tokens
                for r in requests
            ),
            latest_cpu_hit_tokens=sum(
                r.external_cached_prefill_tokens for r in requests
            ),
        )

    # --- Breakdown by root / subagent ---
    for label, subset in [("root", root_requests), ("subagent", subagent_requests)]:
        sub_ttft = [r.prefill_latency for r in subset]
        sub_e2e = [r.total_time for r in subset]
        sub_tpot = [
            r.decode_time_sum / (r.generation_idx - 1)
            for r in subset
            if r.generation_idx > 1
        ]
        result[f"{label}_ttft_s"] = _percentile_summary(sub_ttft)
        result[f"{label}_tpot_s"] = _percentile_summary(sub_tpot)
        result[f"{label}_e2e_s"] = _percentile_summary(sub_e2e)
        result[f"{label}_output_tokens"] = sum(r.generation_idx for r in subset)
        result[f"{label}_input_tokens"] = sum(r.prefill_len for r in subset)

    return result


def _percentile_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(max(values)),
    }


def slo_report(
    workload_config,
    first_token: list[float],
    decode_token: list[float],
    duration: float,
) -> dict[str, Any] | None:
    """Attainment of the workload's latency targets; ``None`` without targets.

    ``first_token`` and ``decode_token`` hold each request's time to first
    token and mean inter-token latency. A ``max`` target bounds every request
    and goodput counts the requests within all of them; ``p50`` and ``p99``
    targets bound the run's percentiles.
    """
    report: dict[str, Any] = {}
    within = [True] * len(first_token)
    bounded = False
    for name, values in (
        ("first_token_latency", first_token),
        ("decode_token_latency", decode_token),
    ):
        target = getattr(workload_config, name, None)
        bounds = {
            key: value
            for key, value in asdict(target or MetricData(None, None, None)).items()
            if value is not None
        }
        if not bounds:
            continue
        observed = asdict(MetricData.from_list(values))
        report[name] = {
            "target": bounds,
            "observed": {key: observed[key] for key in bounds},
            "met": {
                key: observed[key] is not None and bool(observed[key] <= bound)
                for key, bound in bounds.items()
            },
        }
        if "max" in bounds:
            bounded = True
            within = [
                ok and value <= bounds["max"] for ok, value in zip(within, values)
            ]
    if not report:
        return None
    if bounded:
        report["goodput_requests"] = sum(within)
        report["goodput_requests_per_s"] = sum(within) / duration if duration else 0.0
    return report
