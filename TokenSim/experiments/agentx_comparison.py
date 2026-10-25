"""Compare complete AgentX predictions with one frozen hardware run per case."""

from __future__ import annotations

import math
from pathlib import Path
import json

from TokenSim.experiments.fingerprints import file_hash
from TokenSim.errors import ConfigurationError


CRITICAL_METRICS = frozenset(
    ("output_tps", "ttft_p50_s", "ttft_p90_s", "tpot_p50_s", "tpot_p90_s")
)
METRICS = {
    "output_tps": (("output_token_throughput_tps",), ("output_tput_tps",)),
    "input_tps": (("input_token_throughput_tps",), ("input_tput_tps",)),
    "successful_requests": (("request_count",), ("request_count",)),
    **{
        f"{name}_{stat}_s": ((predicted, stat), (observed, stat))
        for name, predicted, observed in (
            ("ttft", "ttft_s", "ttft_s"),
            ("tpot", "tpot_s", "itl_s"),
            ("e2e", "e2e_s", "e2e_s"),
        )
        for stat in ("mean", "p50", "p90", "p95", "p99")
    },
}


def _value(mapping, path):
    value = mapping
    for name in path:
        if not isinstance(value, dict) or name not in value:
            return None
        value = value[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _prediction_metrics(result):
    value = result.get("agentx_metrics")
    return value if isinstance(value, dict) else {}


def compare_case(baseline, current, reference):
    """Retain every metric and flag regressions without a composite error score."""
    errors = []
    for label, result in (("baseline", baseline), ("current", current)):
        metrics = _prediction_metrics(result)
        if (
            result.get("termination_reason") != "completed"
            or result.get("notdone") != []
        ):
            errors.append(f"{label}: incomplete or failed replay")
        count = metrics.get("request_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            errors.append(f"{label}: no successful requests")
        if metrics.get("concurrency") != reference.get("concurrency"):
            errors.append(f"{label}: reference concurrency mismatch")
        expected_window = reference.get("window", {}).get("issue_s")
        if expected_window is None or metrics.get("issue_window_s") != expected_window:
            errors.append(f"{label}: reference issue window mismatch")
    observed = reference.get("metrics", {})
    rows = []
    for name, (prediction_path, reference_path) in METRICS.items():
        truth = _value(observed, reference_path)
        before = _value(_prediction_metrics(baseline), prediction_path)
        after = _value(_prediction_metrics(current), prediction_path)
        row = dict(
            metric=name,
            critical=name in CRITICAL_METRICS,
            reference=truth,
            baseline=before,
            current=after,
            status="unavailable",
        )
        if (
            truth is not None
            and truth > 0
            and before is not None
            and after is not None
            and before > 0
            and after > 0
        ):
            before_error, after_error = 100 * (before / truth - 1), 100 * (
                after / truth - 1
            )
            change = abs(after_error) - abs(before_error)
            row.update(
                baseline_error_percent=before_error,
                current_error_percent=after_error,
                absolute_error_change_pp=change,
                status=(
                    "unchanged"
                    if math.isclose(change, 0, abs_tol=1e-10)
                    else "improved" if change < 0 else "regressed"
                ),
            )
        elif name in CRITICAL_METRICS:
            errors.append(f"{name}: missing or nonpositive timing/throughput")
        rows.append(row)
    cache = {}
    for label, result in (("baseline", baseline), ("current", current)):
        metrics = _prediction_metrics(result)
        cache[label] = {
            name: metrics.get(name)
            for name in (
                "gpu_cache_hit_rate",
                "cpu_cache_hit_rate",
                "recomputed_tokens",
            )
        }
    cache["reference"] = {
        name: observed.get(name)
        for name in ("gpu_cache_hit_rate", "cpu_cache_hit_rate")
    }
    cache["comparison_scope"] = (
        "diagnostic: hardware server counters and successful-request first-admission cohorts differ"
    )
    return dict(
        reference_id=reference.get("reference_id"),
        hardware=reference.get("hardware"),
        status="invalid" if errors else "valid",
        validation_errors=errors,
        metrics=rows,
        cache_diagnostics=cache,
        critical_regressions=[
            r["metric"] for r in rows if r["critical"] and r["status"] == "regressed"
        ],
    )


def compare_manifest(path):
    """Load only artifacts named by a before/after manifest; execute no commands."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    reference_path = (
        Path(manifest["reference_directory"]) / "inferencex_key_points.json"
    )
    reference_rows = json.loads(reference_path.read_text())
    references = {r["reference_id"]: r for r in reference_rows}
    if len(references) != len(reference_rows):
        raise ConfigurationError("duplicate AgentX hardware reference identity")
    groups = {}
    for case in manifest["cases"]:
        group = groups.setdefault(case["hardware"], {})
        if case["label"] in group:
            raise ConfigurationError("duplicate AgentX comparison case")
        group[case["label"]] = case
    comparisons = []
    for hardware, group in sorted(groups.items()):
        row = dict(hardware=hardware, status="incomplete")
        if set(group) != {"baseline", "current"}:
            row["diagnostic"] = "exactly one baseline and current case required"
            comparisons.append(row)
            continue
        inputs = {}
        for label, case in group.items():
            output = Path(case["output"]) / f"agentx_c{case['concurrency']}.json"
            if output.exists():
                inputs[label] = (output, json.loads(output.read_text()))
        if len(inputs) == 2:
            ids = {case["reference_id"] for case in group.values()}
            arguments = []
            for case in group.values():
                args = list(case["args"])
                if args.count("--results_path") != 1 or args[-1] == "--results_path":
                    raise ConfigurationError(
                        "comparison commands require one result path"
                    )
                index = args.index("--results_path")
                del args[index : index + 2]
                arguments.append(args)
            if arguments[0] != arguments[1]:
                row.update(
                    status="invalid",
                    diagnostic="baseline/current simulation arguments differ",
                )
            elif len(ids) != 1 or next(iter(ids)) not in references:
                row.update(
                    status="invalid", diagnostic="reference run identity mismatch"
                )
            elif references[next(iter(ids))].get("hardware") != hardware:
                row.update(status="invalid", diagnostic="reference hardware mismatch")
            else:
                row = compare_case(
                    inputs["baseline"][1],
                    inputs["current"][1],
                    references[next(iter(ids))],
                )
                row["artifacts"] = {
                    label: dict(path=str(source), sha256=file_hash(source))
                    for label, (source, _) in inputs.items()
                }
        comparisons.append(row)
    return dict(
        schema_version=1,
        manifest_sha256=file_hash(path),
        reference_sha256=file_hash(reference_path),
        status=(
            "valid"
            if comparisons and all(x["status"] == "valid" for x in comparisons)
            else "incomplete_or_invalid"
        ),
        cases=comparisons,
    )
