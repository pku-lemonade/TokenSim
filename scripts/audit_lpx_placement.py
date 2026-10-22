"""Audit saved execution and placement searches without rerunning their DAGs."""

import argparse
from bisect import bisect_left
from collections import Counter
import hashlib
import tarfile
from copy import deepcopy
import json
from math import isclose
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lpx.replay_profiles import source_identity

NAMES = (
    "logical-endpoint",
    "compiled-zero",
    "compiled-floor-005",
    "capacity-catalog",
    "capacity-floor-005",
    "capacity-floor-010",
    "capacity-bf16-floor-005",
)
WORKLOADS = ((1000, 1000, 3400), (10000, 1500, 3382), (100000, 2000, 3431))


def compact(row):
    if row is None:
        return None
    result = deepcopy(row)
    if "partition_evidence" in result:
        result["partition_evidence"].pop("stage_rank_bytes", None)
    for workload in result.get("workloads", []):
        workload.pop("interpolation", None)
        workload.pop("withheld_probes", None)
        workload.pop("serial_replay", None)
    return result


def result_location(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def integrated_seconds(curve):
    samples = sorted(curve["probes"], key=lambda row: row["context"])
    contexts = [row["context"] for row in samples]
    total = 0.0
    for context in range(curve["low"], curve["high"] + 1):
        index = bisect_left(contexts, context)
        right = samples[index]
        if right["context"] == context:
            total += right["latency_s"]
        else:
            left = samples[index - 1]
            fraction = (context - left["context"]) / (
                right["context"] - left["context"]
            )
            total += left["latency_s"] + fraction * (
                right["latency_s"] - left["latency_s"]
            )
    return total


def archive_identity(path):
    digest = hashlib.sha256()
    with tarfile.open(path, "r:gz") as archive:
        for item in sorted(archive.getmembers(), key=lambda item: item.name):
            name = item.name
            selected = (
                name.startswith(("TokenSim/", "scripts/lpx/"))
                and name.endswith(".py")
                or name == "scripts/lpx/replay_engine.cpp"
                or name.startswith(
                    ("data/devices/", "data/models/", "data/topologies/")
                )
                and name.endswith(".yaml")
                or name.startswith("data/operator_data/")
            )
            if selected and item.isfile():
                digest.update(name.encode())
                digest.update(archive.extractfile(item).read())
    return digest.hexdigest()


def audit_robustness(out, snapshots):
    path = out / "robustness/results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    assert data["source_sha256"] in snapshots
    assert (
        data["analysis_script_sha256"]
        == hashlib.sha256(
            (ROOT / "scripts/analyze_lpx_robustness.py").read_bytes()
        ).hexdigest()
    )
    for entry in data["input_results"]:
        source_path = Path(entry["path"])
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == entry["sha256"]
        assert entry["source_sha256"] in snapshots
    counts = Counter()
    assert len(data["reserve_matrix"]) == 24
    for row in data["reserve_matrix"]:
        candidates = row["native_memory_minima"]
        assert len(candidates) == 3
        for candidate in candidates:
            memory = candidate["memory"]
            assert memory["minimum_headroom_bytes"] >= 0
            assert memory["peak_rank"]["capacity_bytes"] == (
                500000000 - row["extra_reserve_bytes"]
            )
            counts["native_capacity_admissions"] += 1
        for target in ("common_3400", "published_targets"):
            selected = row[target]
            assert selected["acceptance"][target]
            assert selected["native_memory"]["minimum_headroom_bytes"] >= 0
    for boundary in data["execution_boundaries"]:
        lower, upper = boundary["native_limit_bracket_us"]
        assert lower < boundary["estimated_instruction_limit_us"] < upper
        assert isclose(upper - lower, 0.0002)
        for check in boundary["native_serial_checks"]:
            point = Path(check["result_file"])
            manifest = json.loads(point.with_name("manifest.json").read_text())
            assert manifest["key"] == check["checkpoint_key"]
            assert hashlib.sha256(point.read_bytes()).hexdigest() == (
                manifest["artifacts"]["result.json"]["sha256"]
            )
            result = json.loads(point.read_text())
            assert result["acceptance"]["published_targets"] == check["passed"]
            assert check["passed"] == (check["instruction_floor_us"] == lower)
            for work, base in zip(result["workloads"], boundary["workloads"]):
                expected = (
                    base["base_seconds_per_step"]
                    + (check["instruction_floor_us"] - 0.05)
                    * base["seconds_per_instruction_us"]
                )
                assert isclose(1 / work["generation_tps"], expected, rel_tol=1e-9)
                assert isclose(
                    integrated_seconds(work["interpolation"]),
                    work["generation_seconds"],
                    rel_tol=1e-12,
                )
                replay = work["serial_replay"]
                assert replay["clients"] == 1
                assert len(replay["request_timings"]) == 55
                assert replay["executed_decode_steps"] == 55 * (
                    work["output_tokens"] - 1
                )
                for request in replay["request_timings"]:
                    assert isclose(
                        request["last_output_at"] - request["first_output_at"],
                        work["generation_seconds"],
                        rel_tol=1e-9,
                    )
                counts["serial_workloads"] += 1
                counts["requests"] += len(replay["request_timings"])
                counts["decode_forwards"] += replay["executed_decode_steps"]
    reserved = data["reserved_complete_request"]
    point = Path(reserved["result_file"])
    manifest = json.loads(point.with_name("manifest.json").read_text())
    assert manifest["key"] == reserved["checkpoint_key"]
    assert hashlib.sha256(point.read_bytes()).hexdigest() == (
        manifest["artifacts"]["result.json"]["sha256"]
    )
    request = json.loads(point.read_text())
    assert request == reserved["result"]
    assert request["full_prefill_executed"] and request["input_tokens"] == 100000
    assert request["executed_prefill_chunks"] == 6250
    assert request["executed_decode_steps"] == 1999
    assert request["memory"]["minimum_headroom_bytes"] >= 0
    assert request["memory"]["peak_rank"]["capacity_bytes"] == (
        500000000 - reserved["extra_reserve_bytes"]
    )
    counts["complete_prefill_tokens"] += request["input_tokens"]
    counts["complete_request_decode_forwards"] += request["executed_decode_steps"]
    return dict(
        status="passed",
        source_sha256=data["source_sha256"],
        result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        counts=dict(counts),
        full_result=result_location(path),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-execution-study"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "experiments/lpx/execution-placement-results.json",
    )
    args = parser.parse_args()
    out = args.output.resolve()
    snapshots = {source_identity(): "current checkout"}
    for filename in (
        "source-snapshot",
        "final-source-snapshot",
        "compatibility-source-snapshot",
    ):
        if (
            not (out / f"{filename}.json").exists()
            and not (out / f"{filename}.tar.gz").exists()
        ):
            continue
        metadata = json.loads((out / f"{filename}.json").read_text())
        archive = out / f"{filename}.tar.gz"
        if "archive_sha256" in metadata:
            assert (
                hashlib.sha256(archive.read_bytes()).hexdigest()
                == metadata["archive_sha256"]
            )
        digest = archive_identity(archive)
        assert digest == metadata["source_sha256"]
        snapshots[digest] = filename
    counts = Counter()
    scenarios, max_error = {}, 0.0
    for name in NAMES:
        data = json.loads((out / name / "results.json").read_text())
        digest = data["source_sha256"]
        assert digest in snapshots
        assert data["summary"]["points"] == len(data["rows"]) == 236
        assert (
            len({json.dumps(r["layout"], sort_keys=True) for r in data["rows"]}) == 236
        )
        expected = {
            (tp, pp)
            for tp in (1, 2, 4, 8, 16, 32)
            for pp in range(1, min(60, 256 // tp) + 1)
        }
        assert {
            (r["layout"]["a_tp"], r["layout"]["a_pp"]) for r in data["rows"]
        } == expected
        assert len(data["receipts"]) == len(data["rows"]) + len(data["verified"])
        for receipt, row in zip(data["receipts"], data["rows"] + data["verified"]):
            path = Path(receipt["result_file"])
            manifest = json.loads(path.with_name("manifest.json").read_text())
            signature = manifest["signature"]
            key = hashlib.sha256(
                json.dumps(
                    signature,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            assert key == receipt["key"] == manifest["key"]
            assert manifest["status"] == "complete"
            assert signature["parameters"]["settings"] == data["settings"]
            assert signature["parameters"]["layout"] == row["layout"]
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest()
                == manifest["artifacts"]["result.json"]["sha256"]
            )
            assert json.loads(path.read_text()) == row
            counts["verified_checkpoint_files"] += 1
        for row in data["rows"] + data["verified"]:
            if row["status"] != "ok":
                continue
            tp, pp = row["layout"]["a_tp"], row["layout"]["a_pp"]
            assert tp * pp == row["lp30"] == row["memory"]["ranks"] <= 256
            assert row["memory"]["maximum_context"] == 102000
            assert row["memory"]["minimum_headroom_bytes"] >= 0
            peak = row["memory"]["peak_rank"]
            assert peak["total_bytes"] == sum(
                peak[k] for k in ("weights_bytes", "buffers_bytes", "kv_bytes")
            )
            assert peak["total_bytes"] <= peak["capacity_bytes"] == 500000000
            if data["settings"].get("pipeline_partition", "balanced") == "capacity":
                layer_counts = row["pipeline_layer_counts"]
                assert len(layer_counts) == pp and sum(layer_counts) == 60
                assert all(type(n) is int and n > 0 for n in layer_counts)
                assert layer_counts == row["partition_evidence"]["layer_counts"]
                assert (
                    max(map(max, row["partition_evidence"]["stage_rank_bytes"]))
                    <= 500000000
                )
            passed_common, passed_published = True, True
            assert len(row["workloads"]) == 3
            for work, (inputs, outputs, target) in zip(row["workloads"], WORKLOADS):
                assert (work["input_tokens"], work["output_tokens"]) == (
                    inputs,
                    outputs,
                )
                rate = work["generation_tps"]
                assert isclose(
                    rate * work["generation_seconds"], outputs - 1, rel_tol=1e-12
                )
                passed_common &= rate >= 3400
                passed_published &= rate >= target
                curve = work["interpolation"]
                assert (curve["low"], curve["high"]) == (
                    inputs + 1,
                    inputs + outputs - 1,
                )
                assert isclose(
                    integrated_seconds(curve), work["generation_seconds"], rel_tol=1e-12
                )
                for probe in curve["probes"] + work.get("withheld_probes", []):
                    assert probe["source_sha256"] == digest
                    if "error_s" in probe:
                        max_error = max(max_error, probe["error_s"])
                        assert probe["error_s"] <= curve["tolerance_s"]
                        counts["withheld_probes"] += 1
                replay = work.get("serial_replay")
                if replay is None:
                    continue
                counts["verified_workloads"] += 1
                assert replay["clients"] == 1
                assert (replay["measured_requests"], replay["warmup_requests"]) == (
                    50,
                    5,
                )
                assert len(replay["request_timings"]) == 55
                assert replay["executed_decode_steps"] == 55 * (outputs - 1)
                previous = 0.0
                for index, request in enumerate(replay["request_timings"]):
                    assert request["request_id"] == index
                    assert request["warmup"] == (index < 5)
                    assert request["first_output_at"] >= previous
                    elapsed = request["last_output_at"] - request["first_output_at"]
                    assert isclose(elapsed, work["generation_seconds"], rel_tol=1e-9)
                    assert isclose((outputs - 1) / elapsed, rate, rel_tol=1e-9)
                    assert request["final_context"] == inputs + outputs - 1
                    previous = request["last_output_at"]
                counts["requests"] += 55
                counts["measured_requests"] += 50
                counts["decode_forwards"] += replay["executed_decode_steps"]
            assert row["acceptance"]["common_3400"] == passed_common
            assert row["acceptance"]["published_targets"] == passed_published
        for kind in ("common_3400", "published_targets"):
            valid = [
                r for r in data["rows"] if r["status"] == "ok" and r["acceptance"][kind]
            ]
            best = data["summary"][kind]
            assert (best is None) == (not valid)
            if valid:
                assert best["lp30"] == min(r["lp30"] for r in valid)
        counts["screened_layout_scenarios"] += len(data["rows"])
        counts["verified_candidates"] += len(data["verified"])
        scenarios[name] = dict(
            source_sha256=digest,
            settings=data["settings"],
            weight_dtype=data["model"]["dtype"],
            host_seconds=data["host_seconds"],
            counts={
                k: data["summary"][k] for k in ("points", "feasible", "infeasible")
            },
            selected={
                k: compact(data["summary"][k])
                for k in ("common_3400", "published_targets", "fastest")
            },
            feasible=[compact(r) for r in data["rows"] if r["status"] == "ok"],
            verified_layouts=[r["layout"] for r in data["verified"]],
            full_result=result_location(out / name / "results.json"),
        )
    prefill = json.loads((out / "prefill-validation.json").read_text())
    workspace = json.loads((out / "prefill-workspace-capacity.json").read_text())
    assert prefill["source_sha256"] == workspace["source_sha256"]
    assert prefill["source_sha256"] in snapshots
    assert (
        prefill["validation_script_sha256"]
        == hashlib.sha256(
            (ROOT / "scripts/validate_lpx_placement.py").read_bytes()
        ).hexdigest()
    )
    assert len(workspace["rows"]) == 24
    for row in prefill["results"]:
        assert row["memory"]["minimum_headroom_bytes"] >= 0
        assert row["executed_decode_steps"] == row["output_tokens"] - 1
        if row["full_prefill_executed"]:
            assert row["executed_prefill_chunks"] == -(
                -row["input_tokens"] // row["prefill_chunk"]
            )
            counts["complete_prefill_tokens"] += row["input_tokens"]
    assert counts["complete_prefill_tokens"] == 101000
    resume = None
    if (out / "resume-audit.json").exists():
        resume = json.loads((out / "resume-audit.json").read_text())
        original = json.loads((out / "capacity-floor-005/results.json").read_text())
        resumed = json.loads(
            (out / "capacity-floor-005/resumed-results.json").read_text()
        )
        for field in resume["unchanged_fields"]:
            assert original[field] == resumed[field]
        assert all(receipt["reused"] for receipt in resumed["receipts"])
        assert len(resumed["receipts"]) == resume["reused_points"]
        assert original["host_seconds"] == resume["original_host_seconds"]
        assert resumed["host_seconds"] == resume["resumed_host_seconds"]
        resume["source_sha256"] = original["source_sha256"]
    compatibility = None
    if (out / "compatibility-audit.json").exists():
        compatibility = json.loads((out / "compatibility-audit.json").read_text())
        assert compatibility["status"] == "passed"
        assert compatibility["source_sha256"] in snapshots
    audit = dict(
        status="passed", counts=dict(counts), maximum_withheld_error_s=max_error
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    portable = dict(
        schema="tokensim.lpx_execution_placement_study",
        version=1,
        scope="Conditional pure-LPX decode search; uncalibrated timing assumptions and unverified prefill workspace.",
        workloads=[
            dict(input_tokens=i, output_tokens=o, published_or_user_target_tps=t)
            for i, o, t in WORKLOADS
        ],
        audit=audit,
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scenarios=scenarios,
        source_snapshots=snapshots,
        prefill_validation=prefill,
        prefill_workspace=workspace,
        resume_validation=resume,
        compatibility_validation=compatibility,
        robustness_validation=audit_robustness(out, snapshots),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(portable, indent=2, allow_nan=False) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
