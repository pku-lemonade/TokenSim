"""Project zero-handoff timing bounds over fixed uniform-layout SRAM frontiers.

Projection assumes per-layer kernel and intra-stage collective costs are invariant
under stage translation and contiguous cuts. It is conditional model evidence,
not a bound on alternative kernels, memory policies or physical LPX hardware.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError
from scripts.lpx.mfu_metrics import critical_path_seconds
from scripts.lpx.replay_engine import ReplayNode
from scripts.lpx.replay_profiles import NativeProfile
from scripts.search_lpx_native import write_json


def saved_profile(path, expected_input):
    with gzip.open(path, "rt") as stream:
        raw = json.load(stream)
    body = hashlib.sha256(
        json.dumps(raw["nodes"], separators=(",", ":")).encode()
    ).hexdigest()
    if raw["input_sha256"] != expected_input or body != raw["body_sha256"]:
        raise ConfigurationError("partition projection profile fingerprint differs")
    return NativeProfile(
        tuple(
            ReplayNode(tuple(row[0]), tuple(row[1]), *row[2:]) for row in raw["nodes"]
        ),
        raw["metadata"],
    )


def stage_chips(resources):
    return {
        int(match.group(1))
        for resource in resources
        if (match := re.search(r"(?:lp30:|/worker:)(\d+)(?::|$)", resource))
    }


def zero_handoff_bound(profile, *, tp, pp):
    """Remove direct PP wire and propagation costs, preserving dependencies."""
    integer(tp, "projection tensor width", minimum=1)
    integer(pp, "projection pipeline depth", minimum=1)
    fabric = profile.metadata["streaming"]["fabric"]["config"]
    if fabric["routes"]:
        raise ConfigurationError("partition projection requires default direct routes")
    children = [[] for _ in profile.nodes]
    for index, node in enumerate(profile.nodes):
        for parent in node.predecessors:
            if parent < 0 or parent >= len(profile.nodes):
                raise ConfigurationError("projection has an invalid predecessor")
            children[parent].append(index)
    removed = set()
    wires = 0
    for index, node in enumerate(profile.nodes):
        chips = stage_chips(node.resources)
        if any(chip >= tp * pp for chip in chips):
            raise ConfigurationError("projection resource exceeds its chip inventory")
        if len({chip // tp for chip in chips}) <= 1:
            continue
        if (
            any(
                not re.fullmatch(r"lp30:\d+:(?:port:\d+:)?(?:tx|rx)", key)
                for key in node.resources
            )
            or len(children[index]) != 1
        ):
            raise ConfigurationError("cross-stage work is not a direct handoff")
        flight = children[index][0]
        next_node = profile.nodes[flight]
        if (
            next_node.predecessors != (index,)
            or next_node.resources
            or next_node.duration
            or next_node.occupancy
        ):
            raise ConfigurationError("handoff propagation shape differs")
        removed.update((index, flight))
        wires += 1
    if wires != tp * (pp - 1):
        raise ConfigurationError("projection handoff count differs from the layout")
    relaxed = tuple(
        (
            replace(node, resources=(), primary=None, duration=0, occupancy=0, delay=0)
            if index in removed
            else node
        )
        for index, node in enumerate(profile.nodes)
    )
    original = critical_path_seconds(profile.nodes)
    lower = critical_path_seconds(relaxed)
    return dict(
        critical_path_s=original,
        zero_handoff_critical_path_s=lower,
        removed_critical_path_s=original - lower,
        handoff_wires=wires,
        removed_nodes=len(removed),
    )


def projected_frontier(capacity_row, *, batch, latency_s, flops, peak_per_chip):
    integer(batch, "projection batch", minimum=1)
    for value, name in (
        (latency_s, "latency"),
        (flops, "FLOPs"),
        (peak_per_chip, "peak"),
    ):
        number(value, f"projection {name}", positive=True)
    tp = capacity_row["tp"]
    points = []
    for pp, resident in capacity_row["maximum_resident_by_pp"].items():
        pp = int(pp)
        effective = resident // batch * batch
        if effective:
            points.append(
                dict(
                    pp=pp,
                    chips=tp * pp,
                    optimistic_resident_requests=effective,
                    aggregate_tps_upper=effective / latency_s,
                    mfu_percent_upper=100
                    * effective
                    * flops
                    / (latency_s * tp * pp * peak_per_chip),
                )
            )
    return points


def family_key(case):
    return case["tp"], case["kvp"], case["kv_parallel_layers"]


def matrix_latency_bound(work, *, batch, tp, fp8_peak, bf16_peak, gemm, attention):
    """Necessary arithmetic time with serial native layers and shared matrix units.

    Useful FLOPs undercount replicated projections. The bound ignores memory,
    scalar work and communication, and requires the native no-layer-overlap rule.
    """
    integer(batch, "matrix bound cohort width", minimum=1)
    integer(tp, "matrix bound tensor width", minimum=1)
    for value in (fp8_peak, bf16_peak):
        number(value, "matrix bound positive rate", positive=True)
    for value in (gemm, attention):
        number(value, "matrix bound efficiency", positive=True, maximum=1)
    for key in ("linear", "attention"):
        number(work[key], f"matrix bound {key} work")
    return (
        batch
        / tp
        * (
            work["linear"] / (fp8_peak * gemm)
            + work["attention"] / (bf16_peak * attention)
        )
    )


def analyze_reference(job):
    path, key, case, context, source = job
    profile = saved_profile(path, key)
    metadata = profile.metadata
    if (
        metadata["batch"] != case["batch"]
        or metadata["phase"] != "decode"
        or metadata["context"] != context
        or metadata["model_id"] != "gemma-4-31b-text"
        or metadata["source_sha256"] != source
    ):
        raise ConfigurationError("projection profile workload differs from its record")
    return dict(
        case=case,
        input_sha256=key,
        **zero_handoff_bound(profile, tp=case["tp"], pp=case["pp"]),
    )


def width_coverage(progress, capacity, groups):
    work = progress["metric"]["useful_flops_per_token"]
    peak = progress["metric"]["fp8_peak_per_chip"]
    compute = progress["shared_timing_config"]["compute"]
    best = progress["best_observed"]["mfu_percent"]
    references = {(*family_key(row), row["batch"]): row for row in groups}
    rows = []
    for family in capacity.values():
        frontier = family.get("maximum_resident_by_pp", {})
        for batch in range(1, max(frontier.values(), default=0) + 1):
            key = (*family_key(family), batch)
            lower = matrix_latency_bound(
                work,
                batch=batch,
                tp=family["tp"],
                fp8_peak=peak,
                bf16_peak=progress["metric"].get("bf16_peak_per_chip", peak / 2),
                gemm=compute["gemm_mfu"],
                attention=compute["attention_mfu"],
            )
            reference = references.get(key)
            if reference:
                lower = max(lower, reference["zero_handoff_critical_path_s"])
            points = projected_frontier(
                family,
                batch=batch,
                latency_s=lower,
                flops=work["total"],
                peak_per_chip=peak,
            )
            upper = max(point["mfu_percent_upper"] for point in points)
            status = (
                "excluded_by_bound"
                if upper < best
                else "requires_direct_layout_bound" if reference else "missing_profile"
            )
            rows.append(
                dict(
                    tp=key[0],
                    kvp=key[1],
                    kv_parallel_layers=key[2],
                    batch=batch,
                    status=status,
                    mfu_percent_upper=upper,
                    latency_lower_s=lower,
                )
            )
    return dict(
        widths=len(rows),
        missing_profiles=sum(row["status"] == "missing_profile" for row in rows),
        excluded=sum(row["status"] == "excluded_by_bound" for row in rows),
        rows=rows,
        scope="Only the supplied conservative footprint family and serial native layer model; complete equal-width cohorts and the catalog BF16 peak convention apply.",
    )


def project(
    progress,
    additional,
    profile_paths,
    *,
    capacity_record=None,
    allow_memory_changes=False,
    workers=1,
):
    integer(workers, "projection workers", minimum=1)
    records = [*progress["timing_bounds"], *progress["observed_probes"]]
    for document in additional:
        records.extend(
            row["result"]
            for row in document["results"]
            if row["result"]["status"] in {"bounded", "ok"}
        )
    index = {path.stem.removesuffix(".json"): path for path in profile_paths}
    grouped = defaultdict(list)
    jobs = {}
    for row in records:
        case = row["case"]
        expected_config = deepcopy(progress["shared_timing_config"])
        if "memory" in expected_config:
            expected_config["memory"]["prefill_chunk"] = max(
                expected_config["memory"]["prefill_chunk"], case["batch"]
            )
        actual_config = row.get("config", expected_config)
        if allow_memory_changes:
            actual_config = {k: v for k, v in actual_config.items() if k != "memory"}
            expected_config.pop("memory", None)
        if actual_config != expected_config:
            raise ConfigurationError("projection mixes different timing configurations")
        identity = row.get("profile_identity", row.get("profile"))
        key = identity["input_sha256"]
        if key not in index:
            raise ConfigurationError(f"missing captured profile: {key}")
        jobs[key] = (
            index[key],
            key,
            case,
            progress["metric"]["decode_context"],
            row["source_sha256"],
        )
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            evidence = list(pool.map(analyze_reference, jobs.values()))
    else:
        evidence = [analyze_reference(job) for job in jobs.values()]
    for reference in evidence:
        case = reference["case"]
        grouped[(*family_key(case), case["batch"])].append(reference)
    rows = []
    work = progress["metric"]["useful_flops_per_token"]["total"]
    peak = progress["metric"]["fp8_peak_per_chip"]
    best = progress["best_observed"]["mfu_percent"]
    capacity_record = capacity_record or progress["capacity_screen"]
    actual_config = dict(capacity_record["config"])
    expected_config = dict(progress["shared_timing_config"])
    if allow_memory_changes:
        actual_config.pop("memory", None)
        expected_config.pop("memory", None)
    if actual_config != expected_config:
        raise ConfigurationError("projection capacity timing configuration differs")
    capacity = {family_key(row): row for row in capacity_record["rows"]}
    for (tp, kvp, mode, batch), references in sorted(grouped.items()):
        costs = [row["zero_handoff_critical_path_s"] for row in references]
        lower = min(costs)
        invariant = max(costs) - lower <= 1e-10
        if not invariant:
            raise ConfigurationError("reference cuts disagree on zero-handoff cost")
        points = projected_frontier(
            capacity[tp, kvp, mode],
            batch=batch,
            latency_s=lower,
            flops=work,
            peak_per_chip=peak,
        )
        if not points:
            continue
        rows.append(
            dict(
                tp=tp,
                kvp=kvp,
                kv_parallel_layers=mode,
                batch=batch,
                references=references,
                zero_handoff_critical_path_s=lower,
                projected_points=len(points),
                optimistic_peak=max(
                    points, key=lambda point: point["mfu_percent_upper"]
                ),
                competitive_points=[
                    point for point in points if point["mfu_percent_upper"] >= best
                ],
            )
        )
    return dict(
        best_observed_mfu_percent=best,
        projected_groups=len(rows),
        projected_points=sum(row["projected_points"] for row in rows),
        rows=rows,
        capacity_config=capacity_record["config"],
        allow_memory_policy_differences=allow_memory_changes,
        width_coverage=width_coverage(progress, capacity, rows),
        scope="Conditional zero-handoff projections across uniform contiguous PP cuts with translation-invariant kernels; footprint vectors and complete equal-width cohorts constrain coverage. This is not a global hardware optimum.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--additional-results", type=Path, nargs="*", default=[])
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity-record", type=Path)
    parser.add_argument("--allow-memory-policy-differences", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        progress = json.loads(args.progress.read_text())
        additional = [json.loads(path.read_text()) for path in args.additional_results]
        result = project(
            progress,
            additional,
            args.profile_root.rglob("*.json.gz"),
            capacity_record=(
                json.loads(args.capacity_record.read_text())
                if args.capacity_record
                else None
            ),
            allow_memory_changes=args.allow_memory_policy_differences,
            workers=args.workers,
        )
        write_json(args.output, result)
        print(
            json.dumps(
                dict(
                    groups=result["projected_groups"],
                    points=result["projected_points"],
                    competitive=sum(
                        len(row["competitive_points"]) for row in result["rows"]
                    ),
                )
            )
        )
        return 0
    except (ConfigurationError, OSError, ValueError, KeyError) as exc:
        print(f"LPX partition bounds: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
