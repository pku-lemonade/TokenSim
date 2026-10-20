"""Export LPX study frontiers and processor-power sensitivity from point indexes."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import hashlib
import json
from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.lpx.frontier import pareto
from scripts.lpx_power import POWER_SCOPES, processor_power_rows
from scripts.lpx.plotting import draw_simulation, style_axis, save


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def point_source(row, verify_bytes=False):
    manifest_path = Path(row["manifest_file"])
    manifest = json.loads(manifest_path.read_text())
    if manifest["key"] != row["checkpoint_key"] or manifest["status"] != "complete":
        raise ValueError("point index does not match a committed checkpoint")
    if row.get("point_audit") != "passed":
        raise ValueError("study point has no completed accounting audit")
    if verify_bytes:
        artifact = manifest["artifacts"]["result.json"]
        with Path(row["result_file"]).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != artifact["sha256"]:
            raise ValueError("point result checksum differs from its manifest")
    return dict(
        result_file=row["result_file"],
        manifest_file=str(manifest_path),
        checkpoint_key=manifest["key"],
        elapsed_seconds=manifest["elapsed_seconds"],
    )


def power_rows(rows, scope, *, rubin_watts=1800, lp30_watts=450):
    result = processor_power_rows(
        rows, scope, rubin_watts=rubin_watts, lp30_watts=lp30_watts
    )
    baseline = max(r["tokens_per_s_per_mw"] for r in result if r["mode"] == "rubin")
    for row in result:
        row["normalized_throughput"] = row["tokens_per_s_per_mw"] / baseline
    return result, baseline


def observed_comparisons(
    rows,
    generation_floors=(100, 250, 500, 1000, 2000, 4000),
    efficiency_floors=(0.1, 0.25, 0.5, 0.75, 1.0),
):
    result = {}
    for name, thresholds, constraint, objective in (
        (
            "minimum_generation_tps",
            generation_floors,
            "tokens_per_s_per_user",
            "tokens_per_s_per_mw",
        ),
        (
            "minimum_normalized_efficiency",
            efficiency_floors,
            "normalized_throughput",
            "tokens_per_s_per_user",
        ),
    ):
        comparisons = []
        for threshold in thresholds:
            selected = {}
            for mode in ("rubin", "af", "pd", "draft"):
                feasible = [
                    r for r in rows if r["mode"] == mode and r[constraint] >= threshold
                ]
                selected[mode] = (
                    max(feasible, key=lambda r: (r[objective], r[constraint]))
                    if feasible
                    else None
                )
            baseline = selected["rubin"]
            comparisons.append(
                dict(
                    threshold=threshold,
                    objective=objective,
                    modes={
                        mode: (
                            dict(
                                case=row["case"],
                                users=row["batch"],
                                value=row[objective],
                                relative_to_rubin=(
                                    row[objective] / baseline[objective]
                                    if baseline is not None
                                    else None
                                ),
                            )
                            if row is not None
                            else None
                        )
                        for mode, row in selected.items()
                    },
                )
            )
        result[name] = comparisons
    return result


def export_case(output, index, *, verify_bytes=False):
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        f"{r['case']}:{r['batch']}": point_source(r, verify_bytes)
        for r in index["results"]
    }
    figures = []
    datasets = {}
    for scope in POWER_SCOPES:
        rows, baseline = power_rows(index["results"], scope)
        frontiers = {
            mode: pareto([r for r in rows if r["mode"] == mode])
            for mode in ("rubin", "draft", "af", "pd")
        }
        joint = pareto(rows)
        datasets[scope] = dict(
            normalization_tokens_per_s_per_mw=baseline,
            frontiers=frontiers,
            joint_frontier=joint,
            results=rows,
            comparisons=observed_comparisons(rows),
        )
        with (output / f"points-{scope}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            fields = (
                "case",
                "mode",
                "batch",
                "gpu_count",
                "lpu_count",
                "minimum_nvl72_racks",
                "minimum_lpx_racks",
                "tokens_per_s_per_user",
                "output_tokens_per_s",
                "power_watts",
                "tokens_per_s_per_mw",
                "normalized_throughput",
                "achieved_qps",
                "latency_p50_s",
                "ttft_p50_s",
                "completed_requests",
                "result_file",
            )
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                stats = row["request_statistics"]
                writer.writerow(
                    dict(
                        row,
                        achieved_qps=stats["achieved_qps"],
                        latency_p50_s=stats["latency_s"]["p50"],
                        ttft_p50_s=stats["ttft_s"]["p50"],
                        completed_requests=stats["completed_requests"],
                    )
                )
        fig, ax = plt.subplots(figsize=(10.8, 6.4))
        fig.subplots_adjust(left=0.12, right=0.97, top=0.83, bottom=0.24)
        style_axis(
            ax,
            max(r["tokens_per_s_per_user"] for r in rows) * 1.12,
            max(r["normalized_throughput"] for r in rows) * 1.12,
        )
        draw_simulation(ax, dict(results=rows, frontiers=frontiers))
        ax.set_xlabel("Per-request generation speed P50 (tokens/s/user)")
        ax.set_ylabel("Output TPS/MW, normalized to observed Rubin peak")
        fig.suptitle(index["model"]["model_id"], fontsize=15)
        fig.text(
            0.12,
            0.875,
            f"400K cached / 4K new / 400 output | {index['phase']} | sampled DAG replay",
            fontsize=10,
        )
        fig.text(
            0.12,
            0.10,
            f"Power scope: {scope}; unmeasured processor TDP hypotheses.",
            fontsize=9,
        )
        fig.text(
            0.12,
            0.06,
            "NVIDIA model, compiler mapping, full system power and tokenizer are undisclosed.",
            fontsize=9,
        )
        name = f"frontier-{scope}"
        save(fig, output, name)
        figures.extend([str(output / f"{name}.png"), str(output / f"{name}.svg")])
    summary = dict(
        schema="tokensim.lpx_study_analysis",
        version=1,
        model=index["model"],
        draft_model=index["draft_model"],
        workload=index["workload"],
        phase=index["phase"],
        source_state=index["state"],
        source_sha256=index["source_sha256"],
        cases=index["cases"],
        sources=sources,
        rejected=[r for r in index["results"] if r["status"] != "ok"],
        power_assumptions="Allocated processor TDP; optional rounding to complete 72-GPU/256-LPU processor inventories. Host/network power excluded.",
        execution_fidelity="sampled_native_operator_dags; numerical accuracy requires native controls",
        datasets=datasets,
        figures=figures,
    )
    write_json(output / "analysis.json", summary)
    return summary


def combine_indexes(indexes):
    result = deepcopy(indexes[0])
    rows, cases = {}, {}
    source_states, phases = set(), set()
    for index in indexes:
        for key in ("model", "draft_model", "workload", "source_sha256"):
            if index[key] != result[key]:
                raise ValueError(f"cannot combine different {key}")
        variant = index.get("variant", "mb1-adp1-pf0")
        source_states.add(index["state"])
        phases.add(index["phase"])
        for case in index["cases"]:
            alias = f"{case['name']}@{variant}"
            cases[alias] = dict(
                case, name=alias, original_case=case["name"], variant=variant
            )
        for row in index["results"]:
            alias = f"{row['case']}@{variant}"
            key = (alias, row["batch"])
            if key in rows:
                raise ValueError("duplicate case/user points in analysis inputs")
            rows[key] = dict(
                row, case=alias, original_case=row["case"], variant=variant
            )
    result.update(
        results=list(rows.values()),
        cases=list(cases.values()),
        state="complete" if source_states == {"complete"} else "running",
        phase=next(iter(phases)) if len(phases) == 1 else "mixed",
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-bytes", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    groups = {}
    for path in args.index:
        index = json.loads(path.read_text())
        if index.get("schema") != "tokensim.lpx_study_index":
            parser.error("expected an audited study index")
        groups.setdefault(index["model"]["model_id"], []).append(index)
    for model, indexes in groups.items():
        index = combine_indexes(indexes)
        result = export_case(
            args.output / index["model"]["model_id"],
            index,
            verify_bytes=args.verify_bytes,
        )
        print(
            f"{index['model']['model_id']}: {len(index['results'])} points; {index['state']}"
        )
        for figure in result["figures"]:
            print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
