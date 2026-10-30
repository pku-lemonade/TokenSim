"""Run C2 or C3 analysis only after the corresponding full collection validates."""

import argparse
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.errors import WorkloadValidationError
from scripts import analyze_lpx_cc16k_peak as peak
from scripts import run_lpx_gpu_cc302 as collector
from scripts import run_lpx_gpu_study as study
from scripts.analyze_lpx_gpu_smoke import read_outputs
from scripts.lpx.gpu_profile import file_record
from scripts.monitor_lpx_gpu_study import save
from scripts.run_lpx_gpu_followup import CODING, validate_counters, validate_outputs


def export(requests, outputs, directory):
    study.run_script(
        "generate_lpx_gpu_outputs.py",
        "--requests",
        requests,
        "--outputs",
        outputs,
        "--run-label",
        "export",
        "--export",
        directory,
    )
    return directory


def earlier_rows(rows, identities):
    selected = [record for record in rows if record["row"]["id"] in identities]
    actual = [(record["row"]["id"], record["row"]["turn"]) for record in selected]
    if (
        len(identities) != 40
        or len(actual) != 40
        or set(actual) != {(identity, 0) for identity in identities}
    ):
        raise WorkloadValidationError(
            "Earlier subset must contain each of the 40 IDs exactly once"
        )
    return selected


def row_audit(path):
    rows = study.read_json(path)["rows"]
    return dict(
        **file_record(path),
        identities=[
            dict(id=record["row"]["id"], turn=record["row"]["turn"]) for record in rows
        ],
    )


def peak_identities(lookup, prices, audits, step):
    rows = {
        (row["data"], row["minimum_ngram"], row["draft_tokens"]): row["per_row"]
        for row in lookup["results"]
        if row.get("prompt") == "full" and "per_row" in row
    }
    results = []
    for setting in prices["results"]:
        for dataset, field in (
            (peak.COMPLETION, "completion"),
            (peak.EARLIER, "earlier_40"),
        ):
            selected = rows[
                (dataset, setting["minimum_ngram"], setting["draft_tokens"])
            ]
            identities = audits[dataset]["identities"]
            if len(identities) != len(selected):
                raise WorkloadValidationError("Replay row identities differ in count")
            for priced in setting[field]:
                speeds = peak.lookup_speeds(selected, step, priced["extra_us"])
                maximum = float(max(speeds))
                if abs(maximum - priced["maximum_tps"]) > 1e-8:
                    raise WorkloadValidationError(
                        "Recomputed peak differs from pricing artifact"
                    )
                results.append(
                    dict(
                        data=dataset,
                        minimum_ngram=setting["minimum_ngram"],
                        draft_tokens=setting["draft_tokens"],
                        extra_us=priced["extra_us"],
                        maximum_tps=maximum,
                        fastest=[
                            identity
                            for identity, speed in zip(identities, speeds)
                            if speed == maximum
                        ],
                    )
                )
    return dict(results=results, rows=audits)


def lookup(work, requests):
    outputs = collector.artifact("plain-outputs.jsonl")
    control = study.artifact("followup-plain-k0-outputs.jsonl", "triton")
    collector.validate(work, requests, outputs, "plain", 0)
    validate_outputs(work, control, [CODING], "plain", 0)
    directory = work / "cc302/rows"
    plain = export(requests, outputs, directory / "plain") / "speed-cc16k-plain.json"
    coding = (
        export(work / "inputs/requests.jsonl", control, directory / "coding")
        / "speed-coding-plain.json"
    )
    earlier = directory / "speed-cc16k40-plain.json"
    identities = {
        row["id"]
        for row in read_outputs(work / "inputs/requests.jsonl")
        if row["set"] == "speed-cc16k"
    }
    save(earlier, dict(rows=earlier_rows(study.read_json(plain)["rows"], identities)))
    paths = {peak.CODING: coding, peak.COMPLETION: plain, peak.EARLIER: earlier}
    arguments = ["--gemma-tokenizer", work / "inputs/tokenizer.json"]
    for dataset, path in paths.items():
        arguments += ["--data", f"{dataset}={path}#answer-alone"]
    replay = collector.artifact("lookup-results.json")
    summary = collector.artifact("peak.json")
    study.run_script(
        "measure_lpx_lookup_acceptance.py", *arguments, "--per-row", "--summary", replay
    )
    study.run_script(
        "analyze_lpx_cc16k_peak.py", "--lookup", replay, "--summary", summary
    )
    step = peak.context_model(
        peak.verification_steps(
            peak.load("article-sweep-results.json"),
            peak.load("article-speculative-results.json"),
        )
    )
    save(
        collector.artifact("peak-identities.json"),
        peak_identities(
            study.read_json(replay),
            study.read_json(summary),
            {dataset: row_audit(path) for dataset, path in paths.items()},
            step,
        ),
    )


def verify_sources(path):
    audit = study.read_json(path)
    records = []
    for name, digest in audit["source_hashes"].items():
        source = Path(name)
        record = file_record(source if source.is_absolute() else ROOT / source)
        if record["sha256"] != digest:
            raise WorkloadValidationError(
                f"Previously audited source changed: {source}"
            )
        records.append(record)
    return dict(source_audit=file_record(path), sources=records)


def native(work, requests):
    outputs = collector.artifact("ngram-k7-outputs.jsonl")
    plain = collector.artifact("plain-outputs.jsonl")
    collector.validate(work, requests, plain, "plain", 0)
    collector.validate(work, requests, outputs, "ngram", 7)
    audit = verify_sources(study.artifact("followup-ngram-source-audit.json", "triton"))
    control = study.artifact("mtp-k3-smoke.jsonl", "triton")
    validate_counters(read_outputs(control), "mtp", 3)
    audit.update(plain_control=file_record(plain), mtp_control=file_record(control))
    save(collector.artifact("ngram-source-audit.json"), audit)
    directory = export(requests, outputs, work / "cc302/rows/ngram")
    study.run_script(
        "analyze_lpx_gpu_ngram.py",
        "--outputs",
        outputs,
        "--rows",
        directory,
        "--gemma-tokenizer",
        work / "inputs/tokenizer.json",
        "--summary",
        collector.artifact("ngram-comparison.json"),
        "--server-command",
        collector.artifact("ngram-k7-server-command.json"),
        "--validated-engine-steps",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/lpx-gpu"))
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--phase", choices=("lookup", "native", "all"), required=True)
    args = parser.parse_args(argv)
    if args.work_dir.resolve().is_relative_to(
        ROOT
    ) or args.requests.resolve().is_relative_to(ROOT):
        parser.error(
            "work-dir and prompt-containing requests must remain outside the repository"
        )
    directory = args.work_dir / "cc302"
    directory.mkdir(exist_ok=True)
    with (directory / "analysis.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = study.read_json(collector.artifact("collection-status.json"))
        required = {
            "lookup": {"plain"},
            "native": {"plain", "ngram-k7"},
            "all": {"plain", "ngram-k7"},
        }[args.phase]
        if not required <= set(status.get("completed", [])):
            raise WorkloadValidationError(
                "Requested collection stages are not yet complete"
            )
        collector.audit_inputs(args.work_dir, args.requests)
        if args.phase in ("lookup", "all"):
            lookup(args.work_dir, args.requests)
        if args.phase in ("native", "all"):
            native(args.work_dir, args.requests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
