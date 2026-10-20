"""Run the LPX workload grid and retain audited references to complete points."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from dataclasses import asdict, replace
import json
import math
import hashlib

from TokenSim.config.validation import integer, number
from TokenSim.hardware import HardwareContext
from scripts.lpx.config import load_cases
from scripts.lpx.frontier import pareto
from scripts.lpx.runner import evaluate_jobs
from scripts.lpx.replay_profiles import source_identity
from scripts.reproduce_lpx import repository_state, write_json


def study_cases(model):
    expert = model == "gpt-oss-2t-expert-proxy"
    config = ROOT / (
        "experiments/lpx/expert-sweep.json" if expert else "experiments/lpx/sweep.json"
    )
    _, workload, original = load_cases(config)
    base = {case.name: case for case in original}
    stages = 9 if expert else 8
    ruby = base["rubin-tp8"]
    pipeline = replace(ruby.decode, a_pp=stages, f_pp=stages)
    cases = [
        ruby,
        base["rubin-tp16"],
        replace(
            ruby,
            name="rubin-tp4-pp2",
            decode=replace(ruby.decode, a_tp=4, f_ep=4, a_pp=2, f_pp=2),
        ),
        replace(ruby, name="rubin-pipeline", decode=pipeline),
        base["af-tp8"],
    ]
    variants = (
        [(2, 320, 4), (4, 160, 4), (8, 80, 4)]
        if expert
        else [(2, 128, 10), (4, 128, 5), (8, 128, 3)]
    )
    for tp, ep, pp in variants:
        layout = replace(base["af-tp8"].decode, f_tp=tp, f_ep=ep, f_pp=pp)
        if layout != base["af-tp8"].decode:
            cases.append(
                replace(base["af-tp8"], name=f"af-f{tp}-ep{ep}-pp{pp}", decode=layout)
            )
    tp, ep, pp = variants[1]
    minimum = replace(base["af-tp8"].decode, f_tp=tp, f_ep=ep, f_pp=pp)
    cases.extend(
        [
            replace(
                base["af-tp8"],
                name="af-prefill-pipeline",
                decode=minimum,
                prefill=replace(base["af-tp8"].prefill, a_pp=2, f_pp=2),
            ),
            replace(
                base["af-tp8"],
                name="af-pipeline",
                decode=replace(minimum, a_pp=stages),
                prefill=pipeline,
            ),
            base["pd-tp16"],
            replace(
                base["pd-tp16"],
                name="pd-compact",
                decode=replace(
                    base["pd-tp16"].decode,
                    a_pp=18 if expert else 40,
                    f_tp=tp,
                    f_ep=ep,
                    f_pp=pp,
                ),
            ),
            base["draft-tp8"],
        ]
    )
    draft = replace(base["draft-tp8"].draft, a_tp=8, f_ep=8)
    cases.extend(
        [
            replace(
                base["draft-tp8"], name="draft-lpx256", draft=draft, draft_a_replicas=8
            ),
            replace(
                base["draft-tp8"],
                name="draft-pipeline-lpx256",
                decode=pipeline,
                draft=draft,
                draft_a_replicas=8,
            ),
        ]
    )
    return replace(workload, prefill_chunk=4000), cases


def extension_candidates(rows, limit):
    candidates = set()
    for row in rows:
        if row["status"] != "ok" or row["batch"] != limit:
            continue
        earlier = [
            r
            for r in rows
            if r["case"] == row["case"] and r["status"] == "ok" and r["batch"] < limit
        ]
        if earlier and row["output_tokens_per_s"] > 1.02 * max(
            r["output_tokens_per_s"] for r in earlier
        ):
            candidates.add(row["case"])
    return candidates


def run_grid(args, workload, cases, selected, *, phase, prior=()):
    hardware = HardwareContext.load(ROOT / "data")
    target, draft = hardware.models.resolve(args.model), hardware.models.resolve(
        "gpt-oss-draft-proxy"
    )
    revision, dirty = repository_state()
    destination = args.output / phase
    destination.mkdir(parents=True, exist_ok=True)
    iso = args.isolated_latencies
    jobs = []
    for case, clients in selected:
        formal = phase == "formal"
        settings = dict(
            checkpoint_output=str(args.output / ".experiments"),
            result_format="reference",
            resume=not args.no_resume,
            draft_spec=draft,
            stream_metrics=True,
            execution_backend="sampled-replay",
            ffn_batching=False,
            replay_samples=args.samples,
            replay_prefill_samples=1,
            replay_cache=str(args.profile_cache),
            micro_batch=args.micro_batch,
            a_dp=args.a_replicas,
            warmup_steps=args.formal_warmup_steps if formal else 2,
            warmup_seconds=max(1, 10 * iso.get(case.name, 0)) if formal else 0,
            measurement_turns=(
                max(args.minimum_turns, math.ceil(args.minimum_requests / clients))
                if formal
                else args.coarse_turns
            ),
            measurement_seconds=(
                max(10, args.duration_multiple * iso.get(case.name, 0))
                if formal
                else 0.001
            ),
        )
        jobs.append((target, workload, case, clients, settings))
    header = dict(
        schema="tokensim.lpx_study_index",
        version=1,
        phase=phase,
        variant=f"mb{args.micro_batch}-adp{args.a_replicas}-pf{int(args.prefill_tp8_pp4)}",
        git_revision=revision,
        git_worktree_dirty=dirty,
        source_sha256=source_identity(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model=target.to_dict(),
        draft_model=draft.to_dict(),
        workload=asdict(workload),
        cases=[asdict(case) for case in cases],
        execution_backend="sampled-replay",
        replay_samples=args.samples,
        ffn_batching=False,
        state="running",
    )
    completed = {}
    for index, result in evaluate_jobs(
        jobs, processes=args.formal_jobs if phase == "formal" else args.jobs
    ):
        completed[index] = result
        status = result["status"]
        print(f"{phase}: {result['case']} users={result['batch']} {status}", flush=True)
        write_json(
            destination / "index.json",
            dict(header, results=[*prior, *(completed[i] for i in sorted(completed))]),
        )
    rows = [*prior, *(completed[i] for i in sorted(completed))]
    write_json(destination / "index.json", dict(header, state="complete", results=rows))
    return rows


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--model",
        choices=("gpt-oss-2t-wide-proxy", "gpt-oss-2t-expert-proxy"),
        default="gpt-oss-2t-wide-proxy",
    )
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument(
        "--profile-cache", type=Path, default=ROOT / "outputs/lpx-replay-profile-cache"
    )
    cli.add_argument("--users", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    cli.add_argument("--maximum-users", type=int, default=512)
    cli.add_argument("--micro-batch", type=int, default=1)
    cli.add_argument("--a-replicas", type=int, default=1)
    cli.add_argument("--cases", nargs="+")
    cli.add_argument("--prefill-tp8-pp4", action="store_true")
    cli.add_argument("--jobs", type=int, default=4)
    cli.add_argument("--formal-jobs", type=int, default=2)
    cli.add_argument("--samples", type=int, default=16)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--coarse-turns", type=int, default=2)
    cli.add_argument("--minimum-requests", type=int, default=500)
    cli.add_argument("--minimum-turns", type=int, default=20)
    cli.add_argument("--formal-warmup-steps", type=int, default=2)
    cli.add_argument("--duration-multiple", type=float, default=100)
    cli.add_argument("--phase", choices=("coarse", "formal", "all"), default="all")
    cli.add_argument("--no-resume", action="store_true")
    args = cli.parse_args(argv)
    for name in (
        "jobs",
        "micro_batch",
        "a_replicas",
        "formal_jobs",
        "samples",
        "coarse_turns",
        "minimum_requests",
        "minimum_turns",
        "formal_warmup_steps",
    ):
        integer(getattr(args, name), name, minimum=1)
    for users in args.users:
        integer(users, "users", minimum=1)
    integer(args.maximum_users, "maximum users", minimum=max(args.users))
    number(args.duration_multiple, "duration multiple", positive=True)
    if len(set(args.users)) != len(args.users):
        cli.error("user counts must be unique")
    args.output = args.output.resolve()
    args.profile_cache = args.profile_cache.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    workload, cases = study_cases(args.model)
    if args.cases:
        unknown = set(args.cases) - {c.name for c in cases}
        if unknown:
            cli.error(f"unknown study cases: {sorted(unknown)}")
        cases = [c for c in cases if c.name in args.cases]
    if args.prefill_tp8_pp4:
        cases = [
            (
                replace(c, prefill=replace(c.prefill, a_tp=8, f_ep=8, a_pp=4, f_pp=4))
                if c.prefill
                else c
            )
            for c in cases
        ]
    workload = replace(workload, seed=args.seed)
    write_json(
        args.output / "configuration.json",
        dict(
            schema_version=1,
            workload=asdict(workload),
            batches=args.users,
            cases=[asdict(c) for c in cases],
        ),
    )
    args.isolated_latencies = {}
    if args.phase in {"coarse", "all"}:
        rows = run_grid(
            args,
            workload,
            cases,
            [(c, n) for c in cases for n in args.users],
            phase="coarse",
        )
    else:
        previous = json.loads((args.output / "coarse/index.json").read_text())
        if (
            previous["model"]["model_id"] != args.model
            or previous["workload"] != asdict(workload)
            or previous["cases"] != [asdict(c) for c in cases]
            or previous["source_sha256"] != source_identity()
        ):
            cli.error(
                "formal phase requires matching coarse model, workload, configurations and code"
            )
        rows = previous["results"]
    if args.phase in {"coarse", "all"}:
        limit = max(args.users)
        while limit < args.maximum_users:
            candidates = extension_candidates(rows, limit)
            if not candidates:
                break
            limit *= 2
            args.users.append(limit)
            rows = run_grid(
                args,
                workload,
                cases,
                [(case, limit) for case in cases if case.name in candidates],
                phase="coarse",
                prior=rows,
            )
    args.users = sorted({r["batch"] for r in rows})
    if args.phase == "coarse":
        return 0
    for row in rows:
        if row["status"] == "ok" and row["batch"] == 1:
            args.isolated_latencies[row["case"]] = row["request_statistics"][
                "latency_s"
            ]["mean"]
    frontier = []
    for mode in ("rubin", "af", "pd", "draft"):
        frontier.extend(pareto([r for r in rows if r["mode"] == mode]))
    identities = {(r["case"], r["batch"]) for r in frontier}
    write_json(
        args.output / "formal-selection.json",
        [dict(case=c, users=n) for c, n in sorted(identities)],
    )
    missing = {name for name, _ in identities} - args.isolated_latencies.keys()
    if missing:
        cli.error(
            f"formal phase requires coarse single-client latencies: {sorted(missing)}"
        )
    chosen = [
        (case, clients)
        for case in cases
        for clients in args.users
        if (case.name, clients) in identities
    ]
    run_grid(args, workload, cases, chosen, phase="formal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
