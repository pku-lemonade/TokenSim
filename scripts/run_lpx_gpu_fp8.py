"""Collect the separate T3 FP8-target, BF16-assistant Triton comparison.

Reuse the study runner's process ownership, exclusive lock, streaming client,
document gate and pricing. A failed startup or document control stops collection
and preserves the error; it is not automatically classified as unsupported FP8.
"""

import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_lpx_gpu_study as study

PROFILE = "triton-fp8"


def collect_fp8(work):
    study.validate_document_control("triton")
    study.start_server(work, 3, profile=PROFILE)
    study.collect(
        work,
        3,
        "aa-docs",
        limit=1,
        max_tokens=128,
        phase="warmup",
        warmup_tag=time.time_ns(),
        profile=PROFILE,
    )
    study.collect(work, 3, "aa-docs", limit=1, phase="smoke", profile=PROFILE)
    study.validate_document_control(PROFILE)
    study.collect(work, 3, "aa-docs", profile=PROFILE)
    ratio = study.measure_ratio(work, 3, profile=PROFILE)
    study.run_script(
        "analyze_lpx_gpu_acceptance.py",
        "--outputs",
        study.artifact("mtp-k3-outputs.jsonl", PROFILE),
        "--summary",
        study.artifact("mtp-k3-acceptance.json", PROFILE),
        "--ratio",
        study.pricing_ratio([ratio]),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/lpx-gpu"))
    args = parser.parse_args(argv)
    if args.work_dir.resolve().is_relative_to(ROOT):
        parser.error("work-dir must be outside the repository")
    with (args.work_dir / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(
            stage="t3",
            profile=PROFILE,
            pid=os.getpid(),
            status="running",
            started_unix=time.time(),
        )
        path = study.artifact("status.json", PROFILE)
        study.write_json(path, state)
        try:
            try:
                collect_fp8(args.work_dir)
            finally:
                study.stop_server(args.work_dir)
        except Exception as error:
            state.update(status="failed", error=str(error), finished_unix=time.time())
            study.write_json(path, state)
            raise
        state.update(status="complete", finished_unix=time.time())
        study.write_json(path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
