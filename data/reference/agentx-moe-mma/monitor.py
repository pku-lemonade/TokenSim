"""Observe the existing full windows and validate terminal results without reruns."""

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[2]
RECORDS = json.loads((EVIDENCE / "processes.json").read_text())
SCOPE = json.loads((EVIDENCE / "scope-audit.json").read_text())
BATCH_ROOT = ROOT / "results/agentx-dsv4-full/native-moe-mma/20261008-dynamic-tails-pipe"


def process_state(pid, expected):
    proc = Path("/proc") / str(pid)
    try:
        command = (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return "missing"
    if state == "Z":
        return "zombie"
    assert [part.decode() for part in command] == expected, f"PID {pid} identity changed"
    return state


def last_window(root):
    path = root / "native_replay/cache_windows.jsonl"
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 131072))
        lines = stream.read().splitlines()
    # A writer may be midway through the last line. Earlier complete records
    # still describe observed progress, and process identity is checked above.
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        return {key: row[key] for key in ("simulation_time", "phase", "phase_counts")}
    raise RuntimeError(f"No complete progress window in {path}")


def validate(hardware, destination):
    subprocess.run(
        [sys.executable, "scripts/validate_agentx_moe_mma.py", str(BATCH_ROOT),
         "--hardware", *hardware, "--output", str(EVIDENCE / destination)],
        cwd=ROOT, check=True,
    )


done = set()
missing_since = {}
while len(done) < len(RECORDS):
    for source in SCOPE["runtime_sources"]:
        assert hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() == source["sha256"], source["path"]
    observations = {}
    for record in RECORDS:
        sku = record["hardware"]
        root = Path(record["result_root"]) / sku
        state_path = root / "run_status.json"
        if state_path.exists():
            terminal = json.loads(state_path.read_text())
            assert terminal["exit_code"] == 0, (sku, terminal)
            if sku not in done:
                accepted_path = EVIDENCE / f"{sku}-validation.json"
                if accepted_path.exists():
                    accepted = json.loads(accepted_path.read_text())["checkpoints"][sku]
                    result = root / f"agentx_c{196 if sku == 'b200' else 384}.json"
                    assert hashlib.sha256(result.read_bytes()).hexdigest() == accepted["result_sha256"]
                else:
                    validate([sku], f"{sku}-validation.json")
                done.add(sku)
            observations[sku] = {"status": "complete", "terminal": terminal}
            continue
        state = process_state(record["benchmark_pid"], shlex.split(record["benchmark_command"]))
        launcher = process_state(record["launcher_pid"], record["command"])
        if state in ("missing", "zombie"):
            missing_since.setdefault(sku, time.monotonic())
            assert launcher not in ("missing", "zombie") and time.monotonic() - missing_since[sku] < 120, (sku, state, launcher)
        else:
            missing_since.pop(sku, None)
        observations[sku] = {"status": "running", "benchmark_pid": record["benchmark_pid"],
                             "process_state": state, "launcher_state": launcher, **last_window(root)}
    snapshot = {"observed_unix_s": time.time(), "checkpoints": observations,
                "runtime_hashes_verified": True, "rerun_performed": False}
    temporary = EVIDENCE / "monitor-state.json.tmp"
    temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
    temporary.replace(EVIDENCE / "monitor-state.json")
    print(json.dumps(snapshot), flush=True)
    if len(done) < len(RECORDS):
        time.sleep(45)
validate([record["hardware"] for record in RECORDS], "pipeline-validation.json")
print("Both original full windows passed strict combined validation.", flush=True)
