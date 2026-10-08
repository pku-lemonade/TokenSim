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
BATCH_ROOT = ROOT / "results/agentx-dsv4-full/native-mtp-execution/20261008-effective-K"


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
    if not path.exists() or not path.stat().st_size:
        return {"simulation_time": None, "phase": "initializing", "phase_counts": {}}
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
    return {"simulation_time": None, "phase": "writing first observation", "phase_counts": {}}


def validate(hardware, modes, destination):
    subprocess.run(
        [sys.executable, "scripts/validate_agentx_mtp_execution.py", str(BATCH_ROOT),
         "--hardware", *hardware, "--mode", *modes, "--output", str(EVIDENCE / destination)],
        cwd=ROOT, check=True,
    )


done = set()
missing_since = {}
while len(done) < len(RECORDS):
    for source in SCOPE["runtime_sources"] + SCOPE["extra_execution_inputs"]:
        assert hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() == source["sha256"], source["path"]
    observations = {}
    for record in RECORDS:
        sku = record["hardware"]
        key = record["key"]
        root = Path(record["result_root"]) / sku
        state_path = root / "run_status.json"
        if state_path.exists():
            terminal = json.loads(state_path.read_text())
            assert terminal["exit_code"] == 0, (sku, terminal)
            if key not in done:
                accepted_path = EVIDENCE / f"{key}-validation.json"
                if accepted_path.exists():
                    accepted = json.loads(accepted_path.read_text())["checkpoints"][key]
                    result = root / f"agentx_c{196 if sku == 'b200' else 384}.json"
                    assert hashlib.sha256(result.read_bytes()).hexdigest() == accepted["result_sha256"]
                else:
                    validate([sku], [record["mode"]], f"{key}-validation.json")
                done.add(key)
            observations[key] = {"status": "complete", "terminal": terminal}
            continue
        state = process_state(record["benchmark_pid"], record["benchmark_command"])
        launcher = process_state(record["launcher_pid"], record["command"])
        if state in ("missing", "zombie"):
            missing_since.setdefault(key, time.monotonic())
            assert launcher not in ("missing", "zombie") and time.monotonic() - missing_since[key] < 120, (sku, state, launcher)
        else:
            missing_since.pop(key, None)
        observations[key] = {"status": "running", "benchmark_pid": record["benchmark_pid"],
                             "process_state": state, "launcher_state": launcher, **last_window(root)}
    snapshot = {"observed_unix_s": time.time(), "checkpoints": observations,
                "runtime_hashes_verified": True, "rerun_performed": False}
    temporary = EVIDENCE / "monitor-state.json.tmp"
    temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
    temporary.replace(EVIDENCE / "monitor-state.json")
    print(json.dumps(snapshot), flush=True)
    if len(done) < len(RECORDS):
        time.sleep(45)
validate(["b200", "b300"], ["main", "disabled"], "pipeline-validation.json")
print("All four original K3/K0 full windows passed strict combined validation.", flush=True)
