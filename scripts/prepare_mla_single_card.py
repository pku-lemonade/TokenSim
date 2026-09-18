"""Prepare reproducible CPU shape discovery for the two-card MLA study."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_deepseek_mla import ROOT, ShapeDiscovery, configuration, simulate, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="pkua100-a100-pcie")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hw = json.loads((ROOT / "data/operator_data/deepseek-v3-theoretical.json").read_text())
    hw.update(device=args.device, memory_capacity_bytes=81920*1024**2,
              status="Target GPU inventory verified; compute/bandwidth and link remain assumptions before calibration")
    discovery = ShapeDiscovery(hw)
    experiments = []
    for context in (4096, 16384):
        for af in (1, 2, 4, 8, 16, 32):
            experiments.append(dict(kind="pipeline", batch=32, context=context, af_batch_size=af))
        for batch in (1, 2, 4, 8, 16, 32, 64, 128):
            experiments.append(dict(kind="concurrency", batch=batch, context=context, af_batch_size=8))
    for item in experiments:
        config = configuration(af_batch_size=item["af_batch_size"], assumptions=hw, router_location="f")
        simulate(discovery, config, item["batch"], item["context"])
    # Full local batches cover calibration beyond the microbatch=8 curve.
    for batch in (1, 2, 4, 8, 16, 32, 64, 128):
        config = configuration(af_batch_size=batch, assumptions=hw, router_location="f")
        simulate(discovery, config, batch, 4096)
    core = next(deepcopy(s) for s in discovery.shapes.values() if s["table"] == "mla_attention")
    cases = [dict(s, split="runtime") for s in discovery.shapes.values() if s["table"] != "communication"]
    for batch in (1, 4, 16, 32):
        for context in (1024, 4096, 16384, 65536):
            key = dict(core["key"], batch_size=batch, context_lengths=",".join([str(context)]*batch))
            cases.append(dict(table="mla_attention", key=key, split="runtime"))
    for batch in (2, 8):
        for context in (8192, 32768):
            key = dict(core["key"], batch_size=batch, context_lengths=",".join([str(context)]*batch))
            cases.append(dict(table="mla_attention", key=key, split="holdout"))
    unique = {(s["table"], json.dumps(s["key"], sort_keys=True)): s for s in cases}
    cases = list(unique.values())
    # Probe each kernel family before spending minutes on its full sweep.
    priority = {"mla_attention": 0, "batched_gemm": 1, "gemm": 2, "elementwise": 3, "moe": 4, "moe_group": 5}
    cases.sort(key=lambda s: (priority[s["table"]], json.dumps(s["key"], sort_keys=True)))
    for index, case in enumerate(cases):
        case["case_id"] = f"case{index:04d}"
    write_json(args.output / "hardware-assumptions.json", hw)
    write_json(args.output / "cases.json", cases)
    write_json(args.output / "runtime-shapes.json", list(discovery.shapes.values()))
    write_json(args.output / "experiments.json", experiments)
    write_json(args.output / "model.json", json.loads((ROOT / "data/models/deepseek-v3-671b-mla.json").read_text()))
    print(json.dumps(dict(cases=len(cases), tables=dict(Counter(s["table"] for s in cases)),
                          experiments=len(experiments))))


if __name__ == "__main__":
    main()
