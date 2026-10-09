#!/usr/bin/env python3
"""Run an explicit AgentX preset through the ordinary simulator entry point."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import build_parser, main


def prepare(case, backend=None, seed=0, result=None, matrix=None):
    matrix = matrix or yaml.safe_load(
        (ROOT / "data/experiments/agentx.yaml").read_text()
    )
    if case not in matrix["cases"]:
        raise ValueError(f'unknown case {case!r}; choices: {list(matrix["cases"])}')
    point = {**matrix["defaults"], **matrix["cases"][case]}
    selected = backend or point.pop("cache_backend", "simple_cpu")
    args = build_parser().parse_args([])
    for key, value in point.items():
        if key == "cache_backend":
            continue
        if not hasattr(args, key):
            raise ValueError(f"unsupported preset field: {key}")
        setattr(args, key, value)
    for key in ("model_config_path", "cluster", "dataset_path", "data_root"):
        value = getattr(args, key)
        if value is not None and not Path(value).is_absolute():
            setattr(args, key, str(ROOT / value))
    args.random_seed = seed
    args.results_path = str(
        result or ROOT / "results" / case / f"seed{seed}" / selected
    )
    directory = Path(args.results_path)
    directory.mkdir(parents=True, exist_ok=True)
    if selected == "hbm":
        connector = {"kv_connector": "NoopConnector", "kv_role": "kv_both"}
    elif selected == "simple_cpu":
        connector = {
            "kv_connector": "SimpleCPUOffloadConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use_per_rank": 356125000000,
                "host_available_bytes": 3246161657856,
                "lazy_offload": False,
                "transfer_utilization": 0.75,
            },
        }
    else:
        connector = json.loads(
            (ROOT / "data/kv_transfer/mooncake_store_dram_agentx.json").read_text()
        )
    path = directory / "kv_transfer_config.json"
    path.write_text(json.dumps(connector, indent=2) + "\n")
    args.kv_transfer_config = str(path)
    (directory / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    return args


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--backend", choices=("simple_cpu", "mooncake", "hbm"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_path", type=Path)
    options = parser.parse_args(argv)
    args = prepare(options.case, options.backend, options.seed, options.results_path)
    main(args)


if __name__ == "__main__":
    cli()
