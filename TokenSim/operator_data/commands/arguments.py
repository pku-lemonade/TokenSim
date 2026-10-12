"""Command-line arguments for operator dataset tools."""

from __future__ import annotations

import argparse
from TokenSim.hardware.context import DEFAULT_DATA_ROOT
from .operations import (
    cmd_manifest,
    cmd_generate,
    cmd_import_aiconfigurator,
    cmd_import_nccl,
    cmd_validate,
    cmd_coverage,
    cmd_calibrate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate, import, validate and calibrate operator latency tables"
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        dest="data_root",
        help="directory holding devices/, topologies/, models/ and operator_data/",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shape_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--models", help="comma-separated model ids from data/models")
        p.add_argument(
            "--manifest", help="existing shape manifest YAML (overrides --models)"
        )
        p.add_argument(
            "--experiment-id",
            dest="experiment_id",
            help="manifest identifier (default: the model ids)",
        )
        p.add_argument("--tp", help="comma-separated TP sizes (default 1)")
        p.add_argument("--ep", help="comma-separated EP sizes (default 1)")
        p.add_argument("--batch", help="comma-separated decode batch sizes")
        p.add_argument(
            "--prefill-tokens",
            dest="prefill_tokens",
            help="comma-separated prefill token counts",
        )
        p.add_argument("--context", help="comma-separated decode context lengths")
        p.add_argument(
            "--no-collectives",
            dest="no_collectives",
            action="store_true",
            help="omit communication shapes",
        )

    p = sub.add_parser(
        "manifest",
        help="write a shape manifest for models x parallel layouts x workload grid",
    )
    add_shape_args(p)
    p.add_argument(
        "--out",
        help="manifest YAML path (default: data/operator_data/manifests/<id>.yaml)",
    )
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("generate", help="generate an analytical package for a device")
    p.add_argument("--device", required=True, help="device id from data/devices")
    p.add_argument("--backend", default="analytical", help="package backend name")
    p.add_argument(
        "--calibration",
        help="YAML with per-table calibration and/or analytical parameter overrides",
    )
    p.add_argument(
        "--dataset-version", dest="dataset_version", help="package dataset label"
    )
    p.add_argument(
        "--out",
        help="package directory (default: data/operator_data/<device>/<backend>)",
    )
    p.add_argument(
        "--merge", action="store_true", help="merge into an existing package at --out"
    )
    p.add_argument(
        "--keep-existing",
        dest="keep_existing",
        action="store_true",
        help="on key clash keep existing rows",
    )
    add_shape_args(p)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser(
        "import-aiconfigurator", help="convert an AIConfigurator system directory"
    )
    p.add_argument(
        "--system-dir",
        dest="system_dir",
        required=True,
        help="AIConfigurator systems/data/<system> directory or a collector run",
    )
    p.add_argument("--device", required=True, help="TokenSim device id")
    p.add_argument(
        "--backend",
        default="trtllm",
        help="framework tables to import: trtllm, vllm or sglang",
    )
    p.add_argument(
        "--version", help="framework collection version (default: newest per table)"
    )
    p.add_argument(
        "--nccl-version", dest="nccl_version", help="NCCL collection version"
    )
    p.add_argument(
        "--dataset-version", dest="dataset_version", help="package dataset label"
    )
    p.add_argument(
        "--upstream-commit",
        dest="upstream_commit",
        default="main",
        help="AIConfigurator commit recorded in source references",
    )
    p.add_argument(
        "--no-custom-allreduce",
        dest="no_custom_allreduce",
        action="store_true",
        help="skip the framework's custom all-reduce measurements",
    )
    p.add_argument(
        "--gpus-per-node",
        dest="gpus_per_node",
        type=int,
        help="EP ranks per node for SGLang wide-EP DeepEP tables "
        "(default: num_gpus_per_node from the upstream system description)",
    )
    p.add_argument(
        "--out",
        help="package directory (default: data/operator_data/<device>/<backend>)",
    )
    p.add_argument(
        "--module-source",
        action="append",
        default=[],
        help="Explicit module source as table=system_directory; records cross-device provenance.",
    )
    p.set_defaults(func=cmd_import_aiconfigurator)

    p = sub.add_parser(
        "import-nccl", help="add nccl-tests output to a package's collective table"
    )
    p.add_argument("--device", required=True, help="TokenSim device id")
    p.add_argument("--backend", default="nccl", help="package backend name")
    p.add_argument("--file", required=True, help="nccl-tests output file")
    p.add_argument(
        "--operation",
        required=True,
        help="collective measured by the file",
        choices=[
            "all_reduce",
            "all_gather",
            "reduce_scatter",
            "all_to_all",
            "send_recv",
            "broadcast",
        ],
    )
    p.add_argument(
        "--group-size",
        dest="group_size",
        type=int,
        required=True,
        help="ranks in the measured communicator",
    )
    p.add_argument(
        "--nodes", type=int, default=1, help="nodes spanned by the communicator"
    )
    p.add_argument(
        "--source-id",
        dest="source_id",
        help="source id recorded with the rows (default derived from the file)",
    )
    p.add_argument("--reference", help="where the measurement can be found")
    p.add_argument("--notes", help="free-text measurement notes")
    p.add_argument(
        "--dataset-version", dest="dataset_version", help="package dataset label"
    )
    p.add_argument(
        "--out",
        help="package directory (default: data/operator_data/<device>/<backend>)",
    )
    p.set_defaults(func=cmd_import_nccl)

    p = sub.add_parser("validate", help="validate a package directory")
    p.add_argument("package", help="package directory")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser(
        "coverage",
        help="report exact/interpolated/extrapolated/missing per manifest query",
    )
    p.add_argument("--device", required=True, help="TokenSim device id")
    p.add_argument("--backend", help="package backend (default: preferred backend)")
    p.add_argument("--out", help="write summary and missing shapes to YAML")
    add_shape_args(p)
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser(
        "calibrate", help="fit analytical parameters to a measured package"
    )
    p.add_argument("--device", required=True, help="TokenSim device id")
    p.add_argument("--backend", help="package backend (default: preferred backend)")
    p.add_argument(
        "--holdout",
        type=float,
        default=0.2,
        help="fraction of rows held out to report fit error",
    )
    p.add_argument("--seed", type=int, default=0, help="holdout sampling seed")
    p.add_argument(
        "--max-rows",
        dest="max_rows",
        type=int,
        default=3000,
        help="rows sampled per table for fitting",
    )
    p.add_argument("--out", help="write the fitted calibration to YAML")
    p.set_defaults(func=cmd_calibrate)
    return parser
