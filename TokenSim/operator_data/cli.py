"""Command line tools for operator data.

Examples::

    python -m TokenSim.operator_data.cli manifest --device groqchip_v1 --models llama-3-8b --tp 1,8         --out data/operator_data/groqchip_v1/analytical/shape_manifests/exp1.yaml
    python -m TokenSim.operator_data.cli generate --device groqchip_v1 --models llama-3-8b,mixtral-8x7b --tp 1,8,64
    python -m TokenSim.operator_data.cli import-aiconfigurator --system-dir <aic>/systems/data/a100_sxm --device a100_sxm_80g
    python -m TokenSim.operator_data.cli import-nccl --device a100_sxm_80g --file all_reduce_8.txt --operation all_reduce --group-size 8
    python -m TokenSim.operator_data.cli validate data/operator_data/a100_sxm_80g/trtllm
    python -m TokenSim.operator_data.cli coverage --device a100_sxm_80g --backend trtllm --models llama-3-70b --tp 4
    python -m TokenSim.operator_data.cli calibrate --device a100_sxm_80g --backend trtllm
"""

from __future__ import annotations

import sys
from TokenSim.errors import ConfigurationError
from .commands.arguments import build_parser
from .commands.operations import (
    _ints,
    _grid,
    _manifest_from_args,
    _print,
    cmd_manifest,
    cmd_generate,
    cmd_import_aiconfigurator,
    cmd_import_nccl,
    cmd_validate,
    cmd_coverage,
    cmd_calibrate,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
