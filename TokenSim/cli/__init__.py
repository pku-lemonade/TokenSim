"""Public command-line entry point for TokenSim."""

from __future__ import annotations

import sys
from TokenSim.errors import TokenSimError
from .arguments import build_parser, DATA_ROOT, LATENCY_BACKEND_KEYWORDS
from .configuration import (
    simulation_config,
    max_num_batched_tokens,
    build_parallel_config,
    validate_moe_parallel_config,
)
from .run import main, ROOT
from TokenSim.reporting import write_failure


def cli(argv=None):
    args = build_parser().parse_args(argv)
    try:
        main(args)
    except TokenSimError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0
