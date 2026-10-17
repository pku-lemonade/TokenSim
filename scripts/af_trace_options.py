"""Shared AF trace arguments, independent of model and hardware settings."""

import argparse

from TokenSim.af.trace.config import AFTraceConfig, EVENT_TYPES


def add_trace_arguments(parser):
    group = parser.add_argument_group("trace collection and export")
    group.add_argument(
        "--trace",
        choices=("full", "validate", "off"),
        default="full",
        help="full: retain and export; validate: check then discard; off: skip events",
    )
    group.add_argument(
        "--trace-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="check complete event invariants before archival or discard",
    )
    group.add_argument(
        "--trace-detail",
        choices=("full", "timing"),
        default="full",
        help="export all event fields or only identifiers and timing fields",
    )
    group.add_argument(
        "--trace-event-types",
        nargs="+",
        choices=EVENT_TYPES,
        help="export only these event types; collection and validation remain complete",
    )
    group.add_argument(
        "--trace-iterations",
        type=int,
        nargs="+",
        help="export only these zero-based iteration IDs",
    )
    group.add_argument(
        "--trace-layers",
        type=int,
        nargs="+",
        help="export only these zero-based layer IDs; -1 selects model boundaries",
    )


def trace_configuration(args):
    return AFTraceConfig(
        mode=args.trace,
        validate=args.trace_validation,
        event_types=args.trace_event_types,
        iterations=args.trace_iterations,
        layers=args.trace_layers,
        detail=args.trace_detail,
    )
