"""Shared AF trace arguments, independent of model and hardware settings."""

import argparse
from dataclasses import fields
import json
from pathlib import Path

from TokenSim.af.trace.config import AFTraceConfig, EVENT_FIELDS, EVENT_TYPES
from TokenSim.af.trace.formats import TRACE_FORMATS
from TokenSim.errors import ConfigurationError


def add_trace_arguments(parser):
    group = parser.add_argument_group("trace collection and export")
    group.add_argument(
        "--trace-format",
        choices=TRACE_FORMATS,
        default="json",
        help="saved trace representation; binary traces can be converted to JSON later",
    )
    group.add_argument(
        "--trace-config", type=Path, help="JSON trace configuration file"
    )
    group.add_argument(
        "--trace",
        choices=("full", "validate", "off"),
        default=None,
        help="full: retain and export; validate: check then discard; off: skip events",
    )
    group.add_argument(
        "--trace-validation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="check complete event invariants before archival or discard",
    )
    group.add_argument(
        "--trace-detail",
        choices=("full", "timing"),
        default=None,
        help="export all event fields or only identifiers and timing fields",
    )
    group.add_argument(
        "--trace-fields",
        nargs="+",
        choices=EVENT_FIELDS,
        help="export these event argument fields; Chrome timeline fields remain present",
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
    values = {}
    path = getattr(args, "trace_config", None)
    if path is not None:
        try:
            values = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigurationError(
                f"cannot read trace configuration {path}: {exc}"
            ) from exc
        AFTraceConfig.from_mapping(values)
    names = {"mode": "trace", "validate": "trace_validation"}
    for field in fields(AFTraceConfig):
        value = getattr(args, names.get(field.name, "trace_" + field.name), None)
        if value is not None:
            values[field.name] = value
    return AFTraceConfig.from_mapping(values)
