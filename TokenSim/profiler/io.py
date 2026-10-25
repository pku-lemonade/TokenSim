"""Validated Chrome event input and collection manifests."""

from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

from TokenSim.errors import ConfigurationError


def _read(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, ValueError) as error:
        raise ConfigurationError(
            f"cannot read profiler trace {path}: {error}"
        ) from error
    rows = document.get("traceEvents") if isinstance(document, dict) else document
    if not isinstance(rows, list):
        raise ConfigurationError("trace must contain an event array")
    metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
    if not isinstance(metadata, dict):
        raise ConfigurationError("trace metadata must be an object")
    if isinstance(document, dict) and "baseTimeNanoseconds" in document:
        metadata = dict(metadata, baseTimeNanoseconds=document["baseTimeNanoseconds"])
    if isinstance(document, dict) and "host_name" in document:
        metadata = dict(metadata, host_name=document["host_name"])
    return rows, metadata, digest.hexdigest()


def _complete(rows):
    stacks = defaultdict(list)
    result = []
    ignored = defaultdict(int)
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ConfigurationError("trace events must be objects")
        row = dict(raw, _index=index)
        phase = row.get("ph")
        thread = str(row.get("pid")), str(row.get("tid"))
        if phase == "B":
            if type(row.get("ts")) not in (int, float) or not math.isfinite(row["ts"]):
                raise ConfigurationError("trace begin timestamp must be finite")
            stacks[thread].append(row)
            continue
        if phase == "E":
            if not stacks[thread]:
                raise ConfigurationError("trace end event has no begin event")
            if type(row.get("ts")) not in (int, float) or not math.isfinite(row["ts"]):
                raise ConfigurationError("trace end timestamp must be finite")
            start = stacks[thread].pop()
            row = dict(start, ph="X", dur=row["ts"] - start["ts"])
        elif phase != "X":
            ignored[str(phase)] += 1
            continue
        if (
            any(
                type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
                for key in ("ts", "dur")
            )
            or row["dur"] < 0
        ):
            raise ConfigurationError(
                "trace timestamps and durations must be finite and nonnegative"
            )
        if not isinstance(row.get("args", {}), dict):
            raise ConfigurationError("trace event args must be an object")
        result.append(row)
    if any(stacks.values()):
        raise ConfigurationError("trace contains unclosed begin events")
    return result, dict(ignored)


def _manifest(path, *, exact=False):
    if exact and not Path(path).is_file():
        raise ConfigurationError(f"collection manifest not found: {path}")
    candidates = (
        (Path(path),)
        if exact
        else tuple(
            folder / "manifest.json"
            for folder in (Path(path).parent, Path(path).parent.parent)
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            try:
                manifest = json.loads(candidate.read_text())
                if not isinstance(manifest, dict) or not isinstance(
                    manifest.get("model_config", {}), dict
                ):
                    raise ConfigurationError(
                        "collection manifest and model_config must be objects"
                    )
                for key in (
                    "batch_size",
                    "input_tokens",
                    "output_tokens",
                    "tensor_parallel_size",
                ):
                    if key in manifest and (
                        type(manifest[key]) is not int or manifest[key] < 1
                    ):
                        raise ConfigurationError(
                            f"collection {key} must be a positive integer"
                        )
                for key in ("paired_captures", "worker_identity"):
                    if key in manifest and (
                        not isinstance(manifest[key], list)
                        or not all(isinstance(item, dict) for item in manifest[key])
                    ):
                        raise ConfigurationError(
                            f"collection {key} must be a list of objects"
                        )
                return manifest
            except (OSError, ValueError) as error:
                raise ConfigurationError(
                    f"invalid collection manifest {candidate}: {error}"
                ) from error
    return {}
