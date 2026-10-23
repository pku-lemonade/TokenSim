"""Transfer observed ownership only across identical per-stream kernel sequences."""

from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from TokenSim.errors import ConfigurationError
from TokenSim.profiler.attribution import _gpu, _observed, _rank, activity_details
from TokenSim.profiler.io import _read, _complete
from TokenSim.profiler.records import union_duration


def kernel_correspondence(structure, timing):
    """Require exact inventories and stream order, without using durations."""

    def streams(rows):
        result = defaultdict(list)
        for row in rows:
            if _gpu(row):
                args = row.get("args", {})
                result[str(args.get("device")), str(args.get("stream"))].append(row)
        for items in result.values():
            items.sort(key=lambda row: (row["ts"], row["_index"]))
        return result

    left, right = streams(structure), streams(timing)
    if left.keys() != right.keys():
        raise ConfigurationError(
            "paired capture has different device/stream inventories"
        )
    result = {}
    for key, expected in left.items():
        observed = right[key]
        if len(expected) != len(observed):
            raise ConfigurationError(
                f"paired capture activity count differs on stream {key}"
            )
        for ordinal, (a, b) in enumerate(zip(expected, observed)):
            if (a.get("cat"), a.get("name")) != (b.get("cat"), b.get("name")):
                raise ConfigurationError(
                    f"paired capture kernel identity differs on stream {key} at {ordinal}"
                )
            for dimension in ("grid", "block", "bytes"):
                a_value = a.get("args", {}).get(dimension)
                b_value = b.get("args", {}).get(dimension)
                if a_value is not None and b_value is not None and a_value != b_value:
                    raise ConfigurationError(
                        f"paired capture activity {dimension} differs on stream {key} at {ordinal}"
                    )
            result[a["_index"]] = b
    return result


def paired_observed(path, rows, digest, manifest, aliases, file_index, *, root=None):
    bindings = manifest.get("paired_captures", [])
    if not isinstance(bindings, list):
        raise ConfigurationError("paired_captures must be a list")
    root = Path(path).resolve().parent.parent if root is None else root
    if not all(isinstance(entry, dict) for entry in bindings):
        raise ConfigurationError("paired capture bindings must be objects")
    candidates = [
        entry
        for entry in bindings
        if isinstance(entry.get("timing_trace"), str)
        and (root / entry["timing_trace"]).resolve() == Path(path).resolve()
    ]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ConfigurationError(
            "paired capture timing input has multiple structure bindings"
        )
    binding = candidates[0]
    if binding.get("timing_sha256") != digest:
        raise ConfigurationError("paired capture timing SHA-256 differs")
    relative = binding.get("structure_trace")
    if not isinstance(relative, str):
        raise ConfigurationError("paired capture requires a structure_trace path")
    structure_path = (root / relative).resolve()
    if not structure_path.is_relative_to(root):
        raise ConfigurationError(
            "paired capture structure path escapes collection directory"
        )
    raw, metadata, structure_digest = _read(structure_path)
    if structure_digest != binding.get("structure_sha256"):
        raise ConfigurationError("paired capture structure SHA-256 differs")
    structure, _ = _complete(raw)
    if manifest.get("measurement_protocol") == "paired_structure_gpu_timing_v1":
        forwards = sum(
            row.get("name") == "tokensim.forward"
            and not str(row.get("cat", "")).startswith("gpu_")
            for row in structure
        )
        if forwards != manifest.get("output_tokens"):
            raise ConfigurationError(
                "paired capture violates the fixed-batch forward contract"
            )
    correspondence = kernel_correspondence(structure, rows)
    if _rank(structure_path, structure) != _rank(path, rows):
        raise ConfigurationError("paired capture rank differs")
    template = _observed(structure_path, structure, aliases, file_index, manifest)
    result = []
    for span in template:
        ids = span.members or (span.id,)
        members = [correspondence[int(identity.rsplit(":", 1)[1])] for identity in ids]
        details = dict(
            span.details,
            **activity_details(members),
            attribution=(
                "paired_kernel_sequence"
                if span.forward is not None
                else span.details["attribution"]
            ),
            structure_source=str(structure_path),
            structure_sha256=structure_digest,
        )
        intervals = details["service_intervals"]
        result.append(
            replace(
                span,
                id=f"paired:{file_index}:{span.id}",
                source=str(path),
                start_us=min(start for start, _ in intervals),
                end_us=max(end for _, end in intervals),
                members=tuple(
                    f"observed:{file_index}:{row['_index']}" for row in members
                ),
                kernel_sum_us=sum(row["dur"] for row in members),
                busy_us=union_duration(intervals),
                details=details,
            )
        )
    return result
