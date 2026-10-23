"""Pair collection artifacts and describe their measured timing boundary."""

import statistics
from pathlib import Path

from TokenSim.errors import ConfigurationError
from TokenSim.profiler.attribution import _rank
from TokenSim.profiler.io import _read, _complete
from TokenSim.profiler.paired import kernel_correspondence


def bind_captures(structure, timing, root, *, expected_forwards=None):
    def ranked(paths):
        result = {}
        for path in paths:
            raw, metadata, digest = _read(path)
            rows, _ = _complete(raw)
            rank = _rank(path, rows)
            if rank == "unknown" or rank in result:
                raise ConfigurationError(
                    "collection requires one identified trace per rank"
                )
            result[rank] = path, rows, digest, metadata
        return result

    left, right = ranked(structure), ranked(timing)
    if left.keys() != right.keys():
        raise ConfigurationError("structure and timing captures have different ranks")
    bindings = []
    for rank, (path, rows, digest, metadata) in right.items():
        template, expected, template_digest, _ = left[rank]
        if expected_forwards is not None:
            forwards = sum(
                row.get("name") == "tokensim.forward"
                and not str(row.get("cat", "")).startswith("gpu_")
                for row in expected
            )
            if forwards != expected_forwards:
                raise ConfigurationError(
                    "observed forward count differs from the fixed-batch collection contract"
                )
        kernel_correspondence(expected, rows)
        bindings.append(
            dict(
                rank=rank,
                timing_trace=str(Path(path).relative_to(root)),
                timing_sha256=digest,
                structure_trace=str(Path(template).relative_to(root)),
                structure_sha256=template_digest,
            )
        )
    return bindings


def timing_quality(baseline, profiled):
    reference = statistics.median(baseline)
    ratio = profiled / reference if reference else None
    return dict(
        baseline_median_seconds=reference,
        profile_generate_seconds=profiled,
        profile_to_baseline_ratio=ratio,
        status=(
            "within_10_percent"
            if ratio is not None and 0.90 <= ratio <= 1.10
            else "perturbed"
        ),
        scope="generate_wall_excluding_profiler_start_stop_and_export",
    )


def collection_inputs(manifest_path):
    from TokenSim.profiler.io import _manifest

    path = Path(manifest_path).resolve()
    manifest = _manifest(path, exact=True)
    if manifest.get("status") != "completed":
        raise ConfigurationError("collection must be completed before comparison")
    files = manifest.get("trace_files")
    if (
        not isinstance(files, list)
        or not files
        or not all(isinstance(item, str) for item in files)
    ):
        raise ConfigurationError("collection requires selected trace_files")
    inputs = [(path.parent / item).resolve() for item in files]
    if any(not item.is_relative_to(path.parent) for item in inputs) or len(
        set(inputs)
    ) != len(inputs):
        raise ConfigurationError(
            "collection trace paths must be distinct and inside the collection"
        )
    return inputs
