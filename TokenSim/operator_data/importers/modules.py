"""Import measured operator modules and register every row's provenance."""

from pathlib import Path

from TokenSim.operator_data.package import SourceRecord
from TokenSim.operator_data.schema import TABLE_SPECS
from . import dsv4_attention, dsv4_megamoe, mhc


# What each complete module times; recorded with every imported source.
SCOPES = {
    "mhc": (
        "mHC (hyper-connection residual mixing) pre/post module; each row "
        "covers the recorded sites of one layer."
    ),
    "dsv4_megamoe": (
        "Fused expert-parallel MoE module (DeepGEMM MegaMoE, FP8 activations "
        "and MXFP4 weights): routed dispatch, expert compute and combine under "
        "CUDA graphs. Gate/top-k, buffer initialization and shared experts are "
        "excluded. Collector buffer capacities are kept per row."
    ),
    "dsv4_attention": (
        "Complete compressed sparse-attention layer: input projections, "
        "compressor/indexer, sparse attention and output projection, measured "
        "with the recorded kernel and FP8 KV cache. Serving stacks with another "
        "sparse-attention kernel or cache layout use these rows as an "
        "approximation."
    ),
}


def import_modules(
    system_path,
    backend,
    version,
    commit,
    tables,
    sources,
    *,
    target_device,
    overrides=None,
):
    # Families and filenames belong to the upstream data format. Model and
    # geometry selectors stay in each row, independent of the target model.
    specs = [
        ("mhc", "mhc", "mhc_module_perf.parquet", backend, mhc.normalize, {}),
        (
            "dsv4_megamoe",
            "moe",
            "dsv4_megamoe_module_perf.parquet",
            "sglang",
            dsv4_megamoe.normalize,
            {},
        ),
    ]
    for kind in ("csa", "hca"):
        for phase in ("context", "generation"):
            specs.append(
                (
                    "dsv4_attention",
                    "sparse_attention",
                    f"dsv4_{kind}_{phase}_module_perf.parquet",
                    backend,
                    dsv4_attention.normalize,
                    {"layer_kind": kind, "phase": phase},
                )
            )
    from .aiconfigurator import _candidates, _read_parquet, _version_label

    overrides = overrides or {}
    known = {spec[0] for spec in specs}
    if set(overrides) - known:
        raise ValueError(
            f"unknown operator module sources: {sorted(set(overrides) - known)}"
        )
    for table, family, filename, source_backend, normalize, extra in specs:
        root = Path(overrides.get(table, system_path))
        requested = version if source_backend == backend else None
        selected = {}
        for directory in _candidates(
            root / family, source_backend, requested, (filename,)
        ):
            path = directory / filename
            if not path.is_file():
                continue
            raw = _read_parquet(path)
            label = _version_label(directory, raw, requested or "unknown")
            source_id = f"aiconfigurator:{root.name}:{table}:{source_backend}:{label}"
            kwargs = dict(source_id=source_id, source_hardware=root.name, **extra)
            kwargs[
                "collection_version" if table == "dsv4_megamoe" else "source_version"
            ] = label
            rows = normalize(raw, **kwargs)
            notes = SCOPES[table]
            if root.name != target_device or source_backend != backend:
                notes += (
                    f" Measured on {root.name} with {source_backend} and used for "
                    f"{target_device} with {backend} without a conversion factor."
                )
            sources[source_id] = SourceRecord(
                source_id=source_id,
                grade="B",
                method="imported",
                device=root.name,
                backend=f"{source_backend}:{label}",
                reference=f"https://github.com/ai-dynamo/aiconfigurator/tree/{commit}/aic-core/src/aiconfigurator_core/systems/data/{root.name}/{family}/{source_backend}/{label}",
                notes=notes,
            )
            for row in rows:
                identity = tuple(row[name] for name in TABLE_SPECS[table].key_fields)
                selected.setdefault(identity, row)
        if selected:
            existing = tables.setdefault(table, [])
            identities = {
                tuple(row[name] for name in TABLE_SPECS[table].key_fields)
                for row in existing
            }
            existing.extend(
                row for identity, row in selected.items() if identity not in identities
            )
