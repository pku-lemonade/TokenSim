"""Operator latency tables: offline generation, validation, and runtime lookup.

Layout of a data package (``data/operator_data/<device>/<backend>/<version>/``)::

    generation_meta.yaml           dataset identity, sources, calibration, row counts
    gemm_perf.parquet              runtime tables: query key + latency_us + source_id
    context_attention_perf.parquet
    generation_attention_perf.parquet
    moe_perf.parquet
    elementwise_perf.parquet
    collective_perf.parquet
    *_analysis.parquet             optional: FLOPs, bytes, roofline terms, bottleneck

The runtime never recomputes rooflines for rows that exist in a table; the
analytical model is only consulted, and reported as such, when a table has no
usable data for a shape.
"""

from TokenSim.operator_data.lookup import (
    LookupResult,
    MissingOperatorDataError,
    OperatorLookup,
)
from TokenSim.operator_data.package import OperatorDataPackage, PackageMeta
from TokenSim.operator_data.schema import TABLE_SPECS, TableSpec

__all__ = [
    "LookupResult",
    "MissingOperatorDataError",
    "OperatorDataPackage",
    "OperatorLookup",
    "PackageMeta",
    "TABLE_SPECS",
    "TableSpec",
]
