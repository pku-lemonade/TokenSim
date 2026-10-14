"""Operator datasets, provenance and table validation."""

from .metadata import (
    META_FILENAME,
    OperatorDataValidationError,
    SourceRecord,
    PackageMeta,
)
from .data import TableIndex, OperatorDataPackage
from .tables import validate_rows, _read_table, _write_table
