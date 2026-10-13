"""Load, index and export immutable operator datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Mapping
from TokenSim.utils import BoundedCache
from TokenSim.hardware._yaml import load_yaml_mapping
from TokenSim.hardware.device import normalize_dtype
from TokenSim.operator_data.schema import INTEGER_FIELDS, TABLE_SPECS, TableSpec
from .metadata import META_FILENAME, PackageMeta, OperatorDataValidationError
from .tables import validate_rows, _read_table, _write_table


@dataclass(frozen=True)
class TableIndex:
    """Rows of one table indexed for lookup.

    ``exact`` maps the full key tuple to its row; ``grids`` maps the discrete
    key to a nested dict walking the table's axes in declared order, so that
    interpolation resolves the outermost axis first.
    """

    exact: Mapping[tuple[Any, ...], Mapping[str, Any]]
    grids: Mapping[tuple[Any, ...], Any]

    @classmethod
    def build(cls, spec: TableSpec, rows: Iterable[Mapping[str, Any]]) -> "TableIndex":
        exact: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        grids: dict[tuple[Any, ...], Any] = {}
        for row in rows:
            exact[tuple(row[f] for f in spec.key_fields)] = row
            discrete = tuple(row[f] for f in spec.discrete_fields)
            if not spec.axes:
                grids[discrete] = row
                continue
            node = grids.setdefault(discrete, {})
            for axis in spec.axes[:-1]:
                node = node.setdefault(row[axis.name], {})
            node[row[spec.axes[-1].name]] = row
        return cls(exact=exact, grids=grids)


@dataclass(frozen=True)
class OperatorDataPackage:
    root: Path | None
    meta: PackageMeta
    tables: Mapping[str, tuple[Mapping[str, Any], ...]]
    analysis: Mapping[str, tuple[Mapping[str, Any], ...]] = field(default_factory=dict)

    @cached_property
    def lookup_index(self) -> Mapping[str, TableIndex]:
        """Per-table lookup index, built once and shared by every ``OperatorLookup``.

        Indexing a few hundred thousand rows takes seconds; a 64-rank run
        creates two lookups per worker, so the index must not be rebuilt per
        consumer. The package is immutable, so sharing is safe.
        """
        return {
            name: TableIndex.build(TABLE_SPECS[name], rows)
            for name, rows in self.tables.items()
        }

    @property
    def dataset_version(self) -> str:
        return self.meta.dataset_version

    @property
    def device_id(self) -> str:
        return self.meta.device_id

    @property
    def backend(self) -> str:
        return self.meta.backend

    @cached_property
    def _source_descriptions(self):
        return BoundedCache(256)

    def describe_sources(self, table: str, key: Mapping[str, Any] | None = None):
        spec = TABLE_SPECS[table]
        selection = {
            name: (
                normalize_dtype(str(value))
                if name in spec.dtype_fields
                else value if name in INTEGER_FIELDS else str(value).strip().lower()
            )
            for name, value in (key or {}).items()
        }
        identity = table, tuple(sorted(selection.items()))
        if identity in self._source_descriptions:
            return self._source_descriptions[identity]
        source_ids = {
            row["source_id"]
            for row in self.tables.get(table, ())
            if all(row.get(name) == value for name, value in selection.items())
        }
        records = [
            (
                self.meta.sources[source_id].to_dict()
                if source_id in self.meta.sources
                else {"source_id": source_id, "grade": "unknown"}
            )
            for source_id in sorted(source_ids)
        ]
        self._source_descriptions[identity] = records
        return records

    # -- loading -----------------------------------------------------------

    @classmethod
    def load(
        cls, root: str | Path, *, strict_sources: bool = True
    ) -> "OperatorDataPackage":
        root_path = Path(root)
        if not root_path.is_dir():
            raise OperatorDataValidationError(
                f"operator-data package directory not found: {root_path}"
            )
        meta_path = root_path / META_FILENAME
        if not meta_path.is_file():
            raise OperatorDataValidationError(f"missing {META_FILENAME} in {root_path}")
        meta = PackageMeta.from_mapping(load_yaml_mapping(meta_path), meta_path)
        tables: dict[str, tuple[Mapping[str, Any], ...]] = {}
        analysis: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for table_name, spec in TABLE_SPECS.items():
            rows = _read_table(root_path, spec.perf_filename)
            if rows is None:
                continue
            checked = validate_rows(
                table_name,
                rows,
                meta,
                strict_sources=strict_sources,
                path=root_path / spec.perf_filename,
            )
            tables[table_name] = tuple(checked)
            analysis_rows = _read_table(root_path, spec.analysis_filename)
            if analysis_rows is not None:
                analysis[table_name] = tuple(analysis_rows)
        if not tables:
            raise OperatorDataValidationError(
                f"{root_path}: package has no *_perf tables"
            )
        for name, expected in meta.table_rows.items():
            actual = len(tables.get(name, ()))
            if actual != expected or name not in tables:
                raise OperatorDataValidationError(
                    f"{root_path}: {name} manifest declares {expected} rows but table has {actual if name in tables else 'no file'}"
                )
        undeclared = set(tables) - set(meta.table_rows)
        if undeclared:
            raise OperatorDataValidationError(
                f"{root_path}: undeclared tables: {sorted(undeclared)}"
            )
        return cls(root=root_path, meta=meta, tables=tables, analysis=analysis)

    @classmethod
    def from_rows(
        cls,
        meta: PackageMeta,
        tables: Mapping[str, Iterable[Mapping[str, Any]]],
        analysis: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
        *,
        strict_sources: bool = True,
    ) -> "OperatorDataPackage":
        checked = {
            name: tuple(
                validate_rows(name, list(rows), meta, strict_sources=strict_sources)
            )
            for name, rows in tables.items()
        }
        return cls(
            root=None,
            meta=meta,
            tables=checked,
            analysis={name: tuple(rows) for name, rows in (analysis or {}).items()},
        )

    # -- writing -----------------------------------------------------------

    def write(self, root: str | Path) -> Path:
        """Replace known package tables; merge explicitly before updating a subset."""
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        for name, spec in TABLE_SPECS.items():
            if name not in self.tables:
                (root_path / spec.perf_filename).unlink(missing_ok=True)
            if not self.analysis.get(name):
                (root_path / spec.analysis_filename).unlink(missing_ok=True)
        table_rows = {}
        for table_name, rows in self.tables.items():
            spec = TABLE_SPECS[table_name]
            _write_table(root_path / spec.perf_filename, rows, spec.required_fields)
            table_rows[table_name] = len(rows)
            analysis_rows = self.analysis.get(table_name)
            if analysis_rows:
                _write_table(root_path / spec.analysis_filename, analysis_rows, None)
        meta_dict = self.meta.to_dict()
        meta_dict["tables"] = {k: {"rows": v} for k, v in table_rows.items()}
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise OperatorDataValidationError(
                "writing packages requires PyYAML"
            ) from exc
        (root_path / META_FILENAME).write_text(
            yaml.safe_dump(meta_dict, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return root_path

    def summary(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "backend": self.backend,
            "dataset_version": self.dataset_version,
            "tables": {name: len(rows) for name, rows in self.tables.items()},
            "sources": sorted(self.meta.sources),
        }
