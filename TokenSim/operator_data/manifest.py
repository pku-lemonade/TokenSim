"""Shape manifests: the set of operator queries an experiment needs answered.

A manifest is generated from model specs, parallel layouts and workload grids
(batch sizes, sequence lengths, context lengths). The same manifest drives the
analytical generator, the coverage report against measured tables, and the
profiling scripts that collect ground truth on real hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from TokenSim.config.cache_config import local_kv_heads
from TokenSim.config.model_config import ModelSpec
from TokenSim.errors import ConfigurationError
from TokenSim.hardware._yaml import load_yaml_mapping
from TokenSim.operator_data.schema import TABLE_SPECS

DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_PREFILL_TOKENS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
DEFAULT_CONTEXT_LENS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
DEFAULT_MESSAGE_BYTES = tuple(1 << exp for exp in range(10, 31))  # 1 KiB .. 1 GiB
DEFAULT_GROUP_SIZES = (2, 4, 8)


@dataclass(frozen=True)
class WorkloadGrid:
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES
    prefill_tokens: tuple[int, ...] = DEFAULT_PREFILL_TOKENS
    context_lens: tuple[int, ...] = DEFAULT_CONTEXT_LENS
    tp_sizes: tuple[int, ...] = (1,)
    ep_sizes: tuple[int, ...] = (1,)
    message_bytes: tuple[int, ...] = DEFAULT_MESSAGE_BYTES
    group_sizes: tuple[int, ...] = DEFAULT_GROUP_SIZES
    collective_operations: tuple[str, ...] = ("all_reduce", "all_to_all")
    include_collectives: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WorkloadGrid":
        values: dict[str, Any] = {}
        for name in (
            "batch_sizes",
            "prefill_tokens",
            "context_lens",
            "tp_sizes",
            "ep_sizes",
            "message_bytes",
            "group_sizes",
        ):
            if name in raw:
                values[name] = tuple(int(v) for v in raw[name])
        if "collective_operations" in raw:
            values["collective_operations"] = tuple(str(v) for v in raw["collective_operations"])
        if "include_collectives" in raw:
            values["include_collectives"] = bool(raw["include_collectives"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_sizes": list(self.batch_sizes),
            "prefill_tokens": list(self.prefill_tokens),
            "context_lens": list(self.context_lens),
            "tp_sizes": list(self.tp_sizes),
            "ep_sizes": list(self.ep_sizes),
            "message_bytes": list(self.message_bytes),
            "group_sizes": list(self.group_sizes),
            "collective_operations": list(self.collective_operations),
            "include_collectives": self.include_collectives,
        }


@dataclass
class ShapeManifest:
    """Ordered, de-duplicated query keys per table plus their provenance."""

    experiment_id: str
    keys: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    roles: dict[str, dict[tuple, set[str]]] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)
    grid: WorkloadGrid | None = None
    notes: str = ""

    def add(self, table: str, key: Mapping[str, Any], role: str = "") -> None:
        if table not in TABLE_SPECS:
            raise ConfigurationError(f"unknown table {table!r}")
        spec = TABLE_SPECS[table]
        clean = {f: key[f] for f in spec.key_fields}
        identity = tuple(clean[f] for f in spec.key_fields)
        table_roles = self.roles.setdefault(table, {})
        if identity not in table_roles:
            self.keys.setdefault(table, []).append(clean)
            table_roles[identity] = set()
        if role:
            table_roles[identity].add(role)

    def count(self) -> dict[str, int]:
        return {table: len(keys) for table, keys in self.keys.items()}

    def total(self) -> int:
        return sum(self.count().values())

    def to_dict(self) -> dict[str, Any]:
        tables = {}
        for table, keys in self.keys.items():
            spec = TABLE_SPECS[table]
            rows = []
            for key in keys:
                identity = tuple(key[f] for f in spec.key_fields)
                rows.append({**key, "roles": sorted(self.roles[table][identity])})
            tables[table] = rows
        return {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "models": list(self.models),
            "grid": self.grid.to_dict() if self.grid else None,
            "notes": self.notes,
            "counts": self.count(),
            "tables": tables,
        }

    def write(self, path: str | Path) -> Path:
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ShapeManifest":
        document = load_yaml_mapping(path)
        manifest = cls(
            experiment_id=str(document.get("experiment_id", Path(path).stem)),
            models=[str(m) for m in document.get("models", [])],
            grid=WorkloadGrid.from_mapping(document["grid"]) if document.get("grid") else None,
            notes=str(document.get("notes", "")),
        )
        for table, rows in (document.get("tables") or {}).items():
            for row in rows:
                roles = row.get("roles") or [""]
                for role in roles:
                    manifest.add(table, row, role)
        return manifest


def build_manifest(
    experiment_id: str,
    models: Sequence[ModelSpec],
    grid: WorkloadGrid | None = None,
    *,
    dtypes: Iterable[str] | None = None,
) -> ShapeManifest:
    grid = grid or WorkloadGrid()
    manifest = ShapeManifest(experiment_id=experiment_id, grid=grid, models=[m.model_id for m in models])
    for model in models:
        model_dtypes = tuple(dtypes) if dtypes else (model.dtype,)
        for dtype in model_dtypes:
            for tp in grid.tp_sizes:
                _add_model_shapes(manifest, model, dtype, tp, grid)
    if grid.include_collectives:
        for operation in grid.collective_operations:
            for group in grid.group_sizes:
                for message in grid.message_bytes:
                    manifest.add(
                        "collective",
                        {
                            "dtype": models[0].dtype if models else "fp16",
                            "operation": operation,
                            "group_size": group,
                            "nodes": 1,
                            "message_bytes": message,
                        },
                        role="tp_collective" if operation == "all_reduce" else "ep_dispatch_combine",
                    )
    return manifest


def _add_model_shapes(manifest: ShapeManifest, model: ModelSpec, dtype: str, tp: int, grid: WorkloadGrid) -> None:
    h = model.hidden_size
    heads_local = math.ceil(model.num_attention_heads / tp)
    kv_heads_local = local_kv_heads(model.num_key_value_heads, tp)
    q_local = heads_local * model.head_dim
    kv_local = kv_heads_local * model.head_dim
    inter_local = math.ceil(model.intermediate_size / tp)
    vocab_local = math.ceil(model.vocab_size / tp)
    tag = f"{model.model_id}:tp{tp}"

    token_counts = sorted(set(grid.batch_sizes) | set(grid.prefill_tokens))
    for m in token_counts:
        manifest.add("gemm", {"dtype": dtype, "m": m, "n": q_local + 2 * kv_local, "k": h}, f"{tag}:qkv_proj")
        manifest.add("gemm", {"dtype": dtype, "m": m, "n": h, "k": q_local}, f"{tag}:o_proj")
        if not model.is_moe or model.dense_layer_count() > 0:
            manifest.add("gemm", {"dtype": dtype, "m": m, "n": model.ffn_projection_count * inter_local, "k": h}, f"{tag}:ffn_gate_up")
            manifest.add("gemm", {"dtype": dtype, "m": m, "n": h, "k": inter_local}, f"{tag}:ffn_down")
        if model.is_moe:
            manifest.add("gemm", {"dtype": dtype, "m": m, "n": model.moe.num_experts, "k": h}, f"{tag}:router")
        for op, hidden in (
            ("rmsnorm", h),
            ("rope", q_local + kv_local),
            ("residual_add", h),
        ):
            manifest.add("elementwise", {"op_name": op, "dtype": dtype, "num_tokens": m, "hidden_size": hidden}, f"{tag}:{op}")
        if model.gated:
            manifest.add("elementwise", {"op_name": "swiglu", "dtype": dtype, "num_tokens": m, "hidden_size": inter_local}, f"{tag}:swiglu")
    for b in grid.batch_sizes:
        manifest.add("gemm", {"dtype": dtype, "m": b, "n": vocab_local, "k": h}, f"{tag}:lm_head")

    attn_common = {
        "attn_dtype": dtype,
        "kv_cache_dtype": model.kv_cache_dtype,
        "num_heads": heads_local,
        "num_kv_heads": kv_heads_local,
        "head_dim": model.head_dim,
        "window_size": model.sliding_window,
    }
    for seq in grid.prefill_tokens:
        for batch in (1, 2, 4, 8):
            if batch * seq > max(grid.prefill_tokens) * 4:
                continue
            manifest.add("context_attention", {**attn_common, "batch_size": batch, "input_seq_len": seq}, f"{tag}:prefill_attention")
    for context in grid.context_lens:
        for batch in grid.batch_sizes:
            manifest.add("generation_attention", {**attn_common, "batch_size": batch, "context_len": context}, f"{tag}:decode_attention")

    if model.is_moe:
        for ep in grid.ep_sizes:
            moe_tp = 1 if ep > 1 else tp
            for tokens in token_counts:
                manifest.add(
                    "moe",
                    {
                        "dtype": dtype,
                        "distribution": "uniform",
                        "num_tokens": tokens,
                        "hidden_size": h,
                        "inter_size": model.moe_intermediate_size,
                        "top_k": model.moe.num_experts_per_tok,
                        "num_experts": model.moe.num_experts,
                        "tp_size": moe_tp,
                        "ep_size": ep,
                    },
                    f"{tag}:ep{ep}:moe_experts",
                )
