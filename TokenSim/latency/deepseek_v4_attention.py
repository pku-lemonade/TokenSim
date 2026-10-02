"""Measured DSV4 CSA/HCA attention modules and bounded extrapolation."""

from collections import Counter
from dataclasses import dataclass
import math

from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.lookup import LookupPolicy, MissingOperatorDataError, OperatorLookup
from TokenSim.operator_data.schema import TABLE_SPECS


@dataclass(frozen=True)
class AttentionEstimate:
    latency_us: float
    match: str
    source_id: str


class DeepSeekV4AttentionEstimator:
    """Price one local layer's complete attention module per layer kind.

    The imported table includes the four projection paths, compressor/indexer,
    sparse attention and output projection.  The caller therefore must not add
    the generic DSV4 projection GEMMs a second time.
    """

    def __init__(self, backend):
        self.backend = backend
        self.model = backend.model
        self.parallel = backend.parallel_config
        if self.model.deepseek_v4 is None or not backend.package:
            raise ConfigurationError("native DSV4 attention requires a model and package")
        if self.parallel.pipeline_parallel_size != 1:
            raise ConfigurationError("native DSV4 attention currently requires PP1")
        if not backend.package.tables.get("dsv4_attention"):
            raise ConfigurationError("native DSV4 attention table is missing")
        self.tp = self.parallel.tensor_parallel_size
        self.context_latency_scale = float(
            getattr(backend, "native_attention_context_scale", 1.0)
        )
        if not math.isfinite(self.context_latency_scale) or self.context_latency_scale < 0:
            raise ConfigurationError(
                "native attention context scale must be finite and nonnegative"
            )
        self.lookup = OperatorLookup(
            backend.package,
            LookupPolicy(extrapolate="analytical", max_extrapolation_ratio=64.0),
            analytical_scaler=self._analytical_work,
        )
        self.cache = {}
        # Runtime lengths are often unique. Caching whole-table scans by
        # length merely repeats that scan at each new context position.
        spec = TABLE_SPECS["dsv4_attention"]
        self._discrete_fields = spec.discrete_fields
        self._axis_constraints = {
            axis: tuple(name for name in spec.axis_names if name != axis)
            for axis in spec.axis_names
        }
        index = {}
        for row in backend.package.tables["dsv4_attention"]:
            discrete = self._discrete_key(row)
            for axis, fields in self._axis_constraints.items():
                identity = (axis, discrete, tuple(row[name] for name in fields))
                index.setdefault(identity, set()).add(row[axis])
        self.axis_cache = {key: sorted(values) for key, values in index.items()}
        self.counts = Counter()
        # ``ModelSpec.layer_specs`` includes the reusable MTP layer after the
        # main stack; the complete attention tables here cover only the 61
        # main CSA/HCA layers.
        layers = self.model.deepseek_v4.layers(self.model.num_layers)[: self.model.num_layers]
        self.layer_counts = Counter(layer.attention_kind for layer in layers)
        if self.layer_counts["csa"] + self.layer_counts["hca"] != self.model.num_layers:
            raise ConfigurationError("native DSV4 attention expects CSA/HCA main layers")
        self.reset_stats()

    def reset_stats(self):
        self.counts.clear()
        self.seconds = 0.0

    def _analytical_work(self, _table, key):
        batch = max(1, float(key["batch_size"]))
        query = max(1, float(key["input_seq_len"]))
        context = max(0.0, float(key["context_len"]))
        ratio = float(key["compress_ratio"])
        if key["phase"] == "generation":
            if key["layer_kind"] == "csa":
                attended = min(1024.0, context / max(1.0, ratio)) + 128.0
                indexer = context / max(1.0, ratio)
            else:
                attended = context / max(1.0, ratio) + 128.0
                indexer = 0.0
            return batch * (1.0 + attended + indexer)
        # The module performs one query for every input token.  Prefix work is
        # compressed, while projection and SWA work remain linear in queries.
        prefix = context / max(1.0, ratio)
        indexer = prefix if key["layer_kind"] == "csa" else 0.0
        return batch * query * (1.0 + 128.0 + prefix + indexer)

    def _query(self, layer_kind, phase, batch_size, input_seq_len, context_len):
        key = {
            "layer_kind": layer_kind,
            "phase": phase,
            "mla_dtype": "bf16",
            "kv_cache_dtype": "fp8",
            "kernel_source": "FLASHMLA_SPARSE_DSV4",
            "kernel_dtype": "fp8_block",
            "model": "deepseek-ai/DeepSeek-V4-Pro",
            "architecture": "DeepseekV4ForCausalLM",
            "num_heads": self.model.num_attention_heads // self.tp,
            "tp_size": self.tp,
            "batch_size": max(1, int(batch_size)),
            "input_seq_len": max(1, int(input_seq_len)) if phase == "context" else 1,
            "context_len": max(0 if phase == "context" else 1, int(context_len)),
            "compress_ratio": 4 if layer_kind == "csa" else 128,
        }
        # The two borrowed SKUs have intentionally different sparse-table
        # lower bounds (for example B300 CSA context starts at q=4 and CSA
        # generation starts at batch=64).  Hold the first measured row for
        # such small out-of-grid requests; the runtime still records the
        # resulting clamp in the match counters and report.
        clamped = False
        if phase == "context" and key["context_len"] == 0:
            zero_prefix = self._axis_values(
                key, "input_seq_len", extra=("context_len", 0)
            )
            if zero_prefix and key["input_seq_len"] < min(zero_prefix):
                key["input_seq_len"] = min(zero_prefix)
                clamped = True
        for axis in ("batch_size", "input_seq_len", "context_len"):
            values = self._axis_values(key, axis)
            if values and key[axis] < values[0]:
                key[axis] = values[0]
                clamped = True
        identity = tuple(key.items())
        cached = self.cache.get(identity)
        if cached is not None:
            self.backend.stats.record("dsv4_attention", cached.match)
            self.counts[f"{phase}:{layer_kind}:{cached.match}"] = self.counts.get(
                f"{phase}:{layer_kind}:{cached.match}", 0
            ) + 1
            return cached
        try:
            row = self.lookup.lookup("dsv4_attention", key)
        except MissingOperatorDataError as exc:
            # This is a configuration/data error for native mode.  Falling
            # back to dense GQA would silently double count or erase DSV4
            # modules, so expose the missing discrete geometry instead.
            self.backend.stats.record_missing(exc)
            raise ConfigurationError(str(exc)) from exc
        result = AttentionEstimate(
            row.latency_us,
            "clamped" if clamped and row.match_type == "exact" else row.match_type,
            row.source_id,
        )
        if len(self.cache) > 65536:
            self.cache.clear()
        self.cache[identity] = result
        self.backend.stats.record("dsv4_attention", result.match)
        self.counts[f"{phase}:{layer_kind}:{result.match}"] += 1
        return result

    def _axis_values(self, key, axis, *, extra=None):
        """Axis values for the same discrete key and exact other-axis values."""
        extras = dict([extra]) if extra else {}
        identity = (
            axis,
            self._discrete_key(key),
            tuple(extras.get(name, key[name]) for name in self._axis_constraints[axis]),
        )
        return self.axis_cache.get(identity, [])

    def _discrete_key(self, key):
        return tuple(
            value.lower() if isinstance(value, str) else value
            for value in (key[name] for name in self._discrete_fields)
        )

    def estimate_prefill(self, batch_size, query_tokens, prefix_tokens):
        total = 0.0
        for kind in ("csa", "hca"):
            total += self.layer_counts[kind] * self._query(
                kind, "context", batch_size, query_tokens, prefix_tokens
            ).latency_us
        total *= self.context_latency_scale
        self.seconds += total * 1e-6
        return total

    def estimate_decode(self, batch_size, context_tokens):
        total = 0.0
        for kind in ("csa", "hca"):
            total += self.layer_counts[kind] * self._query(
                kind, "generation", batch_size, 1, context_tokens
            ).latency_us
        self.seconds += total * 1e-6
        return total

    def stats_dict(self):
        return {
            "native_dsv4_attention_seconds": self.seconds,
            "native_dsv4_attention_counts": dict(self.counts),
        }

    def describe(self):
        return {
            "backend": "FLASHMLA_SPARSE_DSV4 borrowed module table",
            "reference_backend": "FLASHINFER_MLA_SPARSE_DSV4",
            "layer_counts": dict(self.layer_counts),
            "module_scope": "projection + compressor/indexer + sparse attention + output projection",
            "kv_cache_source_layout": "fp8 source table; serving reference plain FP8",
            "indexer_source_layout": "collector default; serving reference MXFP4",
            "extrapolation": "measured boundary with DSV4 compressed-work analytical growth, max 64x",
            "context_latency_scale": self.context_latency_scale,
        }
