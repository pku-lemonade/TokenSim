"""Measured compressed attention CSA/HCA attention modules and bounded extrapolation."""

from collections import Counter
from dataclasses import dataclass
import math

from TokenSim.utils import BoundedCache
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.lookup import (
    LookupPolicy,
    MissingOperatorDataError,
    OperatorLookup,
)
from TokenSim.operator_data.schema import TABLE_SPECS
from TokenSim.operator_data.package import OperatorDataPackage
from TokenSim.latency.attention_work import CompressedAttentionWork


@dataclass(frozen=True)
class AttentionEstimate:
    latency_us: float
    match: str
    source_id: str
    ragged_grid: bool = False
    source_batch_size: int | None = None
    source_context_len: int | None = None
    source_query_tokens: int | None = None


class CompressedAttentionEstimator:
    """Price one local layer's complete attention module per layer kind.

    The imported table includes the four projection paths, compressor/indexer,
    sparse attention and output projection.  The caller therefore must not add
    the generic compressed attention projection GEMMs a second time.
    """

    def __init__(self, backend):
        self.backend = backend
        self.model = backend.model
        self.parallel = backend.parallel_config
        self.binding = self.model.operator_modules.get("attention")
        if not self.binding:
            raise ConfigurationError(
                "measured attention needs an operator_modules.attention binding"
            )
        self.table = self.binding["table"]
        if self.model.components is None or not backend.package:
            raise ConfigurationError(
                "native compressed attention requires a model and package"
            )
        if self.parallel.pipeline_parallel_size != 1:
            raise ConfigurationError(
                "native compressed attention currently requires PP1"
            )
        if not backend.package.tables.get(self.table):
            raise ConfigurationError("native compressed attention table is missing")
        self.tp = self.parallel.tensor_parallel_size
        self.sol = CompressedAttentionWork(self.model, backend.device)
        # Pinned DeepGEMM SM100 paged MQA uses BLOCK_Q = 128 / num_heads.
        # Indexer heads are replicated under TP. Pro has 64 -> two queries
        # per KV-reading Q-block; q=4 requires two such blocks.
        index_heads = self.model.components.index_n_heads
        if index_heads > 128 or 128 % index_heads:
            raise ConfigurationError(
                "native SM100 indexer Q-block needs heads dividing 128"
            )
        self.indexer_queries_per_tile = 128 // index_heads
        self.lookup = OperatorLookup(
            backend.package,
            LookupPolicy(extrapolate="analytical", max_extrapolation_ratio=64.0),
            analytical_scaler=self._analytical_work,
            bind_analytical_axes=True,
        )
        self.cache = BoundedCache(65536)
        self.request_cache = BoundedCache(65536)
        # Runtime lengths are often unique. Caching whole-table scans by
        # length merely repeats that scan at each new context position.
        spec = TABLE_SPECS[self.table]
        self._discrete_fields = spec.discrete_fields
        self._axis_constraints = {
            axis: tuple(name for name in spec.axis_names if name != axis)
            for axis in spec.axis_names
        }
        index = {}
        self.axis_lower_bounds = {}
        for row in backend.package.tables[self.table]:
            discrete = self._discrete_key(row)
            for axis in spec.axis_names:
                identity = (axis, discrete)
                self.axis_lower_bounds[identity] = min(
                    self.axis_lower_bounds.get(identity, row[axis]), row[axis]
                )
            for axis, fields in self._axis_constraints.items():
                identity = (axis, discrete, tuple(row[name] for name in fields))
                index.setdefault(identity, set()).add(row[axis])
        self.axis_cache = {key: sorted(values) for key, values in index.items()}
        self.generation_slices = {}
        self.context_query_floors = {}
        for row in backend.package.tables[self.table]:
            if row["phase"] == "generation":
                batches = self.generation_slices.setdefault(self._discrete_key(row), {})
                batches.setdefault(row["batch_size"], []).append(row)
            elif row["phase"] == "context":
                # Some samples use prefix = total_context - query. The only
                # query at an exact prefix is not the minimum legal query for
                # that batch. Retain genuine query floors across its prefixes.
                floors = self.context_query_floors.setdefault(
                    self._discrete_key(row), {}
                )
                batch = row["batch_size"]
                floors[batch] = min(
                    floors.get(batch, row["input_seq_len"]), row["input_seq_len"]
                )
        self.generation_slice_lookups = {}
        self.counts = Counter()
        # ``ModelSpec.layer_specs`` includes the reusable MTP layer after the
        # main stack; complete attention tables describe its CSA/HCA layers.
        layers = self.model.layer_specs(include_mtp=False)
        self.layer_ratios = {
            layer.attention_kind: layer.compress_ratio for layer in layers
        }
        self.layer_counts = Counter(layer.attention_kind for layer in layers)
        if self.layer_counts["csa"] + self.layer_counts["hca"] != self.model.num_layers:
            raise ConfigurationError(
                "native compressed attention expects CSA/HCA main layers"
            )
        self.reset_stats()

    def reset_stats(self):
        self.counts.clear()
        self.seconds = 0.0
        self.decode_batch_stats = Counter()
        self.mtp_kv_stats = Counter()
        self.composition_stats = Counter()

    def _analytical_work(self, _table, key):
        return self.sol.estimate_us(key)

    def _query(self, layer_kind, phase, batch_size, input_seq_len, context_len):
        # Front cache over the raw arguments; it is cleared together with the
        # resolved-key cache, so a hit returns exactly the object that cache
        # would return.
        request = (layer_kind, phase, batch_size, input_seq_len, context_len)
        cached = self.request_cache.get(request)
        if cached is None:
            cached = self._resolve_query(*request)
            self.request_cache[request] = cached
            return cached
        self.backend.stats.record(self.table, cached.match)
        self.counts[f"{phase}:{layer_kind}:{cached.match}"] += 1
        if cached.ragged_grid:
            self.decode_batch_stats["ragged_grid_lookup_queries"] += 1
        return cached

    def _resolve_query(self, layer_kind, phase, batch_size, input_seq_len, context_len):
        key = {
            "layer_kind": layer_kind,
            "phase": phase,
            **self.binding["key"],
            "num_heads": self.model.num_attention_heads // self.tp,
            "tp_size": self.tp,
            "batch_size": max(1, int(batch_size)),
            "input_seq_len": max(1, int(input_seq_len)) if phase == "context" else 1,
            "context_len": max(0 if phase == "context" else 1, int(context_len)),
            "compress_ratio": self.layer_ratios[layer_kind],
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
            if phase == "context" and axis == "input_seq_len":
                floors = self.context_query_floors.get(self._discrete_key(key), {})
                floor = floors.get(
                    key["batch_size"], min(floors.values()) if floors else None
                )
                if floor is not None and key[axis] < floor:
                    key[axis] = floor
                    clamped = True
                # Resolve the actual query's prefix along the existing 2D
                # grid. An exact-prefix query minimum would replace a valid
                # q512/prefix49152 lookup with q16384, for example.
                continue
            floor = self.axis_lower_bounds.get((axis, self._discrete_key(key)))
            if floor is not None and key[axis] < floor:
                key[axis] = floor
                clamped = True
        identity = (clamped, tuple(key.items()))
        cached = self.cache.get(identity)
        if cached is not None:
            self.backend.stats.record(self.table, cached.match)
            self.counts[f"{phase}:{layer_kind}:{cached.match}"] = (
                self.counts.get(f"{phase}:{layer_kind}:{cached.match}", 0) + 1
            )
            if cached.ragged_grid:
                self.decode_batch_stats["ragged_grid_lookup_queries"] += 1
            return cached
        ragged_grid = False
        try:
            row = self.lookup.lookup(self.table, key)
        except MissingOperatorDataError as exc:
            if phase == "generation" and exc.kind == "out_of_range":
                try:
                    row = self._query_supported_generation_slices(key)
                    ragged_grid = True
                except MissingOperatorDataError:
                    self.backend.stats.record_missing(exc)
                    raise ConfigurationError(str(exc)) from exc
            else:
                # Expose missing discrete geometry instead of silently
                # substituting a dense GQA module or another model.
                self.backend.stats.record_missing(exc)
                raise ConfigurationError(str(exc)) from exc
        result = AttentionEstimate(
            row.latency_us,
            "clamped" if clamped and row.match_type == "exact" else row.match_type,
            row.source_id,
            ragged_grid,
            key["batch_size"],
            key["context_len"],
            key["input_seq_len"],
        )
        self.cache[identity] = result
        self.backend.stats.record(self.table, result.match)
        self.counts[f"{phase}:{layer_kind}:{result.match}"] += 1
        if ragged_grid:
            self.decode_batch_stats["ragged_grid_lookup_queries"] += 1
        return result

    def _query_supported_generation_slices(self, key):
        """Keep only neighboring batch slices that support the context bound.

        The collector's context range shrinks at large batches. A batch=128
        neighbor measured only to context=6144 must not prevent using a
        batch=64 neighbor measured to 16384 for a long-context query. Retain
        the same discrete geometry and all samples in each supported slice;
        the ordinary lookup still applies its 64x bounds along both axes.
        No source rows or extrapolation limits are changed.
        """
        discrete = self._discrete_key(key)
        slices = self.generation_slices.get(discrete, {})
        context = key["context_len"]
        limit = self.lookup.policy.max_extrapolation_ratio
        supported = tuple(
            batch
            for batch, rows in sorted(slices.items())
            if context <= limit * max(row["context_len"] for row in rows)
            and context * limit >= min(row["context_len"] for row in rows)
        )
        if not supported:
            raise MissingOperatorDataError(
                self.table,
                key,
                "no batch slice supports the context bound",
                "out_of_range",
            )
        identity = (discrete, supported)
        lookup = self.generation_slice_lookups.get(identity)
        if lookup is None:
            package = OperatorDataPackage.from_rows(
                self.backend.package.meta,
                {self.table: [row for batch in supported for row in slices[batch]]},
            )
            lookup = self.generation_slice_lookups[identity] = OperatorLookup(
                package,
                self.lookup.policy,
                analytical_scaler=self._analytical_work,
                bind_analytical_axes=True,
            )
        return lookup.lookup(self.table, key)

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
            total += (
                self.layer_counts[kind]
                * self._query(
                    kind, "context", batch_size, query_tokens, prefix_tokens
                ).latency_us
            )
        self.seconds += total * 1e-6
        return total

    def _sol_key(self, kind, phase, batch, queries, context, estimate=None):
        return {
            "phase": phase,
            "batch_size": (estimate.source_batch_size or batch) if estimate else batch,
            "input_seq_len": (
                (estimate.source_query_tokens or queries) if estimate else queries
            ),
            "context_len": (
                (
                    estimate.source_context_len
                    if estimate.source_context_len is not None
                    else context
                )
                if estimate
                else context
            ),
            "compress_ratio": self.layer_ratios[kind],
            "num_heads": self.model.num_attention_heads // self.tp,
            "tp_size": self.tp,
        }

    def _decode_layer(
        self,
        kind,
        batch_size,
        context_tokens,
        *,
        indexer_kv_fraction=1.0,
        byte_weight=1.0,
    ):
        estimate = self._query(kind, "generation", batch_size, 1, context_tokens)
        latency = estimate.latency_us
        key = self._sol_key(kind, "generation", batch_size, 1, context_tokens, estimate)
        if kind == "csa" and indexer_kv_fraction < 1:
            # Honor source-grid clamps; reuse changes only request-local reads.
            math_us, memory_us, indexer_bytes = self.sol.limits_us(key)
            source = max(math_us, memory_us)
            target = max(
                math_us,
                memory_us
                - (1 - indexer_kv_fraction) * indexer_bytes / self.sol.bandwidth * 1e6,
            )
            corrected = latency * target / source
            layers = self.layer_counts[kind]
            self.mtp_kv_stats["source_geometry_indexer_bytes_before"] += (
                byte_weight * layers * indexer_bytes
            )
            self.mtp_kv_stats["source_geometry_indexer_bytes_after"] += (
                byte_weight * layers * indexer_bytes * indexer_kv_fraction
            )
            self.mtp_kv_stats["estimated_seconds_saved"] += (
                byte_weight * layers * (latency - corrected) * 1e-6
            )
            self.mtp_kv_stats["csa_adjustment_queries"] += 1
            latency = corrected
        return latency, key

    def _decode_us(
        self, batch_size, context_tokens, *, indexer_kv_fraction=1.0, byte_weight=1.0
    ):
        total = sum(
            self.layer_counts[kind]
            * self._decode_layer(
                kind,
                batch_size,
                context_tokens,
                indexer_kv_fraction=indexer_kv_fraction,
                byte_weight=byte_weight,
            )[0]
            for kind in ("csa", "hca")
        )
        return total

    def estimate_decode_batch(self, buckets, *, projection_rows=None):
        """Price one ragged decode batch, not a full module per length bucket.

        ``buckets`` maps (KV length, target query width) to request count.
        Module measurements include projections and their weight reads, which
        serving executes once over the full query tensor. For heterogeneous
        lengths use a query-row-weighted mixture of uniform, full-batch table
        estimates. This reproduces homogeneous q=1 batches exactly and counts the
        common module work once. The mixture is an approximation, not a
        measurement of the ragged batch. Query-width expansion supplies the
        compute budget; indexer KV reads are shared within each Q-block using SOL.
        """
        rows_by_context, requests_by_context, tiles_by_context = self._decode_geometry(
            buckets
        )
        rows = sum(rows_by_context.values())
        if not rows:
            return 0.0
        if projection_rows is not None and projection_rows < rows:
            raise ConfigurationError(
                "attention full projection rows must cover logical queries"
            )
        if projection_rows is not None and projection_rows > rows:
            total = self._decode_with_capture_projection(
                rows_by_context, tiles_by_context, projection_rows
            )
        else:
            total = sum(
                count
                / rows
                * self._decode_us(
                    rows,
                    context,
                    indexer_kv_fraction=tiles_by_context[context] / count,
                    byte_weight=count / rows,
                )
                for context, count in sorted(rows_by_context.items())
            )
        self.seconds += total * 1e-6
        self._record_decode_batch(
            rows_by_context, requests_by_context, tiles_by_context
        )
        return total

    def _decode_with_capture_projection(
        self, rows_by_context, tiles_by_context, full_rows
    ):
        rows = sum(rows_by_context.values())
        base = increment = source_floor_rows = projection_rows = 0.0
        for kind in ("csa", "hca"):
            layers = self.layer_counts[kind]
            for context, count in sorted(rows_by_context.items()):
                weight = count / rows
                fraction = tiles_by_context[context] / count
                latency, key = self._decode_layer(
                    kind,
                    rows,
                    context,
                    indexer_kv_fraction=fraction,
                    byte_weight=weight,
                )
                common, phase = self.sol.shared_phase_us(
                    key, indexer_kv_fraction=fraction
                )
                projected = max(full_rows, key["batch_size"])
                new_common = self.sol.shared_phase_us(
                    {**key, "batch_size": projected}, indexer_kv_fraction=fraction
                )[0]
                efficiency = latency / (common + phase)
                base += layers * weight * latency
                increment += (
                    layers * weight * max(0.0, new_common - common) * efficiency
                )
                source_floor_rows += layers * weight * key["batch_size"]
                projection_rows += layers * weight * projected
        stats = self.composition_stats
        stats["capture_batches"] += 1
        stats["capture_logical_rows"] += rows
        stats["capture_descriptor_rows"] += full_rows
        stats["capture_padding_rows"] += full_rows - rows
        # These are weighted source-bound budgets, not physical tensor rows.
        stats["capture_source_floor_layer_rows"] += source_floor_rows
        stats["capture_projection_bound_layer_rows"] += projection_rows
        stats["capture_increment_layer_rows"] += projection_rows - source_floor_rows
        stats["capture_shared_increment_seconds"] += increment * 1e-6
        return base + increment

    def _decode_geometry(self, buckets):
        rows_by_context = Counter()
        requests_by_context = Counter()
        tiles_by_context = Counter()
        for (context, query_width), count in buckets.items():
            if count > 0:
                rows_by_context[context] += count * max(1, int(query_width))
                requests_by_context[context] += count
                width = max(1, int(query_width))
                tiles_by_context[context] += count * math.ceil(
                    width / self.indexer_queries_per_tile
                )
        return rows_by_context, requests_by_context, tiles_by_context

    def _record_decode_batch(
        self, rows_by_context, requests_by_context, tiles_by_context
    ):
        rows = sum(rows_by_context.values())
        self.decode_batch_stats["batches"] += 1
        self.decode_batch_stats["length_groups"] += len(rows_by_context)
        self.decode_batch_stats["query_rows"] += rows
        self.decode_batch_stats["requests"] += sum(requests_by_context.values())
        self.decode_batch_stats["indexer_query_tiles"] += sum(tiles_by_context.values())
        if rows > sum(requests_by_context.values()):
            self.mtp_kv_stats["batches"] += 1

    def estimate_batch(self, context_groups, decode_buckets, *, projection_rows=None):
        """Own common projections once across context groups and decode.

        Whole-module samples cannot identify individual kernel times. Allocate
        each sample to two separate SOL bounds, then normalize both parts to
        that measured latency. Keep the phase parts and price the common part
        once at full tensor rows, using row-weighted anchor efficiencies.
        This is an explicit composition estimate, with no fitted multiplier or
        additional overlap credit. Single context groups and decode-only batches
        use their existing complete module estimates unchanged.
        """
        groups = {shape: count for shape, count in context_groups.items() if count > 0}
        rows_by_context, requests_by_context, tiles_by_context = self._decode_geometry(
            decode_buckets
        )
        decode_rows = sum(rows_by_context.values())
        context_rows = sum(queries * count for (queries, _), count in groups.items())
        rows = context_rows + decode_rows
        if groups and projection_rows is not None and projection_rows != rows:
            raise ConfigurationError(
                "context/mixed attention uses its eager logical tensor rows"
            )
        if not groups:
            return self.estimate_decode_batch(
                decode_buckets, projection_rows=projection_rows
            )
        if len(groups) == 1 and not decode_rows:
            (queries, context), count = next(iter(groups.items()))
            return self.estimate_prefill(count, queries, context)
        total = shared_total = phase_total = standalone_total = 0.0
        for kind in ("csa", "hca"):
            parts = []
            for (queries, context), count in sorted(groups.items()):
                estimate = self._query(kind, "context", count, queries, context)
                key = self._sol_key(kind, "context", count, queries, context, estimate)
                common, phase = self.sol.shared_phase_us(key)
                efficiency = estimate.latency_us / (common + phase)
                parts.append(
                    (
                        queries * count,
                        1.0,
                        efficiency,
                        common,
                        phase,
                        key["batch_size"] * key["input_seq_len"],
                    )
                )
            for context, count in sorted(rows_by_context.items()):
                fraction = tiles_by_context[context] / count
                weight = count / decode_rows
                latency, key = self._decode_layer(
                    kind,
                    decode_rows,
                    context,
                    indexer_kv_fraction=fraction,
                    byte_weight=weight,
                )
                common, phase = self.sol.shared_phase_us(
                    key, indexer_kv_fraction=fraction
                )
                efficiency = latency / (common + phase)
                parts.append(
                    (count, weight, efficiency, common, phase, key["batch_size"])
                )
            # Keep the largest source row floor for small clamped samples.
            # No phase-specific source floor or measured residual is removed.
            common_rows = max(rows, max(part[5] for part in parts))
            common_key = self._sol_key(kind, "generation", common_rows, 1, 0)
            common_us = self.sol.shared_phase_us(common_key)[0]
            efficiency = sum(count * eta for count, _, eta, *_ in parts) / rows
            shared = common_us * efficiency
            phase = sum(weight * eta * work for _, weight, eta, _, work, _ in parts)
            standalone = sum(
                weight * eta * (common + work)
                for _, weight, eta, common, work, _ in parts
            )
            layers = self.layer_counts[kind]
            shared_total += layers * shared
            phase_total += layers * phase
            standalone_total += layers * standalone
            total += layers * (shared + phase)
            self.composition_stats["source_floor_projection_rows"] += (
                layers * common_rows
            )
        if decode_rows:
            self._record_decode_batch(
                rows_by_context, requests_by_context, tiles_by_context
            )
        self.seconds += total * 1e-6
        stats = self.composition_stats
        stats["batches"] += 1
        stats["mixed_phase_batches"] += int(bool(decode_rows))
        stats["context_group_batches"] += int(len(groups) > 1)
        stats["context_groups"] += len(groups)
        stats["decode_length_groups"] += len(rows_by_context)
        stats["query_rows"] += rows
        stats["shared_layer_invocations"] += self.model.num_layers
        stats["logical_projection_rows"] += rows * self.model.num_layers
        stats["shared_projection_seconds"] += shared_total * 1e-6
        stats["phase_seconds"] += phase_total * 1e-6
        stats["standalone_module_seconds"] += standalone_total * 1e-6
        stats["avoided_shared_seconds"] += (standalone_total - total) * 1e-6
        return total

    def stats_dict(self):
        return {
            "seconds": self.seconds,
            "counts": dict(self.counts),
            "decode_batches": dict(self.decode_batch_stats),
            "speculative_kv": dict(self.mtp_kv_stats),
            "composition": dict(self.composition_stats),
        }

    def describe(self):
        return {
            "backend": self.binding["key"].get("kernel_source"),
            "layer_counts": dict(self.layer_counts),
            "module_scope": "projection + compressor/indexer + sparse attention + output projection",
            "kv_cache_source_layout": self.binding.get("layout", {}),
            "serving_kv_dtype": self.model.kv_cache_dtype,
            "serving_indexer_dtype": self.model.components.indexer_cache_dtype,
            "sources": self.backend.package.describe_sources(
                self.table, self.binding["key"]
            ),
            "extrapolation": "measured boundary with parameterized compressed attention FLOP/byte SOL growth, max 64x",
            "extrapolation_sol": self.sol.describe(),
            "extrapolation_axis_binding": "each inner-axis growth uses its measured outer-axis node",
            "decode_batch_composition": "query-row-weighted uniform full-batch module estimates; common projections counted once",
            "capture_projection_rows": "positive shared SOL increment beyond actual-query source floors at eligible descriptor rows; sparse/indexer query ownership unchanged",
            "mixed_batch_composition": {
                "shared_scope": "full-tensor input/output, compressor-input and index-query/weight projections, weights and activation preparation",
                "phase_scope": "sparse attention, index scoring/KV reads and compressor pointwise work; context and decode kept separate",
                "allocation": "two SOL roofline bounds normalized to each whole-module sample; common efficiency row-weighted at total query rows",
                "source_clamps": "each phase retains its source geometry; common rows retain the largest source floor",
                "homogeneous": "single context group and decode-only batches unchanged",
                "uncertainty": "whole-module tables do not measure component launches, ragged fusion or aux-stream overlap; no extra overlap credit",
                "fitted_multiplier": False,
            },
            "decode_query_width": "expanded query compute and sparse-attention reads; indexer KV reuse within Q-blocks",
            "mtp_indexer_kv_reuse": {
                "basis": "pinned DeepGEMM SM100 paged MQA groups request Q-blocks; BLOCK_Q = 128 / indexer heads",
                "query_math": "unchanged at the full expanded query-row count",
                "indexer_queries_per_tile": self.indexer_queries_per_tile,
                "indexer_kv_read_fraction": "sum(requests * ceil(query width / BLOCK_Q)) / query rows, per context; source geometry including clamps",
                "cross_q_block_l2_reuse": "not credited without measured traffic; tile reads are a conservative streaming estimate",
                "sparse_attention_kv_reads": "per query retained; FlashInfer sparse indices may differ across queries",
                "layout_correction": False,
                "latency": "measured full-query-row module times adjusted/source SOL; math bound retained",
            },
            "generation_ragged_grid": "on out-of-range neighbors, use same-geometry batch slices within the existing 64x context/batch bound",
            "context_query_grid": "query floors across prefixes for the matching geometry/batch; coupled query/prefix samples do not inflate supported queries",
        }
