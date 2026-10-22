"""Estimate decode batches, capture geometry and reusable layer work."""

from __future__ import annotations

from collections import Counter
import math
from TokenSim.errors import ConfigurationError


class DecodeAttention:
    """Estimate decode batches, capture geometry and reusable layer work."""

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
        parts = []
        for kind in ("csa", "hca"):
            latency = self._decode_layer(
                kind,
                batch_size,
                context_tokens,
                indexer_kv_fraction=indexer_kv_fraction,
                byte_weight=byte_weight,
            )[0]
            parts.append(self.layer_counts[kind] * latency)
            if self.trace_layer_costs is not None:
                self.trace_layer_costs[kind] = (
                    self.trace_layer_costs.get(kind, 0.0) + byte_weight * latency
                )
        total = sum(parts)
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
                if self.trace_layer_costs is not None:
                    self.trace_layer_costs[kind] = self.trace_layer_costs.get(
                        kind, 0.0
                    ) + weight * (latency + max(0.0, new_common - common) * efficiency)
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
