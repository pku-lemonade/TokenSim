"""Query attention measurements with bounded geometry and projection corrections."""

from __future__ import annotations

from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.lookup import MissingOperatorDataError, OperatorLookup
from TokenSim.operator_data.package import OperatorDataPackage
from .estimates import AttentionEstimate


class AttentionLookup:
    """Query attention measurements with bounded geometry and projection corrections."""

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
            row = None
            if exc.kind == "out_of_range":
                try:
                    if phase == "generation":
                        row = self._query_supported_generation_slices(key)
                        ragged_grid = True
                    else:
                        row = self._query_first_prefix(key, exc)
                        clamped = row is not None
                except MissingOperatorDataError:
                    row = None
            if row is None:
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

    def _query_first_prefix(self, key, error):
        """Raise a short prefix to the first one the failing slice measured.

        A prefix can lie further below a neighboring (batch, query) slice's
        first measured prefix than extrapolation allows (B300 CSA measures q=4
        only from prefix 128). Like the batch and query floors, hold that row.
        ``None`` when the failure is not such a prefix.
        """
        # At most one retry per neighboring (batch, query) slice.
        for _ in range(4):
            values = self._axis_values(error.key, "context_len")
            if not values or key["context_len"] >= values[0]:
                return None
            key["context_len"] = values[0]
            try:
                return self.lookup.lookup(self.table, key)
            except MissingOperatorDataError as exc:
                if exc.kind != "out_of_range":
                    raise
                error = exc
        return None

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
