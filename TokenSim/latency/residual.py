"""Measured compressed transformer mHC pre/post residual mixing cost."""

from dataclasses import dataclass

from TokenSim.utils import BoundedCache
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.lookup import (
    LookupPolicy,
    MissingOperatorDataError,
    OperatorLookup,
)


@dataclass(frozen=True)
class MHCEstimate:
    latency_us: float
    match: str
    source_id: str


class ResidualMixingEstimator:
    def __init__(self, backend):
        self.backend = backend
        model = backend.model
        self.binding = model.operator_modules.get("mhc")
        if not self.binding:
            raise ConfigurationError(
                "measured residual mixing needs an operator_modules.mhc binding"
            )
        self.table = self.binding["table"]
        if (
            model.components is None
            or not backend.package
            or not backend.package.tables.get(self.table)
        ):
            raise ConfigurationError(
                "native compressed attention mHC requires its measured table"
            )
        self.lookup = OperatorLookup(
            backend.package,
            LookupPolicy(extrapolate="analytical"),
            analytical_scaler=lambda _table, key: float(key["num_tokens"]),
        )
        self.cache = BoundedCache(65536)
        self.counts = {}
        self.seconds = 0.0

    def reset_stats(self):
        self.counts.clear()
        self.seconds = 0.0

    def estimate(self, tokens):
        total = 0.0
        for op in ("pre", "post"):
            key = {
                "op_name": op,
                **self.binding["key"],
                "kernel_source": self.binding["kernels"][op],
                "hidden_size": self.backend.model.hidden_size,
                "hc_mult": self.backend.model.components.hc_mult,
                "sinkhorn_iters": self.backend.model.components.hc_sinkhorn_iters,
                "num_tokens": max(1, int(tokens)),
            }
            identity = tuple(key.items())
            result = self.cache.get(identity)
            if result is None:
                try:
                    row = self.lookup.lookup(self.table, key)
                except MissingOperatorDataError as exc:
                    self.backend.stats.record_missing(exc)
                    raise ConfigurationError(str(exc)) from exc
                result = MHCEstimate(row.latency_us, row.match_type, row.source_id)
                self.cache[identity] = result
            self.backend.stats.record(self.table, result.match)
            self.counts[f"{op}:{result.match}"] = (
                self.counts.get(f"{op}:{result.match}", 0) + 1
            )
            total += result.latency_us
        self.seconds += total * 1e-6
        return total

    def stats_dict(self):
        return {
            "native_mhc_seconds": self.seconds,
            "native_mhc_counts": dict(self.counts),
        }

    def describe(self):
        return {
            "backend": "vllm TileLang mhc_module",
            "scope": "pre + post, num_sites=2; one module per transformer layer",
            "hidden_size": self.backend.model.hidden_size,
            "hc_mult": self.backend.model.components.hc_mult,
            "sinkhorn_iters": self.backend.model.components.hc_sinkhorn_iters,
            "residual_dtype": "bf16",
            "parameter_dtype": "fp32",
            "source_grade": "B",
        }
