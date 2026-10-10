"""Request generation and latency targets, independent of models and hardware."""

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

from TokenSim.errors import ConfigurationError
from TokenSim.results import MetricData


@dataclass(frozen=True)
class WorkloadConfig:
    distribution: str = "uniform"
    qps: float = 1.0
    prefill_mean_len: int = 512
    prefill_range_len: int = 0
    decode_mean_len: int = 128
    decode_range_len: int = 0
    decode_len_distribution: str = "uniform"
    first_token_latency: MetricData | None = None
    decode_token_latency: MetricData | None = None

    def __post_init__(self):
        if self.distribution not in {"uniform", "poisson", "burst"}:
            raise ConfigurationError(
                "workload distribution must be uniform, poisson or burst"
            )
        if (
            not isinstance(self.qps, (int, float))
            or isinstance(self.qps, bool)
            or not math.isfinite(self.qps)
            or self.qps <= 0
        ):
            raise ConfigurationError("workload qps must be finite and positive")
        for name in (
            "prefill_mean_len",
            "decode_mean_len",
            "prefill_range_len",
            "decode_range_len",
        ):
            value = getattr(self, name)
            lower = 1 if "mean" in name else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ConfigurationError(
                    f"workload {name} must be an integer >= {lower}"
                )
        if self.decode_len_distribution not in {
            "uniform",
            "exponential",
            "capped_exponential",
            "burst",
        }:
            raise ConfigurationError("unsupported decode_len_distribution")
        if self.prefill_mean_len - self.prefill_range_len // 2 < 1:
            raise ConfigurationError(
                "prefill range must not include nonpositive lengths"
            )
        if (
            self.decode_len_distribution == "uniform"
            and self.decode_mean_len - self.decode_range_len // 2 < 1
        ):
            raise ConfigurationError(
                "decode range must not include nonpositive lengths"
            )
        if (
            self.decode_len_distribution == "capped_exponential"
            and self.decode_range_len == 1
        ):
            raise ConfigurationError("capped exponential range must allow lengths >= 2")
        for name in ("first_token_latency", "decode_token_latency"):
            value = getattr(self, name)
            if isinstance(value, dict):
                value = MetricData(**{"p50": None, "p99": None, "max": None, **value})
                object.__setattr__(self, name, value)
            if value is not None:
                if not isinstance(value, MetricData):
                    raise ConfigurationError(f"{name} must be a latency target object")
                for target in (value.p50, value.p99, value.max):
                    if target is not None and (not math.isfinite(target) or target < 0):
                        raise ConfigurationError(
                            f"{name} targets must be finite and nonnegative"
                        )

    @classmethod
    def from_file(cls, path):
        try:
            raw = json.loads(Path(path).read_text())
            if not isinstance(raw, dict):
                raise ConfigurationError("workload configuration must be a JSON object")
            return cls(**raw)
        except (OSError, ValueError, TypeError) as error:
            raise ConfigurationError(
                f"invalid workload configuration {path}: {error}"
            ) from error

    def from_args(self, args):
        names = (
            "distribution",
            "qps",
            "prefill_mean_len",
            "prefill_range_len",
            "decode_mean_len",
            "decode_range_len",
            "decode_len_distribution",
        )
        return replace(
            self,
            **{
                name: getattr(args, name)
                for name in names
                if getattr(args, name, None) is not None
            },
        )
