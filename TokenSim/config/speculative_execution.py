"""Resolve explicit MTP execution independently from accepted-token progress."""

import math
from dataclasses import dataclass

from TokenSim.errors import ConfigurationError
from TokenSim.config.validation import number


@dataclass(frozen=True)
class MTPAcceptanceConfig:
    """Distribution of output tokens accepted by one verification round."""

    kind: str = "mean"
    rates: tuple[float, ...] = ()
    distribution: tuple[float, ...] = ()

    def __post_init__(self):
        if self.kind not in ("mean", "conditional", "empirical", "legacy"):
            raise ConfigurationError("unknown MTP acceptance kind")
        for name in ("rates", "distribution"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)):
                raise ConfigurationError(f"MTP {name} must be an array")
            for value in values:
                number(value, f"MTP {name}", minimum=0, maximum=1)
            object.__setattr__(self, name, tuple(float(v) for v in values))
        if self.kind != "conditional" and self.rates:
            raise ConfigurationError("MTP rates require conditional acceptance")
        if self.kind != "empirical" and self.distribution:
            raise ConfigurationError("MTP distribution requires empirical acceptance")

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict) or value.keys() - {
            "kind",
            "rates",
            "distribution",
        }:
            raise ConfigurationError("invalid MTP acceptance configuration")
        return cls(**value)

    def validate(self, drafts, mean):
        number(mean, "MTP acceptance length", minimum=1, maximum=1 + drafts)
        if self.kind in ("conditional", "empirical") and mean != 1:
            raise ConfigurationError(
                "MTP mean and an explicit distribution are mutually exclusive"
            )
        if self.kind == "conditional" and (drafts < 1 or len(self.rates) != drafts):
            raise ConfigurationError(
                "conditional MTP requires one rate per draft position"
            )
        if self.kind == "empirical" and (
            drafts < 1
            or len(self.distribution) != drafts + 1
            or not math.isclose(
                math.fsum(self.distribution), 1.0, rel_tol=0, abs_tol=1e-12
            )
        ):
            raise ConfigurationError(
                "empirical MTP requires K+1 probabilities summing to one"
            )

    def describe(self, drafts, mean):
        expected = mean
        if self.kind == "conditional":
            probability, expected = 1.0, 1.0
            for rate in self.rates:
                probability *= rate
                expected += probability
        elif self.kind == "empirical":
            expected = math.fsum((i + 1) * p for i, p in enumerate(self.distribution))
        return dict(
            kind=self.kind,
            rates=list(self.rates),
            distribution=list(self.distribution),
            expected_accepted_tokens=expected,
            sampling_version=(
                "linear_legacy" if self.kind == "legacy" else "blake2b_round_v1"
            ),
            draft_tokens=drafts,
        )


def validate_mtp_token_budget(max_num_batched_tokens, num_speculative_tokens):
    """A decode step must fit the full verification query, including rejected drafts."""
    width = 1 + num_speculative_tokens
    if max_num_batched_tokens is not None and max_num_batched_tokens < width:
        raise ConfigurationError(
            f"max_num_batched_tokens must be at least {width} for MTP "
            f"with {num_speculative_tokens} draft tokens"
        )


def resolve_mtp_tokens(model, acceptance_length, num_speculative_tokens=None):
    if not math.isfinite(acceptance_length) or acceptance_length < 1:
        raise ConfigurationError("MTP acceptance length must be finite and >=1")
    if num_speculative_tokens is None:
        num_speculative_tokens = 0
    if (
        isinstance(num_speculative_tokens, bool)
        or not isinstance(num_speculative_tokens, int)
        or num_speculative_tokens < 0
    ):
        raise ConfigurationError(
            "MTP speculative token count must be a nonnegative integer"
        )
    if num_speculative_tokens and (
        model.components is None or not model.components.num_nextn_predict_layers
    ):
        raise ConfigurationError("MTP draft execution requires a model with MTP layers")
    if acceptance_length > 1 + num_speculative_tokens:
        raise ConfigurationError(
            "MTP acceptance length cannot exceed 1+K; K=0 (disabled) requires an acceptance length of 1"
        )
    return num_speculative_tokens
