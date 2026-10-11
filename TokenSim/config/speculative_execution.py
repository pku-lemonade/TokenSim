"""Resolve explicit MTP execution independently from accepted-token progress."""

import math

from TokenSim.errors import ConfigurationError


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
