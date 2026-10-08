"""Resolve explicit MTP execution independently from accepted-token progress."""

import math

from TokenSim.errors import ConfigurationError


def resolve_mtp_tokens(model, acceptance_length, num_speculative_tokens=None):
    if not math.isfinite(acceptance_length) or acceptance_length < 1:
        raise ConfigurationError("MTP acceptance length must be finite and >=1")
    if num_speculative_tokens is None:
        # Preserve the established default reference K3 while making old
        # AL1 callers actually disable the complete path. Explicit K3/AL1 is
        # the distinct experiment that drafts but accepts only one token.
        num_speculative_tokens = 3 if model.deepseek_v4 and acceptance_length > 1 else 0
    if (isinstance(num_speculative_tokens, bool)
            or not isinstance(num_speculative_tokens, int)
            or num_speculative_tokens < 0):
        raise ConfigurationError("MTP speculative token count must be a nonnegative integer")
    if num_speculative_tokens and (model.deepseek_v4 is None or not model.deepseek_v4.num_nextn_predict_layers):
        raise ConfigurationError("MTP draft execution requires a model with MTP layers")
    if acceptance_length > 1 + num_speculative_tokens:
        raise ConfigurationError("MTP acceptance length cannot exceed1+K; disabled K0 requires AL1")
    return num_speculative_tokens
