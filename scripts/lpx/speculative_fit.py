"""Acceptance arithmetic for chain speculative decoding at a fixed step time.

A verification step checks ``K`` draft tokens plus one target token.  With an
independent per-position acceptance probability ``alpha`` the expected number
of committed tokens per step is ``(1 - alpha ** (K + 1)) / (1 - alpha)``.
"""

from TokenSim.errors import ConfigurationError


def committed_tokens(alpha, draft_tokens):
    """Expected tokens committed by one verification step."""
    if not 0 <= alpha <= 1 or draft_tokens < 0:
        raise ConfigurationError("acceptance must lie in [0, 1] with K >= 0")
    if alpha == 1:
        return draft_tokens + 1.0
    return (1 - alpha ** (draft_tokens + 1)) / (1 - alpha)


def acceptance_for(tokens, draft_tokens):
    """Acceptance giving ``tokens`` committed per step, or None if unreachable."""
    if not 1 <= tokens <= draft_tokens + 1:
        return None
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2
        if committed_tokens(middle, draft_tokens) < tokens:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def token_interval(step_s, low_tps, high_tps):
    """Committed tokens per step that place the speed inside the band."""
    return low_tps * step_s, high_tps * step_s


def shared_match(steps_s, bands, draft_tokens):
    """Acceptance interval meeting every band with one acceptance, or None.

    ``steps_s`` and ``bands`` pair each workload's step time with its
    ``(low, high)`` TPS band.  The result is ``(alpha_low, alpha_high)``.
    """
    low, high = 1.0, draft_tokens + 1.0
    for step, (low_tps, high_tps) in zip(steps_s, bands, strict=True):
        a, b = token_interval(step, low_tps, high_tps)
        low, high = max(low, a), min(high, b)
    if low > high:
        return None
    return acceptance_for(low, draft_tokens), acceptance_for(high, draft_tokens)
