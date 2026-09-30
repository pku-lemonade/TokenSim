"""User-selected relative simulation error bands; data integrity is separate."""

from __future__ import annotations

import math

ACCEPTABLE_ERROR_PERCENT = 25.0
PREFERRED_ERROR_PERCENT = 10.0


def error_assessment(actual, reference, *, comparable=True):
    """Keep missing/incompatible values unassessed, and handle zero references."""
    if not comparable:
        return {
            "delta_percent": None,
            "acceptable": None,
            "preferred": None,
            "status": "incomparable",
        }
    if (
        actual is None
        or reference is None
        or not math.isfinite(actual)
        or not math.isfinite(reference)
    ):
        return {
            "delta_percent": None,
            "acceptable": None,
            "preferred": None,
            "status": "missing",
        }
    if reference == 0:
        error = 0.0 if actual == 0 else None
    else:
        error = (actual / reference - 1) * 100
    # Compare the values, not rounded percentages: 1.25 at the boundary is
    # acceptable, while 1.250001 is outside.
    accepted = (
        actual == 0
        if reference == 0
        else abs(actual - reference) <= abs(reference) * 0.25
    )
    preferred = (
        actual == 0
        if reference == 0
        else abs(actual - reference) <= abs(reference) * 0.10
    )
    return {
        "delta_percent": error,
        "acceptable": accepted,
        "preferred": preferred,
        "status": "preferred" if preferred else "acceptable" if accepted else "outside",
    }
