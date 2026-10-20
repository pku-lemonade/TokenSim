"""Shared assertion errors and numeric comparisons for saved LPX evidence."""

import math

from TokenSim.config.validation import number
from TokenSim.errors import ConfigurationError


def require(condition, message):
    if not condition:
        raise ConfigurationError(f"LPX result audit: {message}")


def equal(actual, expected, label):
    number(actual, label)
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), label)
