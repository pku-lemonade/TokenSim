"""Scalar checks shared by serving, model and storage configurations."""

import math
from numbers import Integral, Real

from TokenSim.errors import ConfigurationError


def integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return int(value)


def number(value, name, *, minimum=0, maximum=None, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < minimum
        or positive
        and value == 0
        or maximum is not None
        and value > maximum
    ):
        raise ConfigurationError(f"{name} must be finite and within its valid range")
    return value


def boolean(value, name):
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean")
    return value
