"""Numerical reference for the experimental value-plus-rotary-key FP8 cache.

This CPU reference defines the bytes and arithmetic without asserting model
quality, a compiled LP30 kernel or equality to independently quantized K/V.
"""

from dataclasses import dataclass

import numpy as np

from TokenSim.config.kv_storage import rotary_indices
from TokenSim.errors import ConfigurationError


def _levels():
    bits = np.arange(127, dtype=np.uint8)
    exponent, fraction = bits >> 3, bits & 7
    return np.where(
        exponent == 0,
        fraction.astype(np.float32) * 2.0**-9,
        (1 + fraction.astype(np.float32) / 8) * np.exp2(exponent.astype(int) - 7),
    ).astype(np.float32)


LEVELS = _levels()


def _finite(value, name):
    result = np.asarray(value, dtype=np.float32)
    if result.ndim < 1 or not np.isfinite(result).all():
        raise ConfigurationError(f"{name} needs finite vectors")
    return result


def _quantize(value):
    """Per-vector FP32 scale, E4M3FN saturation and nearest-even rounding."""
    maximum = np.max(np.abs(value), axis=-1, keepdims=True)
    scale = np.maximum(maximum / 448, np.finfo(np.float32).tiny)
    scale = np.where(maximum == 0, np.float32(1), scale).astype(np.float32)
    normalized = np.minimum(np.abs(value) / scale, 448)
    upper = np.clip(np.searchsorted(LEVELS, normalized), 1, 126)
    lower = upper - 1
    lower_distance = normalized - LEVELS[lower]
    upper_distance = LEVELS[upper] - normalized
    choose_upper = (upper_distance < lower_distance) | (
        (upper_distance == lower_distance) & ((upper & 1) == 0)
    )
    codes = np.where(choose_upper, upper, lower).astype(np.uint8)
    codes |= np.signbit(value).astype(np.uint8) << 7
    return np.ascontiguousarray(codes), np.ascontiguousarray(scale)


def _decode(codes, scales):
    magnitude = codes & 127
    if np.any(magnitude == 127):
        raise ConfigurationError("E4M3FN NaN code is invalid in the shared KV cache")
    sign = np.where(codes & 128, np.float32(-1), np.float32(1))
    return LEVELS[magnitude] * sign * scales


@dataclass(frozen=True)
class PackedKV:
    value_codes: np.ndarray
    rotary_key_codes: np.ndarray
    value_scales: np.ndarray
    rotary_key_scales: np.ndarray

    def __post_init__(self):
        values, keys = self.value_codes, self.rotary_key_codes
        if values.ndim < 1 or keys.shape != (
            *values.shape[:-1],
            len(rotary_indices(values.shape[-1])),
        ):
            raise ConfigurationError("packed shared KV geometry differs")
        for codes, scales in (
            (values, self.value_scales),
            (keys, self.rotary_key_scales),
        ):
            if (
                codes.dtype != np.uint8
                or scales.dtype != np.float32
                or scales.shape != (*codes.shape[:-1], 1)
                or not np.isfinite(scales).all()
                or np.any(scales <= 0)
                or np.any((codes & 127) == 127)
            ):
                raise ConfigurationError("invalid packed shared KV codes or scales")

    @property
    def nbytes(self):
        return sum(
            array.nbytes
            for array in (
                self.value_codes,
                self.rotary_key_codes,
                self.value_scales,
                self.rotary_key_scales,
            )
        )

    def decode(self):
        values = _decode(self.value_codes, self.value_scales)
        keys = values.copy()
        keys[..., rotary_indices(values.shape[-1])] = _decode(
            self.rotary_key_codes, self.rotary_key_scales
        )
        return keys, values


def encode(key, value):
    """Encode post-RoPE keys and normalized values with independent scales."""
    keys, values = _finite(key, "keys"), _finite(value, "values")
    if keys.shape != values.shape or any(size == 0 for size in values.shape):
        raise ConfigurationError("shared KV needs matching nonempty key/value vectors")
    indices = rotary_indices(values.shape[-1])
    value_codes, value_scales = _quantize(values)
    key_codes, key_scales = _quantize(keys[..., indices])
    return PackedKV(value_codes, key_codes, value_scales, key_scales)


def fold_query(query, key_gamma):
    """Fold learned target key normalization weights into nonrotary queries."""
    query, gamma = _finite(query, "query"), _finite(key_gamma, "key gamma")
    if gamma.shape != (query.shape[-1],):
        raise ConfigurationError("key gamma must match the full head dimension")
    weights = gamma.copy()
    weights[list(rotary_indices(query.shape[-1]))] = 1
    return query * weights
