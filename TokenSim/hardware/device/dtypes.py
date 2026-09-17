"""Canonical dtype names, storage sizes, and compute-pipeline mapping.

Every operator table and analytical model uses the canonical names defined here
so that ``bf16``, ``bfloat16`` and ``half`` all resolve to the same key.
"""

from __future__ import annotations

from TokenSim.errors import ConfigurationError

# Canonical dtype names used throughout the operator tables.
_DTYPE_ALIASES = {
    "fp32": "fp32",
    "float32": "fp32",
    "float": "fp32",
    "tf32": "tf32",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp8": "fp8",
    "fp8_e4m3": "fp8",
    "float8": "fp8",
    "fp8_block": "fp8_block",
    "int8": "int8",
    "int8_wo": "int8_wo",
    "int8_sq": "int8_sq",
    "sq": "int8_sq",
    "int4": "int4",
    "int4_wo": "int4_wo",
    "nvfp4": "nvfp4",
    "fp4": "nvfp4",
}

# Bytes per element for the *storage* dtype (weights or activations).
_DTYPE_BYTES = {
    "fp32": 4.0,
    "tf32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "fp8_block": 1.0,
    "int8": 1.0,
    "int8_wo": 1.0,
    "int8_sq": 1.0,
    "int4": 0.5,
    "int4_wo": 0.5,
    "nvfp4": 0.5,
}

# Which peak-compute entry a storage dtype executes on. Weight-only quantized
# GEMMs dequantize to 16-bit activations and run on the fp16 tensor pipe.
_COMPUTE_DTYPE = {
    "fp32": "fp32",
    "tf32": "tf32",
    "fp16": "fp16",
    "bf16": "bf16",
    "fp8": "fp8",
    "fp8_block": "fp8",
    "int8": "int8",
    "int8_sq": "int8",
    "int8_wo": "fp16",
    "int4": "int4",
    "int4_wo": "fp16",
    "nvfp4": "nvfp4",
}


def normalize_dtype(dtype: str) -> str:
    key = str(dtype).strip().lower()
    try:
        return _DTYPE_ALIASES[key]
    except KeyError as exc:
        raise ConfigurationError(f"unsupported dtype {dtype!r}") from exc


def dtype_bytes(dtype: str) -> float:
    return _DTYPE_BYTES[normalize_dtype(dtype)]


def compute_dtype(dtype: str) -> str:
    return _COMPUTE_DTYPE[normalize_dtype(dtype)]
