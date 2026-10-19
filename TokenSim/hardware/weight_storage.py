"""Packed weight bytes and row-wise MXFP4 scale storage."""

from math import ceil, prod

from TokenSim.config.validation import integer
from TokenSim.hardware.device import dtype_bytes, normalize_dtype


def weight_storage_bytes(shape, dtype, *, scale_group_size=32):
    """Return payload and metadata bytes for one stored tensor.

    MXFP4 stores two elements per byte and one 8-bit scale per 32 last-axis
    elements. Padding and scale counts are computed separately for each row.
    """
    for dimension in shape:
        integer(dimension, "weight dimension", minimum=1)
    integer(scale_group_size, "scale group size", minimum=1)
    canonical = normalize_dtype(dtype)
    if canonical in {"mxfp4", "mxfp4_wo"}:
        rows, columns = prod(shape[:-1]), shape[-1]
        return rows * ceil(columns / 2), rows * ceil(columns / scale_group_size)
    return ceil(prod(shape) * dtype_bytes(canonical)), 0
