"""CPU tiled-attention oracle for explicit shared-cache reconstruction kernels.

This uses FP32 arithmetic to check buffer contents and online-softmax algebra.
It does not measure BF16 LP30 execution, model quality or a hardware memory peak.
"""

import numpy as np

from TokenSim.config.kv_storage import rotary_indices
from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError
from scripts.lpx.kv_codec import fold_query


def reference_attention(
    packed, query, key_gamma, *, tile_keys=256, kernel="assemble", scale=1.0
):
    """Attend to one KV head; rows of query may be its different query heads."""
    integer(tile_keys, "reference attention key tile", minimum=1)
    number(scale, "reference attention scale", positive=True)
    if kernel not in {"assemble", "split_dot_packed"}:
        raise ConfigurationError("unknown reference shared KV kernel")
    if packed.value_codes.ndim != 2 or not packed.value_codes.shape[0]:
        raise ConfigurationError(
            "reference attention needs a nonempty single-KV-head cache"
        )
    dim = packed.value_codes.shape[-1]
    source = np.asarray(query, dtype=np.float32)
    if source.ndim != 2 or not source.shape[0] or source.shape[-1] != dim:
        raise ConfigurationError("reference queries must have shape (rows, head_dim)")
    # The operator can overwrite its existing query buffer after Q normalization.
    # A reference-owned copy preserves caller inputs while checking that path.
    folded = source.copy()
    folded[...] = fold_query(folded, key_gamma)
    rotary = list(rotary_indices(dim))
    plain = sorted(set(range(dim)) - set(rotary))
    maximum = np.full(source.shape[0], -np.inf, dtype=np.float32)
    denominator = np.zeros_like(maximum)
    output = np.zeros_like(source)
    # Decode only the current tile; no complete-context BF16 K/V array exists.
    for start in range(0, packed.value_codes.shape[0], tile_keys):
        end = start + tile_keys
        current = type(packed)(
            packed.value_codes[start:end],
            packed.rotary_key_codes[start:end],
            packed.value_scales[start:end],
            packed.rotary_key_scales[start:end],
        )
        value, rotary_key = current.decode_parts()
        if kernel == "assemble":
            virtual_key = value.copy()
            virtual_key[:, rotary] = rotary_key
            scores = folded @ virtual_key.T
        else:
            nonrotary_key = np.ascontiguousarray(value[:, plain])
            scores = folded[:, rotary] @ rotary_key.T
            scores += folded[:, plain] @ nonrotary_key.T
        scores *= scale
        updated = np.maximum(maximum, np.max(scores, axis=-1))
        rescale = np.exp(maximum - updated)
        probabilities = np.exp(scores - updated[:, None])
        output *= rescale[:, None]
        output += probabilities @ value
        denominator = denominator * rescale + np.sum(probabilities, axis=-1)
        maximum = updated
    return output / denominator[:, None]
