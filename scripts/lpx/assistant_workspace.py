"""Cache-owner scratch for assistant attention, separate from target scratch."""

from TokenSim.config.kv_storage import (
    GLOBAL_KV_LAYOUTS,
    SHARED_KV_WORKSPACES,
    extra_workspace,
    shared_workspace_components,
)
from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError


def remote_workspace(
    spec,
    *,
    tp,
    kvp,
    tile_keys,
    global_kv_layout="separate",
    shared_kv_workspace="conservative",
    shared_kv_kernel="assemble",
):
    """Round a cache-owner tile/convert/score/output envelope to whole MiB."""
    for value, name in ((tp, "TP"), (kvp, "DCP"), (tile_keys, "key tile")):
        integer(value, f"assistant remote {name}", minimum=1)
    if tp % kvp:
        raise ConfigurationError("assistant remote DCP must divide TP")
    if (
        global_kv_layout not in GLOBAL_KV_LAYOUTS
        or shared_kv_workspace not in SHARED_KV_WORKSPACES
    ):
        raise ConfigurationError("invalid assistant shared KV storage/workspace policy")
    if shared_kv_workspace != "conservative" and (
        global_kv_layout == "separate" or shared_kv_workspace != shared_kv_kernel
    ):
        raise ConfigurationError(
            "compact assistant workspace must match the shared KV kernel"
        )
    maximum = 0
    for dim, window, kv in zip(spec.dims, spec.windows, spec.kv_heads):
        attention_tp = tp if window else tp // kvp
        if spec.heads % attention_tp or kv % attention_tp:
            raise ConfigurationError(
                "assistant remote cache head shards are indivisible"
            )
        heads, local_kv = spec.heads // attention_tp, kv // attention_tp
        state = 8 * tile_keys * heads + 4 * heads * dim + 8 * heads * (dim + 1)
        envelope = 6 * tile_keys * local_kv * dim + state
        if not window and global_kv_layout != "separate":
            geometry = dict(
                head_dim=dim,
                kv_heads=local_kv,
                query_heads=heads,
                queries=1,
                tile_keys=tile_keys,
                buffers=2,
            )
            if shared_kv_workspace == "conservative":
                envelope += extra_workspace(**geometry)
                # Retain at least the explicit two-buffer physical tile envelope.
                explicit = shared_workspace_components(kernel="assemble", **geometry)
                envelope = max(envelope, state + sum(explicit.values()))
            else:
                explicit = shared_workspace_components(
                    kernel=shared_kv_kernel, **geometry
                )
                # The explicit split kernel already includes its extra score tile.
                envelope = state - 4 * tile_keys * heads + sum(explicit.values())
        maximum = max(maximum, envelope)
    unit = 1048576
    return max(unit, (maximum + unit - 1) // unit * unit)
