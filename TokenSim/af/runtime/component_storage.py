"""Apply per-component storage precision and packed tensor metadata."""

from dataclasses import replace

from TokenSim.hardware.weight_storage import weight_storage_bytes
from TokenSim.errors import ConfigurationError

COMPONENT_OPERATORS = {
    "attention": "attention",
    "router": "router",
    "embedding_lm_head": "embedding_lm_head",
    "dense_ffn": "dense_ffn",
    "routed_expert": "routed_experts",
    "shared_expert": "shared_expert",
}


def stored_components(
    components, *, operator_dtypes, default_dtype, projections, mla, embedding_role
):
    result = []
    for component in components:
        operator = COMPONENT_OPERATORS[component.category]
        dtype = operator_dtypes.get(operator, default_dtype)
        if operator not in operator_dtypes and dtype not in {"mxfp4", "mxfp4_wo"}:
            result.append(component)
            continue
        shape = dict(component.shape)
        h = shape["hidden_size"]
        if component.category == "attention":
            if mla:
                raise ConfigurationError(
                    "MLA component precision overrides require a supported adapter"
                )
            q = shape["q_heads_local"] * shape["head_dim"]
            kv = shape["kv_heads_local"] * shape["head_dim"]
            tensors = ((q + 2 * kv, h), (h, q))
        elif component.category == "router":
            tensors = ((shape["num_experts"], h),)
        elif component.category == "embedding_lm_head":
            copies = (
                1
                if embedding_role != "both" or ".tied." in component.component_id
                else 2
            )
            tensors = ((shape["vocab_size"] // shape["tp_size"], h),) * copies
        else:
            inter = shape["local_intermediate_size"]
            tensors = ((inter, h),) * projections + ((h, inter),)
        storage = [weight_storage_bytes(tensor, dtype) for tensor in tensors]
        payload = sum(row[0] for row in storage)
        scales = sum(row[1] for row in storage)
        result.append(
            replace(
                component,
                bytes=payload + scales,
                storage_dtype=dtype,
                scale_bytes=scales,
                formula=component.formula
                + "; row-wise packed payload and scale metadata",
            )
        )
    return tuple(result)
