"""Observed vLLM kernel families that supply additional ownership evidence."""


def kernel_operation(name):
    name = name.lower()
    if "rotary_embedding" in name:
        return "rope"
    if any(
        token in name for token in ("allreduce", "all_reduce", "cross_device_reduce")
    ):
        return "all_reduce"
    if "allgather" in name or "all_gather" in name:
        return "all_gather"
    if "reducescatter" in name or "reduce_scatter" in name:
        return "reduce_scatter"
    return None


def fused_members(operation, kernel_names):
    names = [name.lower() for name in kernel_names]
    if any("fused_add_rms_norm" in name for name in names):
        predecessor = {
            "ffn_norm": "attention_residual",
            "attention_norm": "ffn_residual",
            "final_norm": "ffn_residual",
        }.get(operation)
        if predecessor:
            return [predecessor, operation]
    if operation in ("moe_module", "output", "ffn_down", "embedding_lookup") and any(
        kernel_operation(name) == "all_reduce" for name in names
    ):
        return [operation, "all_reduce"]
    if operation == "lm_head" and any(
        kernel_operation(name) == "all_gather" for name in names
    ):
        return [operation, "all_gather"]
    return None
