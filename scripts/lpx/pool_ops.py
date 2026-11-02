"""Shared embedding and greedy logits primitives with explicit row counts."""

from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError


def column_shards(width, ranks):
    integer(width, "pool column width", minimum=1)
    integer(ranks, "pool ranks", minimum=1)
    if ranks > width:
        raise ConfigurationError("pool ranks exceed vocabulary entries")
    base, extra = divmod(width, ranks)
    return tuple(base + int(rank < extra) for rank in range(ranks))


def embedding(program, parents, chips, model, rows):
    embedded = []
    h = model.hidden_size
    for chip in chips:
        lookup = program.compute(
            parents,
            chip,
            "elementwise",
            dict(
                op_name=(
                    f"cast_{model.dtype}_to_bf16"
                    if model.dtype != "bf16"
                    else "embedding_lookup"
                ),
                dtype="bf16",
                num_tokens=rows,
                hidden_size=h,
            ),
        )
        embedded.append(
            program.compute(
                (lookup,),
                chip,
                "elementwise",
                dict(
                    op_name="other_residual",
                    dtype="bf16",
                    num_tokens=rows,
                    hidden_size=h,
                ),
            )
        )
    return program.reduce(embedded, chips, rows * h * 2)


def logits(program, parents, chips, model, rows):
    selected = []
    h = model.hidden_size
    for chip, columns in zip(chips, column_shards(model.vocab_size, len(chips))):
        norm = program.compute(
            parents,
            chip,
            "elementwise",
            dict(op_name="rmsnorm", dtype="bf16", num_tokens=rows, hidden_size=h),
        )
        projected = program.compute(
            (norm,), chip, "gemm", dict(dtype=model.dtype, m=rows, n=columns, k=h)
        )
        capped = program.compute(
            (projected,),
            chip,
            "elementwise",
            dict(
                op_name="other_residual",
                dtype="bf16",
                num_tokens=rows,
                hidden_size=columns,
                lpx_vector_operation="logit_softcap",
            ),
        )
        selected.append(
            program.compute(
                (capped,),
                chip,
                "elementwise",
                dict(
                    op_name="argmax", dtype="bf16", num_tokens=rows, hidden_size=columns
                ),
            )
        )
    return program.reduce(selected, chips, rows * 16, element_bytes=4)
