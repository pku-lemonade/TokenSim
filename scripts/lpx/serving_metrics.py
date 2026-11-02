"""Exact useful matrix work for prompt processing and autoregressive outputs."""

import numpy as np

from TokenSim.config.validation import integer
from scripts.lpx.mfu_metrics import useful_matrix_flops


def latency_metrics(tokens, requests):
    """Measure prompt delay and generated-token gaps for completed requests."""
    previous = {}
    intervals = []
    for token in tokens:
        request = token["client"], token["cycle"]
        if request in previous:
            intervals.append(token["time"] - previous[request])
        previous[request] = token["time"]
    ttft = [row["first_output"] - row["start"] for row in requests]
    return dict(
        ttft_mean_s=float(np.mean(ttft)),
        ttft_p99_s=float(np.percentile(ttft, 99)),
        tpot_mean_s=float(np.mean(intervals)) if intervals else None,
        tpot_p50_s=float(np.median(intervals)) if intervals else None,
        tpot_p99_s=float(np.percentile(intervals, 99)) if intervals else None,
        maximum_inter_token_s=max(intervals, default=None),
        measured_inter_token_intervals=len(intervals),
    )


def key_pairs(start, stop, window=0):
    """Sum min(position, window) on the inclusive, one-based position range."""
    integer(start, "attention position start", minimum=1)
    integer(stop, "attention position stop", minimum=start)
    integer(window, "attention window")

    def prefix(end):
        visible = min(end, window) if window else end
        return visible * (visible + 1) // 2 + (end - visible) * visible

    return prefix(stop) - prefix(start - 1)


def forward_work(model, *, offset, queries, logits_rows=0):
    """Count each causal pair once, independently of the chunk partition."""
    integer(offset, "forward prefix tokens")
    integer(queries, "forward query tokens", minimum=1)
    integer(logits_rows, "forward logits rows")
    if logits_rows > queries:
        from TokenSim.errors import ConfigurationError

        raise ConfigurationError("logits rows exceed the computed hidden rows")
    head = 2 * model.hidden_size * model.vocab_size
    linear = queries * (useful_matrix_flops(model, 1)["linear"] - head)
    logits = logits_rows * head
    attention = sum(
        4
        * model.num_attention_heads
        * model.attention_geometry(i).head_dim
        * key_pairs(offset + 1, offset + queries, model.attention_window(i) or 0)
        for i in range(model.num_layers)
    )
    return dict(
        transformer_linear=linear,
        logits=logits,
        attention=attention,
        total=linear + logits + attention,
    )


def serving_work(model, *, inputs, outputs):
    integer(inputs, "serving input tokens", minimum=1)
    integer(outputs, "serving output tokens", minimum=1)
    prompt = forward_work(model, offset=0, queries=inputs, logits_rows=1)
    decode = (
        forward_work(model, offset=inputs, queries=outputs - 1, logits_rows=outputs - 1)
        if outputs > 1
        else dict(transformer_linear=0, logits=0, attention=0, total=0)
    )
    return dict(
        prefill=prompt,
        decode=decode,
        total=prompt["total"] + decode["total"],
        prefill_tokens=inputs,
        decode_forwards=outputs - 1,
        output_tokens=outputs,
        logits_rows=outputs,
        scope="Useful dense matrix FLOPs, exact causal pairs and one prompt-terminal logits row. Scalar work and replicated or padded operations receive no useful credit.",
    )
