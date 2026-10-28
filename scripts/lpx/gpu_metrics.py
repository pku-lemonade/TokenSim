"""Validate vLLM speculative counters and streamed decode measurements."""

import json
import math
import re

from TokenSim.errors import WorkloadValidationError
from scripts.measure_lpx_lookup_acceptance import equivalent_acceptance

COUNTERS = {
    "spec_decode_num_drafts": "drafts",
    "spec_decode_num_draft_tokens": "draft_tokens",
    "spec_decode_num_accepted_tokens": "accepted_tokens",
    "spec_decode_num_accepted_tokens_per_pos": "accepted_per_position",
}
SAMPLE = re.compile(r"^vllm:(\w+)(?:\{(.*?)\})?\s+([^\s]+)")
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def counters(text, model):
    """Sum engine series for one served model, preserving position labels."""
    result = {}
    for line in text.splitlines():
        match = SAMPLE.match(line)
        if not match:
            continue
        name, labels, value = match.groups()
        name = name.removesuffix("_total")
        if name not in COUNTERS:
            continue
        labels = {
            key: json.loads('"' + value + '"')
            for key, value in LABEL.findall(labels or "")
        }
        if labels.get("model_name") != model:
            continue
        value = float(value)
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise WorkloadValidationError(
                "Speculative counters must be nonnegative integers"
            )
        field = COUNTERS[name]
        if field == "accepted_per_position":
            position = labels.get("position")
            if position is None:
                raise WorkloadValidationError("Acceptance series has no position label")
            positions = result.setdefault(field, {})
            positions[position] = positions.get(position, 0) + int(value)
        else:
            result[field] = result.get(field, 0) + int(value)
    return result


def counter_delta(before, after, speculative):
    """Reject missing counters and server resets instead of implying zero acceptance."""
    result = {}
    for field in COUNTERS.values():
        if speculative and (field not in before or field not in after):
            raise WorkloadValidationError(f"Missing speculative counter: {field}")
        if field == "accepted_per_position":
            old, new = before.get(field, {}), after.get(field, {})
            if old.keys() - new.keys():
                raise WorkloadValidationError("Acceptance position series disappeared")
            result[field] = {key: value - old.get(key, 0) for key, value in new.items()}
            values = result[field].values()
        else:
            result[field] = after.get(field, 0) - before.get(field, 0)
            values = [result[field]]
        if any(value < 0 for value in values):
            raise WorkloadValidationError("Speculative counter reset during a turn")
    return result


def decode_timing(completion_tokens, arrivals):
    """Time generated-token chunks, excluding events without generated tokens."""
    if completion_tokens < 1 or not arrivals:
        raise WorkloadValidationError(
            "A measurement needs output tokens and text chunks"
        )
    if any(not math.isfinite(value) or value < 0 for value in arrivals):
        raise WorkloadValidationError("Invalid chunk time")
    if any(later < earlier for earlier, later in zip(arrivals, arrivals[1:])):
        raise WorkloadValidationError("Chunk times must be monotonic")
    decode = arrivals[-1] - arrivals[0]
    return dict(
        ttft_s=arrivals[0],
        decode_s=decode,
        output_tps=(completion_tokens - 1) / decode if decode > 0 else None,
    )


def acceptance(turn, draft_length, fixed_verification=True):
    """Derive tau without silently clipping invalid counters to an acceptance."""
    drafts = turn["drafts"]
    accepted = turn["accepted_tokens"]
    proposed = turn["draft_tokens"]
    tokens = turn["completion_tokens"] - 1
    if draft_length < 1 or drafts < 1 or tokens < 1:
        raise WorkloadValidationError(
            "Acceptance needs a positive K, drafts and decode tokens"
        )
    if not 0 <= accepted <= proposed <= drafts * draft_length:
        raise WorkloadValidationError("Inconsistent accepted/proposed token counts")
    positions = turn["accepted_per_position"]
    if set(positions) != {str(position) for position in range(draft_length)}:
        raise WorkloadValidationError(
            "Expected one acceptance counter per draft position"
        )
    if sum(positions.values()) != accepted:
        raise WorkloadValidationError("Position counters do not sum to accepted tokens")
    if any(value < 0 or value > drafts for value in positions.values()):
        raise WorkloadValidationError("Invalid per-position acceptance count")
    tau = tokens / drafts
    if not 1 <= tau <= draft_length + 1:
        raise WorkloadValidationError("Committed tokens per step are outside [1, K+1]")
    result = dict(
        decode_tokens=tokens,
        tau=tau,
        equivalent_acceptance=equivalent_acceptance(tau, draft_length),
        counter_identity_residual=tokens - drafts - accepted,
        unproposed_tokens=drafts * draft_length - proposed,
        per_position_acceptance={
            key: value / drafts for key, value in positions.items()
        },
    )
    if fixed_verification:
        result["per_row"] = [tokens, [0] * draft_length + [drafts]]
    return result
