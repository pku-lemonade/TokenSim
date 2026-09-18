# AF batch timing and fixed GPU cost

Decode Attention now performs one lookup for the entire local batch. Uniform
contexts use `generation_attention` with the actual `batch_size`; mixed contexts
use `generation_attention_batch`, whose `context_lengths` is an ordered,
comma-separated vector, for example `"32,40,48"`. Window limits apply to every
context before lookup. The analytical candidate counts the sum of actual KV
lengths, rather than treating all requests as the longest one. Ragged shapes
require exact data; they do not interpolate encoded vectors.

Formal decode defaults to `af.routed_expert_execution: grouped`. Independent
routed expert nodes in the same block/layer, physical worker, domain rank and
TP/EP group share one compute execution. This group waits for all input
dependencies and acquires the GPU resource once. Routing assignments, weights,
output tensor transfers and per-expert TP collectives retain their semantics.
Different GPUs, layers, blocks and incompatible groups cannot merge. A backend
that actually launches each expert separately can select `per_expert`.

The provider queries `moe_group` once for multiple experts. Its `num_tokens`
counts routed assignments, `top_k` is therefore 1, and `expert_token_counts`
preserves the load of each active expert in expert-ID order. The boundary is
`unweighted_partials`: assignment preparation, projections and activation may
be included, but router/TopK, routing-weight multiplication and final summation
must not be included. In particular, do not use the complete `fused_experts`
latency here and then add weighted combine again. Single experts retain the
existing `moe` lookup.

## Fixed cost and measured observations

For analytical candidates, latency is
`offset_us + multiplier * max(compute_time_us, memory_time_us)`.
The offset belongs to one actual operator/batch/group, not to each logical
request or expert. A compound expert group can contain several kernel launches;
its offset must be calibrated for that whole boundary. This is GPU execution
cost, not a model-load cold start or CPU scheduling delay.

Measured whole-execution observations have precedence and are used unchanged.
They already include their fixed cost. Exact lookup and interpolation never add
another offset. Analysis files identify the analytical candidate separately and
record whether calibration applies to the selected timing. Traces preserve
`timing_method`, `calibration_applied` and `fixed_overhead_us` when available.

The generator and CLI accept optional `shape_calibrations` in addition to
table-wide `calibrations`. For example, the following **illustrative values must
be replaced by measurements**:

```yaml
shape_calibrations:
  - table_name: gemm
    key_match:
      device: a800
      backend: analytical
      weight_dtype: bf16
      activation_dtype: bf16
      n: 512
      k: 256
    calibration:
      multiplier: 1.0
      offset_us: 3.0
      calibration_id: replace-with-measurement-id
```

The most specific matching rule wins; equally specific matches are an error.
Use canonical schema values in `key_match`. Multipliers must be finite and
positive, offsets finite and nonnegative. Hardware and calibration parameters
are included in the generation input hash. There is no universal default GPU
offset: an old uncalibrated package remains uncalibrated until regenerated.

## Existing packages and limits

The two batch tables are optional schema-v2 extensions. Six-table packages
still load. If execution needs a missing batch/shape, lookup fails with an
explicit request to regenerate data; it never falls back to summing singleton
latencies. Regenerate package shapes when enabling grouping or increasing the
batch size. Existing GEMM/context interpolation remains bounded and all other
discrete shape fields must match.

This change addresses decode batch timing, grouped expert execution and fixed
GPU cost. Prefill Attention retains its existing equal-length approximation.
Router/combine placement, CPU scheduling, missing model operators, communication
costs and TP collective fusion are separate modeling questions. A measured tiny
model package is not evidence of accuracy for other models or hardware.
