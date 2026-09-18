# Operator Latency Model (operator_table backend)

This document describes the latency estimation architecture used by TokenSim.
It replaces the legacy `TransformerRoofline` system (a single TFLOPS/bandwidth
table plus V100 all-reduce spreadsheet plus three global calibration
constants). The design goal is to cover A100, H100/H200, H20, RTX 4090,
GB300, and Groq TSP within a single codebase, with the ability to extend to
multi-node, multi-rack communication topologies.

---

## 1. Architecture Overview: Three-Layer Structure

The system is organized into three layers: **data**, **lookup**, and
**latency backend**. Data flows upward from static files through a controlled
interpolation engine into the per-step latency compositor.

```text
data/devices/*.yaml        Device specs (peak compute / memory / on-chip storage / interconnect ports),
                           each value carrying a source_id and evidence grade
data/topologies/links.yaml Link classes (per-direction bandwidth, hop latency, message-size efficiency
                           curve, collective efficiency)
data/topologies/*.yaml     Hierarchical topologies (chip -> node -> rack -> cluster), each level
                           recording only the fan-in count and link class
data/models/*.yaml         Model architectures (HF config.json fields): dense / GQA / MoE
data/operator_data/<device>/<backend>/
                           Operator latency tables (parquet) + generation_meta.yaml + shape_manifests/
        |
        v
TokenSim/hardware          Directory loading, alias resolution, topology coordinates, and group layouts
                           (HardwareContext)
TokenSim/operator_data     Table schemas, package validation, exact match / controlled interpolation /
                           bounded extrapolation, analytical models, generate/import/calibrate CLI
TokenSim/comm              Hierarchical alpha-beta collective communication model (measured table
                           override)
TokenSim/latency           OperatorTableLatencyBackend: per-rank local-shape operator composition,
                           outputting step time and provenance statistics
```

At runtime, no roofline is recomputed from scratch. The query resolution order
is fixed:

> **Exact match** --> **Interpolation within the same discrete key** --> **Bounded extrapolation** (default max 16x; the boundary row's measured efficiency is kept and the analytical model supplies the growth, AIConfigurator's rule) --> **Analytical model for the device family**

Every step records a `match_type` (`exact` / `interpolated` / `extrapolated` /
`analytical`). The result JSON aggregates these counts per table, and any
missing shapes are exported to `missing_shapes_<qps>.json` as the collection
manifest for the next profiling round.

---

## 2. Operator Tables

### 2.1 Common Conventions

All tables share two value columns: `latency_us` (execution time in
microseconds) and `source_id` (a string that must appear in the package's
`generation_meta.yaml` `sources` list, with an associated evidence grade and
acquisition method). The `source_id` enables full provenance tracking from any
simulation result back to the measurement or formula that produced a given
latency number.

Evidence grades and acquisition methods are described in section 7.

### 2.2 Table Schemas

There are seven operator tables. Each table has a set of **key fields** that
uniquely identify a row, a subset of those designated as **axes** (numeric
fields along which interpolation is permitted), and the remainder acting as
**discrete fields** (which must match exactly).

| Table | Key Fields | Interpolation Axes (outer to inner) | Description |
| --- | --- | --- | --- |
| `gemm` | `dtype, m, n, k` | `k` (log), `n` (log), `m` (log) | `C[m,n] = A[m,k] x B[k,n]`; `dtype` is the weight/storage precision |
| `context_attention` | `attn_dtype, kv_cache_dtype, batch_size, input_seq_len, num_heads, num_kv_heads, head_dim, window_size` | `num_heads` (linear), `batch_size` (log), `input_seq_len` (log) | Prefill attention core (QK^T, softmax, PV); batch sequences padded to same length |
| `generation_attention` | `attn_dtype, kv_cache_dtype, batch_size, context_len, num_heads, num_kv_heads, head_dim, window_size` | `num_heads` (linear), `batch_size` (log), `context_len` (log) | Decode attention core; `num_heads`/`num_kv_heads` are TP-local head counts |
| `moe` | `dtype, distribution, num_tokens, hidden_size, inter_size, top_k, num_experts, tp_size, ep_size` | `num_tokens` (log) | Fused/grouped expert FFN on one rank; `num_tokens` is pre-top-k-expansion count |
| `elementwise` | `op_name, dtype, num_tokens, hidden_size` | `hidden_size` (log), `num_tokens` (log) | Memory-bound pointwise kernels |
| `collective` | `dtype, operation, group_size, nodes, message_bytes` | `message_bytes` (log) | nccl-tests convention: full buffer, one out-of-place iteration |
| `ep_all2all` | `dtype, phase, mode, ep_size, nodes, hidden_size, top_k, num_experts, num_tokens` | `num_experts` (log, bound 1024x), `top_k` (linear, bound 8x), `hidden_size` (log), `num_tokens` (log) | Measured DeepEP dispatch or combine for one rank sending `num_tokens` local tokens; `mode` is `deepep_high_throughput`, `deepep_low_latency`, or a framework-specific variant |

#### 2.2.1 `gemm` Table

Records the wall-clock time for a matrix multiplication `C[m,n] = A[m,k] x B[k,n]`.
The `dtype` field specifies the weight/storage precision and is a discrete key
(must match exactly). Supported values include:

- `fp16`, `bf16` -- native half-precision tensor-core execution
- `fp8`, `fp8_block` -- FP8 formats (e.g., E4M3/E5M2 on Hopper and later)
- `int8`, `int8_wo` -- 8-bit integer; the `_wo` suffix denotes weight-only
  quantization where the weight is stored in INT8 but activations and
  accumulation run in FP16
- `int4_wo`, `nvfp4` -- 4-bit weight-only formats

For weight-only precisions (`int8_wo`, `int4_wo`, `nvfp4`), the FLOPs are
counted at the activation precision (FP16) because the tensor cores execute in
the wider format; only memory traffic benefits from the narrower storage.

The three interpolation axes are resolved from outer to inner in the order
`k -> n -> m`. All three use logarithmic (geometric) blending, which matches
the roughly log-linear scaling of GEMM latency with matrix dimension.

#### 2.2.2 `context_attention` Table (Prefill)

Records the combined time for the attention core during prefill: the QK^T
product, softmax, and the PV product. All sequences in the batch are padded to
`input_seq_len`. The `num_heads` and `num_kv_heads` fields reflect the
TP-local head counts after tensor-parallel sharding.

Interpolation axes: `num_heads` (linear blending, because attention time
scales linearly with head count), `batch_size` (log), `input_seq_len` (log).

The `window_size` field supports sliding-window attention (e.g., Mistral
models). A value of 0 means full causal attention.

#### 2.2.3 `generation_attention` Table (Decode)

Similar to `context_attention` but for the decode phase: each request
generates one token and attends to `context_len` cached KV entries. The
backend groups decode requests into context-length buckets (see section 4) to
reduce the number of distinct lookups.

#### 2.2.4 `moe` Table

Records the execution time of a fused or grouped expert FFN on a single rank.
Key details:

- `num_tokens` is the total token count entering the MoE layer **before**
  top-k expansion, for the entire EP group
- `distribution` is a discrete field describing the expert-load distribution
  used during profiling (`uniform`, `balanced`, `power_law_1.01`,
  `power_law_1.2`, etc.)
- `tp_size` and `ep_size` record the parallelism config under which the row
  was measured
- The row gives the time for one rank holding `num_experts / ep_size` experts,
  each sharded `tp_size` ways

The only interpolation axis is `num_tokens` (log). When querying, the backend
tries multiple distribution candidates in order (e.g., for a "uniform"
routing, it tries `uniform`, `balanced`, then `power_law_1.01`) until a match
is found.

#### 2.2.5 `elementwise` Table

Covers memory-bound pointwise kernels. The `op_name` field is discrete and
must be one of:

| op_name | Description |
| --- | --- |
| `rmsnorm` | RMS normalization |
| `layernorm` | Layer normalization |
| `rope` | Rotary positional embedding |
| `residual_add` | Residual connection addition |
| `swiglu` | SwiGLU activation (gated) |
| `gelu` | GELU activation |
| `data_reorder` | Token permutation / data layout transform |
| `other_residual` | Other residual-like operations |

Interpolation axes: `hidden_size` (log), `num_tokens` (log).

#### 2.2.6 `collective` Table

Records measured collective communication times from nccl-tests or equivalent
benchmarks. Following the nccl-tests convention, `message_bytes` is the full
buffer size and the latency represents one out-of-place iteration for
`group_size` ranks spread over `nodes` nodes.

Supported operations: `all_reduce`, `all_gather`, `reduce_scatter`,
`all_to_all`, `send_recv`.

The only interpolation axis is `message_bytes` (log). The analytical
communication model (section 6) handles unmeasured configurations.

### 2.3 Package Directory Layout

```text
data/operator_data/<device_id>/<backend>/
  generation_meta.yaml            Dataset version, device_id, backend, sources list, calibration
                                  parameters, row counts
  gemm_perf.parquet               Runtime narrow tables (one per operator type)
  context_attention_perf.parquet
  generation_attention_perf.parquet
  moe_perf.parquet
  elementwise_perf.parquet
  collective_perf.parquet
  gemm_analysis.parquet           Optional: FLOPs, bytes, compute/memory time, bottleneck,
  ...                             calibration id (for diagnostic and calibration use)
  shape_manifests/<exp>.yaml      Shape manifest used to generate this package
```

The `backend` is the name of the data source stack: `trtllm`, `vllm`,
`sglang`, `nccl`, `analytical`, `measured`, `cuda`, `groq`. A single device
may have multiple backends. The cluster configuration's
`worker_groups[].operator_backend` or the CLI `--operator_backend` flag
selects which backend to use. The default priority order is:

> `trtllm` > `vllm` > `sglang` > `measured` > `cuda` > `groq` > `analytical`

---

## 3. Lookup Strategy

The lookup engine (`TokenSim/operator_data/lookup.py`) implements a
four-stage resolution pipeline for every operator query.

### 3.1 Stage 1: Exact Match

The query key is normalized (dtype aliases resolved, strings lowercased,
float-valued integers cast to int) and looked up in a hash map keyed by the
full composite key. If found, the result is returned immediately with
`match_type = "exact"`.

### 3.2 Stage 2: Multi-Axis Recursive Interpolation

When no exact match exists but the discrete fields (dtype, op_name, etc.)
match at least one row, the system performs **controlled multi-axis
interpolation**. The algorithm works as follows:

1. **Discrete key isolation**: All rows sharing the same discrete key values
   form a (possibly ragged) grid indexed by the axis values.

2. **Recursive outer-first resolution**: The axes declared in the `TableSpec`
   are resolved from **outer to inner**, in the order they appear in the
   `axes` tuple. For a GEMM query, this means `k` is resolved first, then
   `n`, then `m`.

3. **Per-axis blending**: At each axis level, the algorithm finds the two
   nearest measured values bracketing the query target:
   - **Logarithmic blending** (scale = `"log"`): The interpolation weight is
     computed in log space:
     ```
     weight = (log(target) - log(lo)) / (log(hi) - log(lo))
     ```
     This is the default for size-like axes (matrix dimensions, token counts,
     message sizes) where performance scales geometrically.
   - **Linear blending** (scale = `"linear"`): The weight is computed in
     linear space:
     ```
     weight = (target - lo) / (hi - lo)
     ```
     This is used for `num_heads` because attention time scales linearly with
     head count.

4. **Result composition**: The interpolated latency is:
   ```
   latency = lat_lo + weight * (lat_hi - lat_lo)
   ```
   Source IDs from both bracketing rows are combined (joined with `+`). The
   `match_type` is `"interpolated"`.

5. **Multi-source tracking**: When interpolation touches rows from different
   measurement sources, all contributing `source_id` values are collected and
   joined into the result.

#### Why outer-first matters

Consider a GEMM query for `(m=2048, n=4096, k=8192)` with measured points at
`k` values of 4096 and 16384. The outer axis `k` is resolved first: the
engine interpolates between `k=4096` and `k=16384`, and at each of those `k`
values it recursively resolves the inner axes `n` and then `m`. This
structure means the interpolation respects the nested structure of the
measurement grid rather than flattening it.

### 3.3 Stage 3: Bounded Extrapolation

When the query lies outside the measured range on an interpolation axis, the
lookup starts from the nearest boundary row and applies the policy's
`extrapolate` mode (`LookupPolicy` in `TokenSim/operator_data/lookup.py`):

| Mode | Behaviour |
| --- | --- |
| `extrapolate = "analytical"` (**default**) | Keep the boundary row's measured *efficiency* and let the analytical model carry the growth: `latency = measured(boundary) x analytical(target) / analytical(boundary)`. This is the rule AIConfigurator applies past its collected range ("hold the boundary utilisation, let SOL carry the growth"). It needs an analytical scaler; the operator-table backend supplies its device-family model, the collective model supplies its alpha-beta formula. Without a usable scaler the mode degrades to `scale`. |
| `extrapolate = "scale"` | Multiply the boundary latency by `target / boundary` when the axis is logarithmic and the query exceeds the upper boundary (work grows with the axis); below the boundary the fixed cost dominates, so nothing is scaled. |
| `extrapolate = "hold"` | Return the boundary row's latency unchanged. |
| `extrapolate = "none"` | Raise `MissingOperatorDataError` immediately. |

Why `analytical` is the default: linear scaling is right for the GEMM `m`
axis but wrong for axes whose cost is not proportional to the value (decode
attention against `context_len` has a large fixed part; collective latency
against `message_bytes` follows an alpha-beta curve). Using the model's own
growth keeps the extrapolated curve shaped like the physics while the measured
point pins its level.

The **`max_extrapolation_ratio`** parameter (default: **16.0**) caps how far
extrapolation may go on any axis; an `AxisSpec` may override it for axes that
barely move the cost (the DeepEP `num_experts` axis allows 1024x, `top_k` 8x).
If `target / boundary` (or its reciprocal) exceeds the bound, the lookup raises
`MissingOperatorDataError` and the query falls through to Stage 4.

Results that touched extrapolation have `match_type = "extrapolated"`; the
`detail["flags"]` set additionally records `analytical_scaled`,
`linear_scaled` or `held` so a coverage report can tell the three apart.

### 3.4 Stage 4: Analytical Fallback

When all table-based approaches fail (no matching discrete key, extrapolation
ratio exceeded, or the table is empty), the system falls back to a
device-family-specific analytical model (section 5).

### 3.5 Fallback Policy Configuration

The `OperatorTableLatencyBackend` accepts a `fallback` parameter that
controls the overall resolution strategy:

| Fallback | Behavior |
| --- | --- |
| `table_first` (default) | Try table lookup; on failure, use analytical model |
| `table_only` | Table lookup only; raise an error on failure |
| `analytical_only` | Skip tables entirely; use analytical model for everything |

### 3.6 Result Caching

The backend maintains an LRU-like cache (up to 262,144 entries) keyed by
`(table_name, sorted_key_tuple)`. Cache entries are reused across steps,
which is important because decode steps with steady-state batch sizes repeat
the same GEMM and elementwise queries every step.

---

## 4. From Model Config to Operator Queries

The `OperatorTableLatencyBackend` (`TokenSim/latency/operator_table.py`)
decomposes each scheduler step into individual operator queries. It maps the
model architecture and parallel configuration to per-operator lookups with
TP-local dimensions.

### 4.1 Local Dimensions Under Tensor Parallelism

Given `TP = tensor_parallel_size`, the backend computes:

```text
heads_local   = ceil(num_attention_heads / TP)
kv_heads_local = local_kv_heads(num_kv_heads, TP)   # when kv_heads < TP, heads are replicated
q_dim_local   = heads_local * head_dim
kv_dim_local  = kv_heads_local * head_dim
inter_local   = ceil(intermediate_size / TP)
vocab_local   = ceil(vocab_size / TP)
layers_local  = layers assigned to this PP stage
moe_layers_local = MoE layer count from ExpertPlacement for this PP rank
dense_layers_local = layers_local - moe_layers_local
```

### 4.2 Prefill / Recompute Path

For a prefill step with `T` total tokens across `B` requests:

| Component | Operator Queries |
| --- | --- |
| **Attention block** (per layer) | `rmsnorm(T, h)` + `gemm(T, q_local + 2*kv_local, h)` + `rope(T, q_local + kv_local)` + attention core + `gemm(T, h, q_local)` + `residual_add(T, h)` |
| **Attention core** | Requests grouped by `(query_len, kv_len)`. Each group issues one `context_attention(batch=count, input_seq_len=query_len)`. Prefix-cache hits (`kv_len > query_len`) scale the measured latency by the analytical FLOPs ratio. |
| **Dense FFN** (per dense layer) | `rmsnorm(T, h)` + `gemm(T, g*inter_local, h)` + `swiglu(T, inter_local)` + `gemm(T, h, inter_local)` + `residual_add(T, h)` |
| **MoE layer** | `rmsnorm(T, h)` + router `gemm(T, E, h)` + `moe(num_tokens=T*DP_if_EP, tp, ep)` x imbalance + `residual_add(T, h)` |
| **lm_head** (last PP stage only) | `gemm(B, vocab_local, h)` + `rmsnorm(B, h)` |
| **Communication** | Per layer: 1 all-reduce after o_proj + 1 all-reduce after FFN for dense/pure-TP MoE layers (message: `T * h * bytes_per_activation`). EP MoE layers: dispatch + combine all-to-all each (message: `T * top_k * h * bytes_per_activation`). PP stages: one point-to-point transfer of `T * h * bytes_per_activation`. |
| **Fixed overhead** | `device.step_overhead_us` |

#### Prefix-Cache Scaling

When `kv_len > query_len` (prefix-cache hit), the table value for
`input_seq_len = query_len` underestimates the work because queries also
attend to the cached prefix. The backend computes the analytical FLOPs ratio
between the full-range attention and the query-only attention, then scales the
measured latency by `max(1.0, full_flops / base_flops)`.

### 4.3 Decode Path

For a decode step, `T = B` (one token per request). The key difference from
prefill is how attention is handled:

**Context-length bucketing**: Decode requests are grouped by context length
into buckets of `decode_context_bucket` (default 128) width, rounding up.
Each bucket issues a single `generation_attention(batch=count,
context_len=bucket_ceiling)` lookup. This avoids an explosion of unique
attention queries while keeping the approximation tight.

All other operators (GEMM, elementwise, communication) are identical to the
prefill path but with `T = B`.

### 4.4 MoE Straggler Modeling

For MoE layers, the step waits for the slowest expert rank. The backend:

1. Computes a routing histogram from the `ExpertRouting` module
2. Derives per-rank token load from the expert-to-rank mapping
3. Computes `imbalance = max_rank_load / mean_rank_load`
4. Queries the MoE table at balanced load (uniform token count)
5. Queries again at `tokens * imbalance` to estimate the busiest rank's time
6. The straggler cost is `(busiest_time - balanced_time) * moe_layers_local`

When expert parallelism crosses DP replicas, the system assumes all DP
replicas advance in lockstep, so the EP group sees
`tokens * data_parallel_size` input tokens. This is an explicit modeling
assumption.

### 4.5 Avoiding Double-Counting

The operator decomposition carefully avoids double-counting:

- QKV and output projections are **not** in the attention table; they are
  separate GEMM queries
- Softmax **is** included in the attention table
- SwiGLU is counted separately only for dense FFN layers; the MoE table
  already includes the activation function
- Dispatch/combine all-to-all latency is counted only in the communication
  component, not in the MoE operator time

---

## 5. Analytical Fallback

When table lookup fails, the system falls back to a roofline-style analytical
model registered per device family
(`TokenSim/operator_data/analytical/`).

### 5.1 Core Formula

```text
compute_time = FLOPs / (peak_ops(dtype) * efficiency)
memory_time  = total_bytes / (bandwidth * memory_efficiency)
roofline_us  = max(compute_time, memory_time) + kernel_launch_overhead_us
latency_us   = calibration.multiplier * roofline_us + calibration.offset_us
```

The `Calibration` dataclass provides an affine correction: `latency = m *
roofline + b`. The multiplier and offset are fitted from measured tables using
the `calibrate` CLI command (see section 8).

### 5.2 Device Family Models

#### `nvidia_gpu` / `generic`

For HBM/GDDR-based GPUs. Key characteristics:

| Parameter | Default (grade D) | Description |
| --- | --- | --- |
| `gemm_mfu` | 0.70 | Tensor-core utilization for GEMM |
| `attention_mfu` | 0.45 | Utilization for attention kernels |
| `moe_mfu` | 0.50 | Utilization for grouped expert FFN |
| `memory_efficiency` | 0.85 | Fraction of peak HBM bandwidth achieved |
| `elementwise_memory_efficiency` | 0.70 | Memory efficiency for pointwise kernels |
| `kernel_launch_us` | 4.0 | Software overhead per kernel launch |
| `gemm_small_m_knee` | 64.0 | Below this `m`, tensor-core tiles waste work; FLOPs are computed as if `m = knee` |

**Small-m GEMM adjustment**: When `m` (token count) is below the
`gemm_small_m_knee`, tensor-core tiles process at least `knee` rows per pass,
wasting compute on padding. The analytical model accounts for this by
computing effective FLOPs as `2 * max(m, knee) * n * k`.

**MoE overhead**: Grouped GEMM kernels issue gate/up and down kernels plus a
permutation operation, so MoE overhead is set to 3x the base kernel launch
cost.

**Elementwise kernels**: These are memory-bound by definition. The analytical
model uses FP32 peak compute (if available) since these kernels typically
execute in FP32 regardless of the model's weight precision.

#### `groq_tsp`

For the Groq Tensor Streaming Processor (LPU). Key differences from GPU:

| Parameter | Default (grade D) | Description |
| --- | --- | --- |
| `gemm_mfu` | 0.80 | Higher utilization due to deterministic scheduling |
| `attention_mfu` | 0.60 | |
| `moe_mfu` | 0.70 | |
| `memory_efficiency` | 0.90 | SRAM bandwidth is more predictable than HBM |
| `kernel_launch_us` | 0.2 | Software-scheduled, no GPU-style launch overhead |

Weights, activations, and KV cache all reside in on-chip SRAM, so the memory
term uses the SRAM bandwidth (typically ~80 TB/s) rather than HBM bandwidth.
The model also reports whether the rank-local weights fit in SRAM; the
generator turns this into an explicit `fits_on_chip` flag in the analysis
table.

#### Device-level overrides and their provenance

A device YAML can override any analytical parameter in its `analytical:`
block and must then declare where the numbers come from:

```yaml
analytical:
  source_id: aiconfigurator-system-yaml   # declared in sources:
  grade: C
  memory_efficiency: 0.8        # AIConfigurator mem_bw_empirical_scaling_factor
  kernel_launch_us: 3.0         # mem_empirical_constant_latency
  collective_launch_us: 10.0    # node.p2p_latency
memory:
  reserved_bytes: {value: 4169138176, source_id: aiconfigurator-system-yaml, grade: C}
```

All NVIDIA GPU files carry these AIConfigurator-derived values as grade C
defaults. `memory.reserved_bytes` (NCCL buffers plus framework workspace) is
subtracted by `CacheConfig` before KV-cache blocks are computed, and
`collective_launch_us` replaces the family default launch cost in the
communication model. `DeviceSpec.analytical_grade` / `analytical_source_id`
expose the provenance to results.

### 5.3 FLOPs and Bytes Formulas

All work accounting is centralized in `TokenSim/operator_data/workload.py`.
The formulas intentionally match the task specification document:

**GEMM**: `FLOPs = 2 * m * n * k`. Weight bytes = `k * n * dtype_bytes`.
Activation bytes = `(m*k + m*n) * activation_dtype_bytes`.

**Prefill attention**: `FLOPs = 4 * B * S * keys_per_query * h_q *
causal_factor`. Causal masking halves the effective key range (`causal_factor
= 0.5`). Sliding window truncates `keys_per_query` to `min(S, window_size)`.
GQA uses actual `h_kv` for KV bytes rather than `h_q`.

**Decode attention**: `FLOPs = 4 * B * keys * h_q` where `keys =
min(context_len, window_size)`.

**MoE**: Only active experts contribute FLOPs. For a gated FFN:
`FLOPs = local_pairs * 2 * hidden * local_inter * 3` (gate + up + down).
`local_pairs = (T * top_k / ep_size) * imbalance`.

**Elementwise**: FLOPs and bytes per element are tabulated per operation:

| op_name | FLOPs/element | Bytes/element (reads+writes) |
| --- | --- | --- |
| `rmsnorm` | 4 | 2 + weight vector |
| `layernorm` | 5 | 2 + weight vector |
| `rope` | 3 | 2 |
| `residual_add` | 1 | 3 |
| `swiglu` | 4 | 3 |
| `gelu` | 6 | 2 |
| `data_reorder` | 0 | 2 |
| `other_residual` | 1 | 2 |

### 5.4 Calibration

The analytical model's default parameters are grade D (uncalibrated
assumptions). The `calibrate` CLI command fits `(multiplier, offset_us)` per
table against measured data from an operator package:

```bash
python -m TokenSim.operator_data.cli calibrate \
    --device a100_sxm_80g --backend trtllm \
    --out tmp/calib_a100.yaml
```

First-fit results on A100 TRT-LLM data (20% holdout):

| Table | Before (mean relative error) | After |
| --- | --- | --- |
| GEMM | 0.33 | 0.32 |
| Prefill attention | 0.41 | 0.17 |
| Decode attention | 0.58 | 0.42 |
| MoE | 0.46 | 0.36 |

The analytical model is only suitable for filling in order-of-magnitude
estimates. Shape-sensitive kernel behavior (e.g., tile selection, wave
quantization) still requires measured tables.

---

## 6. Communication Model

The hierarchical alpha-beta communication model lives in
`TokenSim/comm/collectives.py`.

### 6.1 Core Formula

Given a group of ranks laid out across a topology, the model computes per
topology level `l` (innermost first):

```text
steps_l       = algorithm-dependent count of sequential transfer rounds
bytes_l       = total payload each endpoint injects at this level
t_l           = steps_l * alpha_l + bytes_l / (beta_l * links_l * c_l * eff(M_l))
t_collective  = launch_us + sum_l(t_l)
```

Where:
- `alpha_l` = hop latency of the level's link class
- `beta_l` = per-direction bandwidth of the link class
- `links_l` = number of parallel links an endpoint can drive at this level
- `c_l` = `LinkClass.collective_efficiency` (fraction of peak bus bandwidth a
  collective achieves at large messages)
- `eff(M) = M / (M + half_bandwidth_bytes)` = message-size ramp fitted from
  nccl-tests curves; `half_bandwidth_bytes` is the message size at which the
  collective reaches half its peak bandwidth
- `launch_us` = software launch/synchronization cost (default: 8.0 us for
  NVIDIA GPUs, 0.5 us for Groq TSP, 10.0 us for generic)

### 6.2 Algorithms

| Algorithm | Rounds (all-reduce) | Injected bytes (all-reduce) | When used |
| --- | --- | --- | --- |
| `ring` | `2(n-1)` | `2(n-1)/n * M` | Bandwidth-optimal for large messages |
| `tree` | `2 * log2(n)` | `2 * M` | Latency-optimal for small messages |
| `direct` | `2` | `2(n-1)/n * M` | One-shot over non-blocking fabric (NVSwitch multicast, Groq software-scheduled network) |
| `auto` | -- | -- | Picks faster of ring and tree per level (matches NCCL tuner behavior) |

For other operations:

- **all_gather / reduce_scatter**: Ring uses `n-1` rounds with `(n-1)/n * M`
  bytes; tree uses `log2(n)` rounds with `M` bytes; direct uses 1 round.
- **all_to_all**: Ring uses `n-1` rounds; direct uses 1 round. Both inject
  `(n-1)/n * M` bytes.
- **send_recv**: Always 1 round, `M` bytes.
- **broadcast**: Tree uses `log2(n)` rounds; ring uses `n-1` rounds. Both
  inject `M` bytes.

### 6.3 Hierarchical Decomposition

For multi-level topologies, the model decomposes the collective operation
across topology levels:

1. **Reduce-scatter** inward at each level
2. When crossing to an outer level, only `1/fan_inner` of the data needs
   processing (the inner level has already reduced its portion)
3. **All-gather** outward to distribute the result

This matches NCCL's hierarchical / 2D algorithm. A 10,000-device group
requires only 3 levels of computation (e.g., 8 GPUs per node, 128 nodes per
rack, ~10 racks).

The `remaining_share` variable tracks the data fraction: after resolving a
level with `fan` participants, the share is divided by `fan` for the next
outer level.

### 6.4 Measured Table Override

When the device's operator package contains a `collective` table and the
query's `(dtype, operation, group_size, nodes)` matches, the model uses the
measured curve directly (via the same interpolation engine from section 3) and
tags the result as `match_type = "measured"`. This avoids the formula
entirely.

Measured collective curves for A100, H100, H200, and GB300 (NCCL and custom
all-reduce) are imported from AIConfigurator. V100 data comes from legacy
spreadsheets.

### 6.4.1 Expert-parallel dispatch and combine

MoE layers under expert parallelism issue one dispatch and one combine per
layer. `CollectiveModel.ep_all2all()` prices the pair from the `ep_all2all`
table when the configured `all2all_backend` is a DeepEP mode
(`deepep_high_throughput` / `deepep_low_latency`; the other DeepEP mode is
tried when the requested one has no rows). The query key is this rank's token
count, the model's hidden size, `top_k`, expert count, EP group size and the
number of nodes the group spans; expert count and `top_k` are interpolation
axes because DeepEP cost is driven by `num_tokens x top_k x hidden_size`.
Without a measured row, or for `naive` / `allgather_reducescatter`, the pair
is two analytical all-to-all collectives scaled by
`EP_ALL2ALL_MODE_SCALE` (naive 1.5, allgather_reducescatter 1.0,
deepep_high_throughput 0.7, deepep_low_latency 0.5; grade D, inherited from
the legacy backend). Results record the outcome in
`parallel_comm_match_type_counts`.

### 6.5 Validation Against Measurements

| Configuration | Message size | Measured | Model |
| --- | --- | --- | --- |
| A100 8-GPU NCCL | 1 MB all-reduce | 54 us | 90 us |
| A100 8-GPU NCCL | 64 MB all-reduce | 1169 us | 946 us |
| H100 8-GPU NCCL | 64 MB all-reduce | 578 us | 577 us |

The formula is conservative at small message sizes (launch overhead
dominates). Using measured table override covers these cases.

### 6.6 Scope and Non-Scope

**In scope**: Service time for TP, PP, and EP collectives computed per step as
non-contended transfer time.

**Not in scope**: Link contention (multiple micro-batches competing for the
same physical link, e.g., A-to-F interconnect sharing). Contention modeling
belongs to the executor layer, which uses SimPy resources for explicitly
declared shared links. The current mainline treats all TP/PP/EP communication
as uncontended service time within a step.

---

## 7. Data Sources and Evidence Grades

### 7.1 Evidence Grade System

Every data point in an operator package carries a `source_id` that maps to a
`SourceRecord` in the package's `generation_meta.yaml`. Each source has:

| Field | Description |
| --- | --- |
| `source_id` | Unique identifier (e.g., `aic_h100_trtllm_gemm_v3`) |
| `grade` | Evidence quality: A, B, C, or D |
| `method` | How the data was obtained |
| `reference` | URL, paper, or internal document reference |
| `notes` | Free-form annotation |
| `device` | Device the data was measured on |
| `backend` | Software stack |

**Grade definitions**:

| Grade | Meaning |
| --- | --- |
| A | Direct measurement on the target device and software stack |
| B | Imported from a trusted third-party benchmark or adjacent device |
| C | Derived from aggregate specifications (e.g., per-GPU peak inferred from system total) |
| D | Analytical model output or engineering assumption |

**Acquisition methods**: `measured`, `imported`, `public_benchmark`,
`analytical`, `assumption`.

### 7.2 AIConfigurator Imports

Two source layouts are accepted by `import-aiconfigurator`: the upstream
database (`<system>/<family>/<backend>/<version>/*.parquet`) and a flat
collector run directory (`*_perf.parquet` or `*_perf.txt` CSV staging files
written by `collector/collect.py` and `network/collect_comm.sh` on your own
node; see `scripts/collect/README.md`). Rows from a collector run are recorded
as `method: measured` with `source_id` prefix `aiconfigurator-collector:`.

Within one family the importer walks every version directory newest-first and
keeps the first row seen per key, so a newer partial re-collection (for
example the TensorRT-LLM 1.3.0rc23 MXFP4 MoE rows) is layered on top of the
previous full collection instead of replacing it. This mirrors AIConfigurator's
own backward-fill rule and is why the Hopper/Blackwell MoE tables hold 150k+
rows rather than the few thousand of the newest directory alone.

Imported systems and backends: `a100_sxm_80g`, `h100_sxm`, `h200_sxm`,
`l40s`, `rtx_pro_6000_server`, `b200_sxm`, `b300_sxm`, `gb200`, `gb300` with
`vllm`, `trtllm` and `sglang`; `intel_arc_pro_b60` with `vllm` (oneCCL
collectives). `vllm` is the default when a cluster names no
`operator_backend`; the precedence is vllm > trtllm > sglang > measured >
analytical. The per-device status table and the reasons versions and kernels
differ between systems live in `data/devices/README.md`.

The `import-aiconfigurator` CLI command ingests measured operator data from
AIConfigurator's system directories. This is the primary source of grade-A
data for NVIDIA GPUs. The import process:

1. Reads the AIConfigurator system directory structure
2. Maps measured GEMM, attention, and collective benchmarks to the
   TokenSim table schemas
3. Assigns appropriate `source_id` values and grade A
4. Writes (or merges into) the target operator package

---

## 8. CLI Tools

All commands are invoked through `python -m TokenSim.operator_data.cli`.

### 8.1 `manifest` -- Generate Shape Manifests

Lists all operator shapes needed for a given model and parallelism
configuration.

```bash
python -m TokenSim.operator_data.cli manifest \
    --models llama-3-70b \
    --tp 4,8 \
    --out tmp/llama70b.yaml
```

The manifest output is a YAML file listing every `(table, key)` pair the
simulation will query. This is the profiling target list: any shape not in the
existing package should be measured.

### 8.2 `generate` -- Generate Analytical Packages

Creates or updates an operator data package using the analytical model.

```bash
python -m TokenSim.operator_data.cli generate \
    --device rtx_4090 \
    --models llama-3-8b \
    --tp 1,2,4,8
```

Options:
- `--calibration <file>` -- YAML file with fitted calibration parameters
- `--batch`, `--prefill-tokens`, `--context` -- Override the default
  workload grid
- `--no-collectives` -- Skip collective shapes
- `--experiment-id` -- Custom name for the shape manifest

### 8.3 `import-aiconfigurator` -- Import Measured Data

Imports measured operator latencies from an AIConfigurator system directory.

```bash
python -m TokenSim.operator_data.cli import-aiconfigurator \
    --system-dir <aic>/systems/data/h100_sxm \
    --device h100_sxm
```

### 8.4 `import-nccl` -- Import NCCL-Tests Output

`scripts/collect/run_nccl_tests.sh` drives nccl-tests (single or multi-node)
and calls this command for every collective; `scripts/collect/run_aiconfigurator.sh`
covers intra-node curves through the AIConfigurator collector.

Parses nccl-tests stdout and imports collective timing data.

```bash
python -m TokenSim.operator_data.cli import-nccl \
    --device a100_sxm_80g \
    --file all_reduce_perf.txt \
    --operation all_reduce \
    --group-size 8
```

### 8.5 `validate` -- Validate a Package

Checks schema conformance, source_id references, and data integrity.

```bash
python -m TokenSim.operator_data.cli validate \
    data/operator_data/h100_sxm/trtllm
```

### 8.6 `coverage` -- Coverage Report

Reports which shapes in a manifest are covered by measured data versus
requiring interpolation, extrapolation, or analytical fallback.

```bash
python -m TokenSim.operator_data.cli coverage \
    --device h100_sxm \
    --models llama-3-70b \
    --tp 8 \
    --out tmp/coverage.yaml
```

### 8.7 `calibrate` -- Fit Analytical Parameters

Fits the analytical model's calibration parameters against measured data in
an existing package.

```bash
python -m TokenSim.operator_data.cli calibrate \
    --device h100_sxm \
    --backend trtllm
```

### 8.8 Simulation-Side Flags

`--operator_backend` (or `worker_groups[].operator_backend`) selects which
imported serving stack answers the queries; unset means `vllm` when the device
has it, otherwise the next available backend (`trtllm`, `sglang`, ...).

The benchmark runner accepts these latency-related flags:

```bash
./benchmark.py \
    --latency_backend operator_table|analytical \
    --latency_fallback table_first|table_only|analytical_only \
    --operator_backend trtllm \
    --decode_context_bucket 128
```

---

## 9. Result Provenance

Every simulation result (`result_<qps>.json`) includes detailed provenance
information:

| Field | Description |
| --- | --- |
| `latency_backends` | Per-device backend name, dataset version, and table row counts |
| `operator_match_type_counts` | Aggregate counts of `exact`, `interpolated`, `extrapolated`, `analytical` across all operator queries |
| `operator_table_match_counts` | Same counts broken down per table (gemm, context_attention, etc.) |
| `operator_component_seconds` | Cumulative wall-clock seconds attributed to each component: `gemm`, `attention`, `moe`, `elementwise`, `lm_head`, `comm`, `overhead` |
| `operator_missing_shape_count` | Number of unique shapes that could not be resolved from tables |
| `parallel_link_type_counts` | Communication event counts by link type |
| `parallel_comm_match_type_counts` | Communication match type breakdown (measured vs analytical) |

Missing shapes are exported separately to `missing_shapes_<qps>.json`. Each
entry includes the table name, the query key, and the reason for failure.
This file serves as the collection manifest for the next profiling round:
measure these shapes and add them to the package to improve coverage.

---

## 10. Known Limitations

- **Ragged batches**: Tables measure padding-aligned regular batches. Ragged
  (continuous) batches are approximated by grouping requests by `(query_len,
  kv_len)` and summing the per-group latencies.
- **Prefix-cache attention**: Prefill attention for prefix-cache hits (where
  `kv_len > query_len`) is estimated by scaling the measured padded-batch
  value using the analytical FLOPs ratio.
- **EP cross-DP lockstep assumption**: When expert parallelism spans data
  parallel replicas, all replicas are assumed to advance in lockstep, making
  the EP group's token count equal to `tokens * data_parallel_size`.
- **No link contention in step time**: Link contention and queuing effects
  from pipelined micro-batches (e.g., A/F link sharing) are not modeled in
  the per-step service time. These belong to the executor layer's SimPy-based
  resource model.
- **GB300 per-GPU peak**: Derived from rack-level total specifications
  (grade C data).
- **H20 specifications**: Sourced from third-party benchmarks (grade B data).
- **Groq 480 GB/s interconnect discrepancy**: The 480 GB/s figure conflicts
  with the ISCA 2022 paper's 4-lane x 25 Gbps/link specification. The data
  directory uses the latter; the former is flagged as unconfirmed.
