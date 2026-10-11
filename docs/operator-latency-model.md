# Operator Latency Model (operator_table backend)

The `operator_table` latency backend (`TokenSim/latency/operator_table.py`)
estimates the duration of every scheduler step. It splits the layers a rank
owns into operators (GEMMs, attention, MoE experts, pointwise kernels,
collectives), looks each operator up in measured tables for the rank's device
and serving stack, and falls back to an analytical model of the device family
when no table answers. `--latency_backend analytical` skips the tables
entirely. Every estimate records where it came from, so a result shows how
much of the simulated time rests on measurements and how much on formulas.

---

## 1. Overview

The system has three layers: static data, the lookup engine and the latency
compositor.

```text
data/devices/*.yaml        Device specs: peak compute per dtype, memory, on-chip storage,
                           interconnect, analytical parameters; every value carries a
                           source_id and an evidence grade
data/topologies/links.yaml Link classes: per-direction bandwidth, hop latency, message-size
                           ramp, collective efficiency
data/topologies/*.yaml     Hierarchical topologies (node -> rack -> cluster); each level
                           records its fan-in, link class and fabric
data/models/*.yaml         Model geometry, per-operator dtypes and operator-module bindings
data/operator_data/<device>/<backend>/
                           Operator tables (parquet) and generation_meta.yaml
        |
        v
TokenSim/hardware          Catalog loading, alias resolution, package selection
                           (HardwareContext)
TokenSim/operator_data     Table schemas, package validation, exact match / interpolation /
                           bounded extrapolation, analytical models, command-line tools
TokenSim/comm              Hierarchical alpha-beta collective model with measured-table
                           override
TokenSim/latency           OperatorTableLatencyBackend and the operator-module estimators
                           (complete attention, residual mixing, fused MoE, MTP draft passes)
```

Every query is resolved in a fixed order:

> **Exact match** --> **Interpolation within the same discrete key** -->
> **Bounded extrapolation** (by default at most 16x past the measured range;
> the boundary row's measured efficiency is kept and an analytical model
> supplies the growth, the rule AIConfigurator uses) --> **Analytical model
> of the device family**

Every query records a match type: `exact`, `interpolated`, `extrapolated` or
`analytical` (complete-attention module queries raised to the first measured
row are reported as `clamped`). The result JSON aggregates these counts per
table, and the shapes no table could answer are written to
`missing_shapes_<qps>.json`, the collection list for the next profiling round
(section 9). Communication lookups (`collective`, `ep_all2all`) follow the same
policy (sections 3.5 and 6), so a run never rests on formulas without saying
so.

---

## 2. Operator Tables

### 2.1 Common Conventions

All tables share two value columns: `latency_us` (execution time in
microseconds) and `source_id`, which must name an entry of the package's
`generation_meta.yaml` `sources` list with its evidence grade and acquisition
method (section 7). Rows may also carry a `kernel` column (the kernel or
framework call that produced the row) and a per-row `grade`; operator-module
tables keep additional metadata columns such as the source hardware and
collection version.

Each table has **key fields** that identify a row. A subset of them are
**axes**, numeric fields along which interpolation is permitted (blended in log
or linear space); the remaining **discrete fields** must match exactly. Dtype
fields are normalized to canonical names (`bfloat16` -> `bf16`,
`half` -> `fp16`, `float8` -> `fp8`, ...).

### 2.2 Primitive Operator Tables

| Table | Key Fields | Interpolation Axes (outer to inner) | Contents |
| --- | --- | --- | --- |
| `gemm` | `dtype, m, n, k` | `k` (log), `n` (log), `m` (log) | `C[m,n] = A[m,k] x B[k,n]`; `dtype` is the weight (storage) dtype |
| `grouped_gemm` | `dtype, out_dtype, batch_size, m, n, k` | `batch_size`, `k`, `n`, `m` (all log) | One grouped kernel over `batch_size` independent weight matrices, `C[g,m,n] = A[g,m,k] x B[g,k,n]` |
| `context_attention` | `attn_dtype, kv_cache_dtype, batch_size, input_seq_len, num_heads, num_kv_heads, head_dim, window_size` | `num_heads` (linear), `batch_size` (log), `input_seq_len` (log, bound 64x) | Prefill attention core (QK^T, softmax, PV) for `batch_size` sequences padded to `input_seq_len` |
| `generation_attention` | `attn_dtype, kv_cache_dtype, batch_size, context_len, num_heads, num_kv_heads, head_dim, window_size` | `num_heads` (linear), `batch_size` (log), `context_len` (log, bound 64x) | Decode attention core: one query token per sequence against `context_len` cached tokens |
| `moe` | `dtype, distribution, num_tokens, hidden_size, inter_size, top_k, num_experts, tp_size, ep_size` | `num_tokens` (log) | Fused/grouped expert FFN on one rank |
| `elementwise` | `op_name, dtype, num_tokens, hidden_size` | `hidden_size` (log), `num_tokens` (log) | Memory-bound pointwise kernels |
| `collective` | `dtype, operation, group_size, nodes, message_bytes` | `message_bytes` (log) | Collective time, nccl-tests convention |
| `ep_all2all` | `dtype, phase, mode, ep_size, nodes, hidden_size, top_k, num_experts, num_tokens` | `num_experts` (log, bound 1024x), `top_k` (linear, bound 8x), `hidden_size` (log), `num_tokens` (log) | Expert-parallel dispatch or combine for one rank |

#### `gemm` and `grouped_gemm`

`dtype` is the weight precision and a discrete key. It also decides which
compute pipe the analytical model uses and how many bytes an activation
element occupies:

| dtype | Compute pipe | Activation bytes |
| --- | --- | --- |
| `fp16`, `bf16` | 16-bit tensor cores | 2 |
| `fp8`, `fp8_block` | FP8 | 1 |
| `int8`, `int8_sq` | INT8 (SmoothQuant W8A8 for `int8_sq`) | 1 |
| `int8_wo`, `int4_wo`, `mxfp4_wo` | 16-bit; weight-only formats dequantize the weights | 2 |
| `nvfp4`, `mxfp4` | FP4, falling back to the FP8 (then FP16) peak when the device lists no FP4 peak | 1 |
| `int4_a8` | FP8 (4-bit weights, FP8 activations) | 1 |
| `int4` | INT4, falling back to the INT8 peak | 1 |

The three `gemm` axes are resolved from outer to inner (`k -> n -> m`) with
logarithmic blending, which matches the roughly log-linear scaling of GEMM
latency with matrix dimension. `grouped_gemm` adds the group count
(`batch_size`) as the outermost axis and an output dtype; the projection plan
of compressed-attention models uses it for the grouped low-rank output
projection.

#### `context_attention` and `generation_attention`

These tables time only the attention core; the QKV and output projections are
separate GEMM queries. `num_heads` and `num_kv_heads` are the TP-local head
counts, `attn_dtype` is the model's activation dtype and `kv_cache_dtype` its
KV-cache dtype. `window_size` supports sliding-window attention; 0 means full
causal attention. `num_heads` is blended linearly because attention time
scales linearly with head count. Decode requests are grouped into
context-length buckets (section 4.4) to limit the number of distinct lookups.

The sequence axes may extrapolate 64x past the measured range: the imported
AIConfigurator tables stop at 16,384 prefill tokens and 131,072 decode context
tokens, while agentic traces reach about one million tokens. With chunked
prefill (section 4.2) prefill queries stay inside the measured range; the
bound mainly serves decode attention over very long contexts and unchunked
runs.

#### `moe`

The row gives the time of one rank holding `num_experts / ep_size` experts,
each sharded `tp_size` ways. `num_tokens` is the number of tokens entering the
layer for the whole expert-parallel group, before top-k expansion; the only
interpolation axis is `num_tokens`. `dtype` is the routed-expert dtype
(`operator_dtypes.routed_expert`, default: the model dtype).

`distribution` is a discrete label for the expert-load distribution used
during collection (`uniform`, `balanced`, `power_law_1.01`, `power_law_1.2`).
The backend tries several labels in order until one matches:

| Model routing | Labels tried |
| --- | --- |
| `uniform` (default) | `uniform`, `balanced`, `power_law_1.01` |
| `skew`, `hot`, `burst` | `power_law_1.2`, the routing name, `uniform`, `balanced` |
| model binding `operator_modules.moe.key.distribution` | only that label |

#### `elementwise`

Covers memory-bound pointwise kernels. `dtype` is the activation dtype and
`op_name` one of:

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

#### `collective`

Measured collective times from nccl-tests or equivalent benchmarks. Following
the nccl-tests convention, `message_bytes` is the full buffer size and the
latency is one out-of-place iteration for `group_size` ranks spread over
`nodes` nodes. Operations: `all_reduce`, `all_gather`, `reduce_scatter`,
`all_to_all`, `send_recv`. Queries use the model's activation dtype. The only
axis is `message_bytes`; unmeasured configurations use the analytical
communication model (section 6).

#### `ep_all2all`

Measured dispatch or combine time (`phase`) for one rank that sends
`num_tokens` local tokens, each routed to `top_k` of `num_experts` experts
spread over `ep_size` ranks on `nodes` nodes. `mode` names the kernel family:
`deepep_high_throughput` and `deepep_low_latency` are selected by
`all2all_backend`; `deepep_v2_context` and `deepep_v2_generation` (vLLM's
split DeepEP variants) are kept in the tables but not selected by the
simulator. DeepEP cost is driven by `num_tokens x top_k x hidden_size`, so the
expert count and `top_k` are interpolation axes rather than discrete keys.

### 2.3 Operator-Module Tables

A module table times a complete fused module whose internal operators would
otherwise be priced one by one. When a module is selected, the compositor
drops the primitives the module already contains, so they are not counted
twice. The table names come from the AIConfigurator data families they are
imported from.

| Table | Discrete key fields | Axes | Timed scope | Replaces |
| --- | --- | --- | --- | --- |
| `dsv4_attention` | `layer_kind` (`csa`/`hca`), `phase` (`context`/`generation`), `mla_dtype`, `kv_cache_dtype`, `kernel_source`, `kernel_dtype`, `model`, `architecture`, `num_heads` (TP-local), `tp_size`, `compress_ratio` | `batch_size` (log), `input_seq_len` (linear, bound 64x), `context_len` (linear, bound 64x) | One complete compressed sparse-attention layer: input projections, compressor/indexer, sparse attention and output projection | attention projection GEMMs, RoPE and the attention core |
| `mhc` | `op_name` (`pre`/`post`), `architecture`, `kernel_source`, `hidden_size`, `hc_mult`, `num_sites`, `sinkhorn_iters` | `num_tokens` (log) | Residual-mixing (hyper-connection) pre or post module for the recorded sites of one layer | the primitive `residual_add` queries |
| `dsv4_megamoe` | `dtype`, `phase`, `kernel_source`, `kernel_dtype`, `distribution`, `pre_dispatch`, `source_policy`, `hidden_size`, `inter_size`, `top_k`, `num_experts`, `num_fused_shared_experts`, `tp_size`, `ep_size` | `num_tokens` (linear; rank-local tokens) | Fused expert-parallel MoE: routed dispatch, expert compute and combine under CUDA graphs; gate/top-k, buffer initialization and shared experts excluded | the `moe` query and the EP dispatch/combine all-to-all |

Module tables are selected in two steps. The model YAML binds a table and the
discrete selectors of the measured kernel under `operator_modules`; geometry
(heads, hidden size, expert counts, compression ratio per layer) and the
parallel layout come from the model and the serving configuration. Excerpt
from `data/models/deepseek-v4-pro.yaml`:

```yaml
operator_modules:
  attention:
    table: dsv4_attention
    key:
      mla_dtype: bf16
      kv_cache_dtype: fp8
      kernel_source: FLASHMLA_SPARSE_DSV4
      kernel_dtype: fp8_block
      model: deepseek-ai/DeepSeek-V4-Pro
      architecture: DeepseekV4ForCausalLM
  mhc:
    table: mhc
    key: {architecture: DeepseekV4ForCausalLM, num_sites: 2}
    kernels:
      pre: vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang
      post: vllm.model_executor.kernels.mhc.tilelang.mhc_post_tilelang
  moe:
    table: dsv4_megamoe
    key:
      kernel_source: deepgemm_megamoe
      kernel_dtype: fp8_fp4
      distribution: power_law_1.01
      pre_dispatch: sglang_jit
      source_policy: random
```

The serving configuration then switches each module on; the default for all
three is `generic` (primitive operators):

| Flag | Binding | Additional requirements |
| --- | --- | --- |
| `--attention_backend module` | `operator_modules.attention` | compressed-attention model (`components`), PP1, CSA/HCA layers |
| `--mhc_backend module` | `operator_modules.mhc` | compressed-attention model |
| `--moe_backend deep_gemm_amxf4_mega_moe` | `operator_modules.moe` (the key must name exactly the five selectors shown) | expert parallelism over more than one rank, PP1, a complete single-node EP group; with DP > 1 also `--dp_lockstep` |

All modules require table-based latency (`--latency_backend operator_table`
with `table_first` or `table_only`) and the bound table in the selected
package. An unmet requirement raises a configuration error at startup or at
the first step that needs it. Module queries never fall back to formulas: a
query whose discrete key has no rows, or that lies beyond the extrapolation
bound, stops the run with an error (complete-attention and residual-mixing
misses are also written to the missing-shape report). The one exception is
complete attention, where queries below the first measured batch or length
are raised to that row and reported as `clamped`.

With the fused MoE module, shared experts are priced separately: when they are
`fp8_block`, by default as an analytical incremental estimate of the fused
shared-expert path (grade D); otherwise, or with
`--disable_moe_shared_fusion`, as serial `gemm` and `swiglu` queries.
Speculative (MTP) draft passes of models with draft layers are priced when
`--mtp_num_speculative_tokens` is positive; `--mtp_acceptance_length` sets the
mean accepted tokens per verification step. See the [AgentX guide](agentx.md)
for a complete configuration.

### 2.4 Package Directory Layout

```text
data/operator_data/<device_id>/<backend>/
  generation_meta.yaml          Dataset version, device_id, backend, sources, calibration,
                                declared row count per table
  <table>_perf.parquet          One file per table present (<table>_perf.csv is read when
                                the parquet file is absent)
  <table>_analysis.parquet      Optional: FLOPs, bytes, compute/memory time, bottleneck and
                                calibration id per row (written by `generate`)
  shape_manifests/<id>.yaml     Manifest a generated package was built from
```

Loading validates the package: `schema_version` 1 and `latency_unit: us`,
every declared table present with its declared row count, no undeclared
tables, every `source_id` declared and no duplicate keys. When the simulator
loads a package it also requires `device_id` and `backend` to equal the
directory names.

`<backend>` names the serving stack whose kernels were measured. The bundled
packages use `vllm`, `trtllm` and `sglang`, plus `nccl` (V100 collectives) and
`analytical` (Groq). A worker group's `operator_backend` in the cluster file
takes precedence over `--operator_backend` (choices `vllm`, `sglang`,
`trtllm`, `analytical`); naming a backend the device does not have is an
error. Without either, the first available backend in this order is used:

> `vllm` > `trtllm` > `sglang` > `measured` > `cuda` > `groq` > `analytical`
> (other names alphabetically)

---

## 3. Lookup Strategy

The lookup engine (`TokenSim/operator_data/lookup.py`) resolves every query
in up to four stages.

### 3.1 Stage 1: Exact Match

The query key is normalized (dtype aliases resolved, strings lowercased,
float-valued integers cast to int) and looked up in a hash map keyed by the
full composite key. A hit returns immediately with `match_type = "exact"`.

### 3.2 Stage 2: Multi-Axis Recursive Interpolation

When no exact row exists but rows share the query's discrete fields, the
engine interpolates:

1. **Discrete key isolation**: all rows sharing the discrete key values form a
   (possibly ragged) grid indexed by the axis values.

2. **Recursive outer-first resolution**: the axes are resolved from outer to
   inner in the order of the table's `axes`. For a GEMM query, `k` is
   resolved first, then `n`, then `m`.

3. **Per-axis blending**: at each level the engine takes the two nearest
   measured values bracketing the target:
   - **Logarithmic blending** (`"log"`), the default for size-like axes:
     ```
     weight = (log(target) - log(lo)) / (log(hi) - log(lo))
     ```
   - **Linear blending** (`"linear"`), used for head counts, `top_k`, the
     `dsv4_attention` sequence axes and the `dsv4_megamoe` token axis:
     ```
     weight = (target - lo) / (hi - lo)
     ```

4. **Result composition**:
   ```
   latency = lat_lo + weight * (lat_hi - lat_lo)
   ```
   The `source_id` values of all contributing rows are joined with `+`, and
   `match_type` is `"interpolated"`.

Consider a GEMM query for `(m=2048, n=4096, k=8192)` with measured points at
`k = 4096` and `k = 16384`. The engine interpolates between the two `k`
values and, at each of them, recursively resolves `n` and then `m`, so the
interpolation follows the nested structure of the measurement grid instead of
flattening it.

### 3.3 Stage 3: Bounded Extrapolation

When the query lies outside the measured range of an axis, the lookup starts
from the nearest boundary row and applies the policy's `extrapolate` mode
(`LookupPolicy` in `TokenSim/operator_data/lookup.py`):

| Mode | Behaviour |
| --- | --- |
| `analytical` (**default**) | `latency = measured(boundary) x analytical(target) / analytical(boundary)`, above or below the measured range. The boundary row's measured *efficiency* is kept and an analytical growth reference carries the change. The operator backend uses its device-family model as reference, the collective model its alpha-beta formula, the complete-attention module a work model of the sparse-attention kernel, and the residual-mixing and fused-MoE modules growth proportional to `num_tokens`. Without a usable reference the mode behaves like `scale`. |
| `empirical` | Fits `latency ~ value^alpha` by least squares over the upper quarter (at least three points) of the measured values on that axis, then multiplies the boundary latency by `(target / boundary)^alpha`. Applies only above the measured range; `alpha` is clamped at 0. With fewer than three usable points, or superlinear fitted growth (`alpha > 1`), it falls back to `analytical`. Not used by default; select it with `LookupPolicy(extrapolate="empirical")`. |
| `scale` | Multiplies the boundary latency by `target / boundary` on logarithmic axes when the target is above the range (work grows with the axis); otherwise returns the boundary latency. |
| `hold` | Returns the boundary row's latency unchanged. |
| `none` | Raises `MissingOperatorDataError` (`extrapolation_disabled`). |

Why `analytical` is the default: scaling by the axis ratio is right for the
GEMM `m` axis but wrong for axes whose cost is not proportional to the value
(decode attention against `context_len` has a large fixed part; collective
latency against `message_bytes` follows an alpha-beta curve). The model's own
growth keeps the extrapolated curve shaped like the work while the measured
point pins its level. AIConfigurator applies the same rule ("hold the
boundary utilisation, let the speed-of-light model carry the growth").

`max_extrapolation_ratio` (default **16**) caps how far a query may go past
the boundary on any axis, in either direction. Axes may override it:
`context_attention.input_seq_len`, `generation_attention.context_len` and the
`dsv4_attention` sequence axes allow 64x (the complete-attention module also
raises its policy bound to 64x on every axis), `ep_all2all.num_experts`
allows 1024x and `ep_all2all.top_k` 8x. Beyond the bound the lookup raises
`MissingOperatorDataError` (`out_of_range`) and the query falls through to
Stage 4.

Results that touched extrapolation have `match_type = "extrapolated"`;
`detail["flags"]` additionally records `analytical_scaled`,
`empirical_scaled`, `linear_scaled` or `held`.

### 3.4 Stage 4: Analytical Fallback

When the table-based stages fail (the table is absent, no row matches the
discrete key, or an axis lies beyond the extrapolation bound), the
device-family analytical model prices the query (section 5) and the miss is
recorded. Operator-module tables do not fall back (section 2.3).

### 3.5 Fallback Policy Configuration

`--latency_fallback` selects one of `FALLBACK_POLICIES`
(`TokenSim/operator_data/lookup.py`). A worker applies the same policy to its
`OperatorTableLatencyBackend` (compute tables) and to its `CollectiveModel`
(`collective`, `ep_all2all`):

| Fallback | Behavior |
| --- | --- |
| `table_first` (default) | Table lookup first; on failure, use the analytical model **and record the miss** in the worker's `MissingShapeReport` |
| `table_only` | Table lookup only; the first miss, including a communication query without a measured row, raises `MissingOperatorDataError` and stops the run |
| `analytical_only` | Skip tables entirely (`--latency_backend analytical` implies this) |

Every miss carries a `kind` (`MISSING_KINDS`): `table_absent` (the package has
no rows for that table, e.g. no `elementwise` data), `discrete_key` (rows
exist but not for this dtype, head layout or `ep_size, nodes`),
`out_of_range` (an axis beyond the extrapolation bound), `no_measured_mode`
(the configured `all2all_backend` is priced by formula by design),
`extrapolation_disabled`, `interpolation_disabled`, `malformed_key` and
`unknown_table`. The report groups misses by `(table, discrete key)`, records
the requested range of every axis, the number of recorded misses and a few
example keys, and is written to `missing_shapes_<qps>.json` (section 9).
Compute and communication misses share one report per worker, so the file is
the collection checklist for the run. The run summary prints the analytical
share per table as a warning whenever it is non-zero.

When the device has no operator package at all, compute operators are priced
analytically without per-shape miss records (MoE and communication queries
still record `table_absent` misses); the analytical-share warning covers every
table.

### 3.6 Result Caching

The backend memoizes estimates per `(table, key)` in a bounded cache
(262,144 entries, cleared when full). Decode steps with steady batch sizes
repeat the same GEMM and elementwise queries, so most steps resolve from the
cache.

The lookup index itself (exact-match map plus per-axis interpolation grids,
`OperatorDataPackage.lookup_index`) is built once per loaded package and
shared by every `OperatorLookup` created for it: each worker creates several
lookups (operator backend, collective model, module estimators). A Blackwell
vLLM package holds about 365,000 rows; loading and indexing it takes a few
seconds once per run.

---

## 4. From Model Config to Operator Queries

The `OperatorTableLatencyBackend` maps the model geometry and the parallel
configuration to per-operator lookups with TP-local dimensions.

### 4.1 Dtypes

| Model field | Used for |
| --- | --- |
| `dtype` | Weight dtype; keys every GEMM-like query unless overridden |
| `operator_dtypes` | Per-operator overrides, e.g. `attention_gemm`, `routed_expert`, `shared_expert`, `router`, `lm_head` |
| `activation_dtype` (default: `dtype`) | `attn_dtype` of the attention tables, `elementwise`, the activation bytes of TP/PP/EP messages, and the `dtype` of `collective` / `ep_all2all` queries |
| `kv_cache_dtype` | `kv_cache_dtype` of the attention tables |

Low-precision checkpoints (fp8, nvfp4, mxfp4) should set
`activation_dtype: bf16`; otherwise attention, elementwise and communication
queries ask for rows in the weight format (for example `nvfp4` attention),
which the tables do not contain, and fall back to formulas.

### 4.2 Local Dimensions Under Tensor Parallelism

Given `TP = tensor_parallel_size`, the backend computes:

```text
heads_local        = ceil(num_attention_heads / TP)
kv_heads_local     = local_kv_heads(num_kv_heads, TP)   # replicated when kv_heads < TP
q_dim_local        = heads_local * head_dim
kv_dim_local       = kv_heads_local * head_dim          # kv_cache_dim / TP for latent KV
inter_local        = ceil(intermediate_size / TP)
vocab_local        = ceil(vocab_size / TP)
layers_local       = layers assigned to this PP stage
moe_layers_local   = MoE layers of this PP stage (ExpertPlacement)
dense_layers_local = layers_local - moe_layers_local
```

### 4.3 Step Composition and Chunked Prefill

`LLMPagedAttnScheduler` builds every step under a token budget,
`max_num_batched_tokens` (default 8192, vLLM V1's online-serving default;
`--no-chunked_prefill` removes the budget). Running requests are served first
in arrival order: a request that has already sampled takes its decode token
(with MTP, the accepted draft tokens), and a request whose context is still
being built (prefill, or recompute after a preemption) takes the next chunk,
`min(remaining_context_tokens, budget_left)`. `--long_prefill_token_threshold`
additionally caps a chunk while other requests are eligible. Whatever budget
is left admits waiting requests, whose first chunk may be partial. A step
therefore mixes decode tokens and prefill chunks (`SchedulePhase.MIXED`), and
a prompt longer than the budget is spread over several steps; only the step
that completes the context samples a token, so TTFT covers all chunks and the
waits between them.

Every request carries `num_computed_tokens` (context tokens whose KV is
present, prefix hits included) and `scheduled_tokens` (this step's share).
`Request.step_tokens` is the latter, or the whole remaining context for a
request that no paged scheduler scheduled (static and dynamic batching). The
backend prices a step from these numbers:

| Query | Rule |
| --- | --- |
| `T` for GEMM, elementwise, MoE and communication | sum of the step's query tokens (MTP verification positions included) |
| decode attention (request already sampled, not recomputing) | `generation_attention(batch=count, context_len=bucket(kv_len))` |
| context chunk (`q_len` = this step's tokens, `kv_len = num_computed_tokens + q_len`) | requests grouped by `(q_len, kv_len)`: `context_attention(batch=count, input_seq_len=q_len)`, scaled by the FLOPs ratio when `kv_len > q_len` (cached prefix and earlier chunks alike) |
| LM head rows | requests sampling this step: decodes plus the last chunk of each context build |

Chunking keeps `input_seq_len <= max_num_batched_tokens`, so long prompts stay
inside the measured prefill range of the tables (16,384 tokens for the
AIConfigurator data) instead of being extrapolated; the growth with context
length lives in the FLOPs ratio, the same correction AIConfigurator applies
for prefixes. With uniform KV caches the blocks of the whole prompt are
allocated at admission rather than per chunk; grouped caches reserve pages per
chunk.

For a step with `T` tokens and `B` sampling requests (standard decoder models;
compressed-attention models differ as described below the table):

| Component | Operator Queries |
| --- | --- |
| **Attention block** (per layer) | `rmsnorm(T, h)` + `gemm(T, q_local + 2*kv_local, h)` + `rope(T, q_local + kv_local)` + attention core + `gemm(T, h, q_local)` + `residual_add(T, h)` |
| **Attention core** | context chunks grouped by `(query_len, kv_len)` issue one `context_attention(batch=count, input_seq_len=query_len)` each, scaled by the FLOPs ratio when `kv_len > query_len`; decode tokens are bucketed by context length (4.4) |
| **Dense FFN** (per dense layer) | `rmsnorm(T, h)` + `gemm(T, g*inter_local, h)` + `swiglu(T, inter_local)` (gated FFNs) + `gemm(T, h, inter_local)` + `residual_add(T, h)`, where `g` is 2 for gated FFNs and 1 otherwise |
| **MoE layer** | `rmsnorm(T, h)` + router `gemm(T, E, h)` + `moe(num_tokens = T * replicas_per_group)` + straggler time (4.5) + `residual_add(T, h)` |
| **LM head** (last PP stage only) | `gemm(B, vocab_local, h)` + `rmsnorm(B, h)` |
| **Communication** | With TP > 1, one all-reduce per layer after the attention output projection, plus one after the FFN for dense layers and for MoE layers without EP (message `T * h * activation_bytes`). EP MoE layers: one dispatch and one combine each (message `T * top_k * h * activation_bytes`). PP: one point-to-point transfer of `T * h * activation_bytes` per step |
| **Fixed overhead** | `device.step_overhead_us` (0 in every bundled device file, grade D) |

Compressed-attention models replace the two attention projections with their
projection plan (four projections, one of them a `grouped_gemm`), add an
analytical hyper-residual output reduction before the LM head (reported as
table `target_hc_head`), and the operator modules of section 2.3 replace the
primitives they contain.

#### Prefix-Cache Scaling

When `kv_len > query_len` (prefix-cache hit, or a later chunk of a long
prompt), the table value for `input_seq_len = query_len` underestimates the
work because queries also attend to the tokens already in KV. The backend
scales the measured latency by `max(1.0, full_flops / base_flops)`, where
`full_flops = 4 * batch * q_len * ((kv_len - q_len) + 0.5 * q_len) *
q_dim_local` (bounded by the sliding window) and `base_flops` is the work of
the measured padded batch.

### 4.4 Decode Tokens

Decode tokens are grouped by context length into buckets of
`--decode_context_bucket` tokens (default 128, rounding up). Each bucket
issues one `generation_attention(batch=count, context_len=bucket_ceiling)`
lookup, which avoids an explosion of distinct attention queries while keeping
the approximation tight. All other operators follow the composition above.

### 4.5 MoE Straggler Modeling

For MoE layers the step waits for the slowest expert rank. The backend:

1. builds a routing histogram for the step (`ExpertRouting`; trace
   histograms or a synthetic distribution, see [MoE](moe.md)),
2. derives the per-rank token load from the expert-to-rank mapping,
3. computes `imbalance = max_rank_load / mean_rank_load`,
4. queries the `moe` table at the balanced token count, and
5. if `imbalance > 1`, queries again at `tokens * imbalance` and adds
   `(busiest_time - balanced_time) * moe_layers_local`.

A row measured with a `power_law_*` distribution already contains its
distribution's slow ranks, so no straggler time is added on top of it.

The `moe` table is keyed by the tokens entering the layer for the whole
expert-parallel group. TP ranks of a replica see the same tokens, so the group
receives `tokens * replicas_per_group`, where `replicas_per_group =
expert_parallel_group_size / tensor_parallel_size`: 1 for
`expert_parallel_scope = per_dp`, `data_parallel_size` for a full `global`
group. Replicas that share a group are assumed to advance in lockstep; this is
an explicit modeling assumption.

### 4.6 Avoiding Double-Counting

- QKV and output projections are **not** in the attention tables; they are
  separate GEMM queries.
- Softmax **is** included in the attention tables.
- SwiGLU is counted separately only for dense FFN layers; the `moe` table
  already includes the activation function.
- Dispatch/combine latency is counted only in the communication component,
  not in the `moe` operator time.
- A complete-attention module includes its projections, RoPE handling and
  attention core; a residual-mixing module includes the residual additions; a
  fused MoE module includes routed dispatch and combine (section 2.3).
- Shared experts count toward parameter memory; the generic MoE path does not
  price their compute (the fused MoE module path does).

---

## 5. Analytical Fallback

When table lookup fails, the system falls back to a roofline-style model
registered per device family (`TokenSim/operator_data/analytical/`).

### 5.1 Core Formula

```text
compute_time = FLOPs / (peak_ops(dtype) * efficiency)
memory_time  = total_bytes / (bandwidth * memory_efficiency)
roofline_us  = max(compute_time, memory_time) + kernel_launch_overhead_us
latency_us   = calibration.multiplier * roofline_us + calibration.offset_us
```

The `Calibration` affine correction (`latency = m * roofline + b`, default
`m = 1`, `b = 0`) is read per table from the package's `calibration:` section
or passed to `generate --calibration`. The `calibrate` command (section 5.5)
fits the efficiency parameters instead.

<a id="analytical-defaults"></a>

### 5.2 Device-Family Defaults

#### `nvidia_gpu` / `generic`

For HBM/GDDR-based GPUs and accelerators:

| Parameter | Default (grade D) | Description |
| --- | --- | --- |
| `gemm_mfu` | 0.70 | Tensor-core utilization for GEMM |
| `attention_mfu` | 0.45 | Utilization for attention kernels |
| `moe_mfu` | 0.50 | Utilization for grouped expert FFN |
| `memory_efficiency` | 0.85 | Fraction of peak memory bandwidth achieved |
| `elementwise_memory_efficiency` | 0.70 | Memory efficiency of pointwise kernels |
| `kernel_launch_us` | 4.0 | Software overhead per kernel launch |
| `gemm_small_m_knee` | 64.0 | Below this `m`, tensor-core tiles waste work; FLOPs are computed as if `m = knee` |

- **Small-m GEMMs**: tensor-core tiles process at least `knee` rows per pass,
  so effective FLOPs are `2 * max(m, knee) * n * k`.
- **MoE overhead**: grouped GEMM kernels issue gate/up and down kernels plus a
  permutation, so the MoE launch overhead is 3x the kernel launch cost.
- **Elementwise kernels**: memory-bound; the model uses the FP32 peak when the
  device lists one.

#### `groq_tsp`

For the Groq Tensor Streaming Processor (LPU):

| Parameter | Default (grade D) | Description |
| --- | --- | --- |
| `gemm_mfu` | 0.80 | Higher utilization due to deterministic scheduling |
| `attention_mfu` | 0.60 | |
| `moe_mfu` | 0.70 | |
| `memory_efficiency` | 0.90 | SRAM bandwidth is more predictable than HBM |
| `kernel_launch_us` | 0.2 | Software-scheduled, no GPU-style launch overhead |

Weights, activations and KV cache reside in on-chip SRAM, so the memory term
uses the on-chip bandwidth (80 TB/s in `groqchip_v1.yaml`) rather than HBM
bandwidth. The model also reports whether the rank-local weights fit in SRAM;
the generator records this as a `fits_on_chip` flag in the analysis table.

#### Device-level overrides and their provenance

A device YAML can override any analytical parameter in its `analytical:`
block and declares where the numbers come from:

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
communication model. `DeviceSpec.analytical_grade` and
`analytical_source_id` expose the provenance to results.

### 5.3 FLOPs and Bytes Formulas

All work accounting is in `TokenSim/operator_data/workload.py`.

- **GEMM**: `FLOPs = 2 * m * n * k`. Weight bytes `k * n * dtype_bytes`;
  activation bytes `(m*k + m*n) * activation_bytes`.
- **Grouped GEMM**: `FLOPs = 2 * g * m * n * k`; weight bytes
  `g * n * k * dtype_bytes`; activation bytes
  `g * m * (k * activation_bytes + n * out_dtype_bytes)`.
- **Prefill attention**: `FLOPs = 4 * B * S * keys_per_query * h_q *
  causal_factor`, with `h_q = num_heads * head_dim`. Causal masking halves
  the effective key range (`causal_factor = 0.5`); a sliding window shorter
  than `S` truncates `keys_per_query` to `window_size` with `causal_factor =
  1`. KV bytes use `h_kv = num_kv_heads * head_dim`.
- **Decode attention**: `FLOPs = 4 * B * keys * h_q`, with
  `keys = min(context_len, window_size)`.
- **MoE**: only routed token-expert pairs contribute: `FLOPs = local_pairs *
  2 * hidden * local_inter * (g + 1)` (gate, up and down for a gated FFN,
  `g = 2`), with `local_pairs = (T * top_k / ep_size) * imbalance` and
  `local_inter = ceil(inter_size / tp_size)`.
- **Elementwise**: FLOPs and bytes per element of the `num_tokens x
  hidden_size` tensor:

| op_name | FLOPs/element | Bytes/element (reads+writes, activation dtype) |
| --- | --- | --- |
| `rmsnorm` | 4 | 2 + weight vector |
| `layernorm` | 5 | 2 + weight vector |
| `rope` | 3 | 2 |
| `residual_add` | 1 | 3 |
| `swiglu` | 4 | 3 |
| `gelu` | 6 | 2 |
| `data_reorder` | 0 | 2 |
| `other_residual` | 1 | 2 |

### 5.4 Accuracy

The analytical model is suitable for order-of-magnitude estimates.
Shape-sensitive kernel behavior (tile selection, wave quantization) still
requires measured tables.

### 5.5 Calibration

The default parameters are grade D assumptions. The `calibrate` command fits
the efficiency parameters of each table (`gemm_mfu`, `attention_mfu`,
`moe_mfu`, `memory_efficiency`, `elementwise_memory_efficiency`,
`kernel_launch_us`, `gemm_small_m_knee`) by a grid search that minimizes the
mean absolute log error against a measured package, holding out 20% of the
rows (sampling at most 3,000 rows per table):

```bash
python -m TokenSim.operator_data.cli calibrate \
    --device a100_sxm_80g --backend trtllm \
    --out tmp/calibration/a100_sxm_80g-trtllm.yaml
```

Mean relative error on the bundled A100 TensorRT-LLM package, before (device
defaults, fit rows) and after fitting (holdout rows):

| Table | Before | After (holdout) |
| --- | --- | --- |
| GEMM | 0.36 | 0.30 |
| Prefill attention | 0.41 | 0.18 |
| Decode attention | 0.62 | 0.41 |
| MoE | 0.47 | 0.39 |

The output YAML holds the fitted parameters and error statistics per table
plus a merged `analytical` block. Shared parameters (`memory_efficiency`,
`kernel_launch_us`) are fitted per table and the merged block keeps the last
table's value, so review the per-table results before copying values into a
device YAML's `analytical:` section (mark them grade C). `generate
--calibration <file>` applies the merged block directly.

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
t_l           = steps_l * alpha_l + bytes_l / (beta_l * links_l * c_l / oversub_l * eff(M_l))
t_collective  = launch_us + sum_l(t_l)
```

Where:
- `alpha_l` = hop latency of the level's link class
- `beta_l` = per-direction bandwidth of the link class
- `links_l` = parallel links an endpoint drives at this level
  (`links_per_endpoint`)
- `c_l` = `LinkClass.collective_efficiency`, the fraction of peak bus
  bandwidth a collective achieves at large messages
- `oversub_l` = the level's `oversubscription` (default 1)
- `eff(M) = M / (M + half_bandwidth_bytes)`, the message-size ramp;
  `half_bandwidth_bytes` is the message size at which a transfer reaches half
  its peak bandwidth
- `launch_us` = software launch/synchronization cost: the device's
  `analytical.collective_launch_us` (10 us in the bundled NVIDIA files), or
  the family default (8.0 us NVIDIA, 0.5 us Groq TSP, 10.0 us generic)

### 6.2 Algorithms

| Algorithm | Rounds (all-reduce) | Injected bytes (all-reduce) | When used |
| --- | --- | --- | --- |
| `ring` | `2(n-1)` | `2(n-1)/n * M` | Bandwidth-optimal for large messages |
| `tree` | `2 * ceil(log2 n)` | `2 * M` | Latency-optimal for small messages |
| `direct` | `2` | `2(n-1)/n * M` | One-shot over a non-blocking fabric (NVSwitch multicast, Groq software-scheduled network) |
| `auto` | -- | -- | Faster of ring and tree per level (matches the NCCL tuner); `direct` for all-to-all and send/recv, `tree` for broadcast |

A topology level can fix the algorithm with `collective_algorithm`. For other
operations:

- **all_gather / reduce_scatter**: ring uses `n-1` rounds with `(n-1)/n * M`
  bytes; tree uses `log2(n)` rounds with `M` bytes; direct uses 1 round.
- **all_to_all**: ring uses `n-1` rounds; direct uses 1 round. Both inject
  `(n-1)/n * M` bytes.
- **send_recv**: always 1 round, `M` bytes.
- **broadcast**: tree uses `log2(n)` rounds; ring uses `n-1` rounds. Both
  inject `M` bytes.

### 6.3 Hierarchical Decomposition

For multi-level topologies the operation runs level by level, matching NCCL's
hierarchical algorithms: reduce-scatter inward, then all-gather outward. After
resolving a level with `fan` participants, the data share handled by the next
outer level is divided by `fan` (all-reduce, reduce-scatter and all-gather).
The cost is computed in O(levels), independent of the number of devices.

### 6.4 Measured Table Override

When the device's operator package contains a `collective` table and a row
matches the query's `(dtype, operation, group_size, nodes)`, the model uses the
measured curve (with the interpolation and extrapolation of section 3; growth
past the measured message sizes follows the alpha-beta formula) and tags the
result `measured` (exact), `interpolated` or `extrapolated`. A query without a
measured row follows the worker's fallback policy (section 3.5):
`table_first` prices it with the formula and records the miss (`collective`,
discrete key `dtype, operation, group_size, nodes`, requested `message_bytes`
range); `table_only` stops the run.

The AIConfigurator packages contain single-node NCCL curves (fp16 and int8;
2, 4 and 8 GPUs, GB200/GB300 2 and 4) and the serving framework's custom
all-reduce in bf16 (GB300 up to 16 GPUs in one NVLink domain); the Intel B60
package contains oneCCL curves (fp16 and int8). The V100 package holds
nccl-tests all-reduce curves in fp32. Because queries use the model's
activation dtype, bf16 TP all-reduces resolve against the custom all-reduce
rows.

### 6.5 Expert-Parallel Dispatch and Combine

MoE layers under expert parallelism issue one dispatch and one combine per
layer. `CollectiveModel.ep_all2all()` prices the pair from the `ep_all2all`
table when the configured `all2all_backend` is a DeepEP mode
(`deepep_high_throughput` / `deepep_low_latency`; the other DeepEP mode is
tried when the requested one has no rows). The query key is this rank's token
count, the model's hidden size, `top_k`, expert count, EP group size and the
number of nodes the group spans. Without a measured row, or for `naive` /
`allgather_reducescatter`, the pair is two analytical all-to-all collectives
scaled by `EP_ALL2ALL_MODE_SCALE` (naive 1.5, allgather_reducescatter 1.0,
deepep_high_throughput 0.7, deepep_low_latency 0.5; project assumptions,
grade D). Both cases are reported: a DeepEP query with no row is a
`table_absent`, `discrete_key` or `out_of_range` miss keyed by `mode,
ep_size, nodes`, a non-DeepEP mode is a `no_measured_mode` miss, and under
`table_only` either stops the run. Results record the outcome in
`parallel_comm_match_type_counts`.

The EP group is the expert-parallel group of `ParallelConfig`
(`expert_parallel_scope`, `expert_parallel_size`). With `per_dp` on an 8-GPU
node the query is `ep_size=8, nodes=1`, which the B200, B300, H100 and H200
packages cover (GB200 and GB300 measure 4-rank groups); a
`global` group over 8 nodes asks for `ep_size=64, nodes=8`, which only the
SGLang H100/H200 tables answer.

### 6.6 Validation Against Measurements

Formula (bundled topologies `dgx_a100_8x_hdr` and `hgx_h100_8x_ndr`, 10 us
launch cost) against the measured NCCL fp16 curves of the TensorRT-LLM
packages:

| Configuration | Message size | Measured | Model |
| --- | --- | --- | --- |
| A100 8-GPU NCCL | 1 MiB all-reduce | 54 us | 92 us |
| A100 8-GPU NCCL | 64 MiB all-reduce | 1169 us | 948 us |
| H100 8-GPU NCCL | 1 MiB all-reduce | 38 us | 63 us |
| H100 8-GPU NCCL | 64 MiB all-reduce | 579 us | 579 us |

The formula is conservative at small message sizes, where launch overhead
dominates; the measured tables cover these cases.

### 6.7 Scope

**In scope**: service time of TP, PP and EP communication, computed per step.

**Not in scope**: link contention, e.g. several micro-batches competing for
the same physical link. TP, PP and EP communication is priced as uncontended
service time within a step; only the grouped KV-storage path queues transfers
on shared links ([Mooncake](mooncake.md)).

---

## 7. Data Sources and Evidence Grades

### 7.1 Evidence Grade System

Every row of an operator package carries a `source_id` that maps to a
`SourceRecord` in the package's `generation_meta.yaml`:

| Field | Description |
| --- | --- |
| `source_id` | Unique identifier (e.g. `aiconfigurator:h100_sxm:gemm:trtllm:1.3.0rc20`) |
| `grade` | Evidence quality: A, B, C or D |
| `method` | How the data was obtained |
| `reference` | URL, paper or path |
| `notes` | Free-form annotation |
| `device` | Device the data was measured on |
| `backend` | Software stack and version |

| Grade | Meaning |
| --- | --- |
| A | Direct measurement on the target device and software stack |
| B | Third-party benchmark, or a measurement on an adjacent device or stack (e.g. module tables borrowed from another system) |
| C | Derived from aggregate specifications or calibrated estimates |
| D | Analytical model output or engineering assumption |

**Acquisition methods**: `measured`, `imported`, `public_benchmark`,
`analytical`, `assumption`.

### 7.2 AIConfigurator Imports

The bundled GPU tables are imported from NVIDIA AIConfigurator with
`import-aiconfigurator` (pinned commit and rebuild steps:
[AIConfigurator data usage](aiconfigurator.md)). Two source layouts are
accepted: the upstream database
(`aic-core/src/aiconfigurator_core/systems/data/<system>/<family>/<backend>/<version>/*_perf.parquet`)
and a flat collector run directory (`*_perf.parquet` or `*_perf.txt` CSV
staging files written by `collector/collect.py` and
`network/collect_comm.sh` on your own node; see `scripts/collect/README.md`).
Database rows are recorded as `method: imported` and collector-run rows as
`method: measured` with the `source_id` prefix `aiconfigurator-collector:`;
both are grade A. Imported operator-module rows are grade B.

The importer converts milliseconds to microseconds, maps dtype and operation
names, keeps the fastest kernel when several cover the same shape (what a
framework's autotuner selects) and records `kernel_source` in the `kernel`
column. Within one family it walks every version directory newest-first and
keeps the first row seen per key, so a newer partial re-collection (for
example the TensorRT-LLM 1.3.0rc23 MXFP4 MoE rows) is layered on top of the
previous full collection instead of replacing it. This mirrors
AIConfigurator's own backward-fill rule. The result is written to `--out`
(default `data/operator_data/<device>/<backend>`); writing replaces the
package's tables, so merge explicitly (`merge_packages`) when updating only a
subset.

Imported systems and backends: `a100_sxm_80g`, `h100_sxm`, `h200_sxm`,
`l40s`, `rtx_pro_6000_server`, `b200_sxm`, `b300_sxm`, `gb200` and `gb300`
with `vllm`, `trtllm` and `sglang`; `intel_arc_pro_b60` with `vllm` (oneCCL
collectives). The per-device status table and the reasons versions and
kernels differ between systems are in `data/devices/README.md`; the coverage
of every bundled package is summarized in the
[data collection checklist](data-collection-checklist.md).

---

## 8. Command-Line Tools

All operator-data commands are invoked as `python -m TokenSim.operator_data.cli
<command>` (installed as `tokensim-operator-data`). The global option
`--data-root` (before the command) selects the data directory; the default is
the checkout's `data/` or `TOKENSIM_DATA_ROOT`. Commands print a JSON summary.

| Command | Purpose |
| --- | --- |
| `manifest` | Write the operator keys needed for models x parallel layouts x workload grid |
| `generate` | Generate an analytical package for a device |
| `import-aiconfigurator` | Convert an AIConfigurator system or collector run directory |
| `import-nccl` | Add nccl-tests output to a package's `collective` table |
| `validate` | Validate a package directory |
| `coverage` | Report exact / interpolated / extrapolated / missing per manifest key |
| `calibrate` | Fit analytical parameters to a measured package |

`manifest`, `generate` and `coverage` share the shape options `--models`
(comma-separated catalog model ids), `--manifest` (an existing manifest,
overrides `--models`), `--experiment-id`, `--tp`, `--ep` (comma-separated
sizes, default 1), `--batch` (decode batch sizes, default 1-256),
`--prefill-tokens` (default 16-8192), `--context` (decode context lengths,
default 128-32768) and `--no-collectives` (skip the default collective grid:
all-reduce and all-to-all over 2, 4 and 8 ranks, 1 KiB to 1 GiB).

### 8.1 `manifest` -- Generate Shape Manifests

```bash
python -m TokenSim.operator_data.cli manifest \
    --models llama-3-70b --tp 4,8 \
    --out tmp/manifests/llama-3-70b.yaml
```

The manifest lists the primitive operator keys (`gemm`, `grouped_gemm`,
`context_attention`, `generation_attention`, `moe`, `elementwise`,
`collective`) for the models, TP/EP sizes and workload grid, tagged with the
roles that need them. Operator-module tables are not included. Without
`--out` it is written to `<data-root>/operator_data/manifests/<id>.yaml`. Use
it as the target list for profiling and as input to `coverage` and
`scripts/collect/collect_elementwise.py`.

### 8.2 `generate` -- Generate Analytical Packages

```bash
python -m TokenSim.operator_data.cli generate \
    --device rtx_4090 --models llama-3-8b --tp 1,2,4,8
```

Evaluates the device family's analytical model for every manifest key and
writes `data/operator_data/rtx_4090/analytical` (grade D, or C with a
calibration), including `*_analysis.parquet` files and the manifest.
Communication keys are skipped; the collective model prices them at run time.
Because `analytical` comes last in the default backend order (section 2.4),
the simulator uses a generated package only when the device has none of the
serving-stack packages, or when it is selected explicitly.

Options: `--backend` (package label, default `analytical`), `--out`,
`--calibration <file>` (per-table `multiplier`/`offset_us` entries and/or an
`analytical:` parameter block, e.g. `calibrate` output), `--dataset-version`,
`--merge` (merge into the package already in the output directory) and
`--keep-existing` (on key clashes keep the existing rows), plus the shape
options.

### 8.3 `import-aiconfigurator` -- Import Measured Data

```bash
git clone https://github.com/ai-dynamo/aiconfigurator.git /tmp/aiconfigurator
git -C /tmp/aiconfigurator checkout f254959eb89e2f206b8f9a77051644d7c1cbdb89
python -m TokenSim.operator_data.cli import-aiconfigurator \
    --system-dir /tmp/aiconfigurator/aic-core/src/aiconfigurator_core/systems/data/h100_sxm \
    --device h100_sxm --backend vllm \
    --upstream-commit f254959eb89e2f206b8f9a77051644d7c1cbdb89 \
    --out /tmp/h100_sxm-vllm
```

Options: `--backend` (default `trtllm`), `--version` (limit the framework
collection version), `--nccl-version`, `--dataset-version`,
`--upstream-commit` (recorded in source references; default `main`),
`--no-custom-allreduce`, `--gpus-per-node` (EP ranks per node for SGLang
wide-EP DeepEP tables; defaults to `num_gpus_per_node` in the upstream
`systems/<system>.yaml`, and a collector run directory without one must pass
it), `--out` and `--module-source
table=system_directory` (repeatable; takes an operator-module table from
another system and records the cross-device provenance).

### 8.4 `import-nccl` -- Import nccl-tests Output

Parses the out-of-place time column of nccl-tests stdout and adds the rows to
the `collective` table of the package at `--out` (default
`data/operator_data/<device>/<backend>`, backend default `nccl`), merging with
the rows already there (new rows win on key clashes). `scripts/collect/run_nccl_tests.sh` runs nccl-tests
and calls this command for every operation. For example, two-node all-reduce
curves written by that script, added to the bundled H100 vLLM package:

```bash
python -m TokenSim.operator_data.cli import-nccl \
    --device h100_sxm --backend vllm \
    --file tmp/nccl/h100_sxm/all_reduce_perf_g8_n2_bfloat16.txt \
    --operation all_reduce --group-size 16 --nodes 2
```

`--operation` is one of `all_reduce`, `all_gather`, `reduce_scatter`,
`all_to_all`, `send_recv`, `broadcast`; `--source-id`, `--reference` and
`--notes` describe the measurement. The rows take the dtype printed by
nccl-tests; TokenSim queries collectives with the model's activation dtype,
so measure with that type (for bf16 models, nccl-tests `-d bfloat16`).

### 8.5 `validate` -- Validate a Package

Checks schema conformance, declared row counts, `source_id` references and
duplicate keys:

```bash
python -m TokenSim.operator_data.cli validate data/operator_data/h100_sxm/vllm
```

### 8.6 `coverage` -- Coverage Report

Reports which manifest keys a package answers exactly, by interpolation or
extrapolation, and which are missing:

```bash
python -m TokenSim.operator_data.cli coverage \
    --device h100_sxm --backend vllm \
    --models llama-3-70b --tp 8 \
    --out tmp/coverage/h100_sxm-llama-3-70b.yaml
```

Without `--backend` the default backend order applies. `--out` writes the
summary and every missing key; without it, the first ten missing keys are
printed.

### 8.7 `calibrate` -- Fit Analytical Parameters

See section 5.5. Options: `--backend`, `--holdout` (default 0.2), `--seed`
(default 0), `--max-rows` (default 3000 per table) and `--out`.

### 8.8 Simulation-Side Flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--latency_backend` | `operator_table` | `analytical` uses formulas only |
| `--latency_fallback` | `table_first` | `table_only`, `analytical_only` (section 3.5) |
| `--operator_backend` | unset | Package backend for every worker group without its own `operator_backend` |
| `--decode_context_bucket` | 128 | Decode context-length bucket width (section 4.4) |
| `--attention_backend`, `--mhc_backend` | `generic` | `module` selects the bound module table |
| `--moe_backend` | `generic` | `deep_gemm_amxf4_mega_moe` selects the fused MoE module |
| `--disable_moe_shared_fusion` | off | Price shared experts serially with the fused MoE module |
| `--data_root` | checkout `data/` | Directory holding `devices/`, `topologies/`, `models/` and `operator_data/` |

A dense model on eight H100 GPUs, priced from the TensorRT-LLM tables:

```bash
./benchmark.py \
    --model llama-3-8b --cluster data/clusters/8_h100/tp8.json \
    --batching paged-attn --request_count 32 --qps 4 \
    --latency_backend operator_table --latency_fallback table_first \
    --operator_backend trtllm --decode_context_bucket 128 \
    --verbose none
```

A compressed-attention MoE model with all three operator modules on eight
B200 GPUs (DP8 with expert parallelism; the cluster selects `vllm`):

```bash
./benchmark.py \
    --model deepseek-v4-pro --cluster data/clusters/8_b200/dep8.json \
    --batching paged-attn --block_size 256 --max_num_seqs 16 \
    --request_count 16 --prefill_mean_len 2048 --prefill_range_len 0 \
    --decode_mean_len 32 --decode_range_len 0 --qps 4 --dp_lockstep \
    --attention_backend module --mhc_backend module \
    --moe_backend deep_gemm_amxf4_mega_moe \
    --verbose none
```

---

## 9. Result Provenance

Every simulation result (`result_<qps>.json`) includes provenance fields:

| Field | Description |
| --- | --- |
| `latency_backends` | Per device: backend, fallback policy, dataset version, operator backend, table row counts, analytical model, cost scopes and module descriptions |
| `operator_query_count` | Number of operator queries |
| `operator_match_type_counts` | Counts per match type across all operator queries |
| `operator_table_match_counts` | The same counts per table |
| `operator_component_seconds` | Simulated seconds per component: `gemm`, `attention`, `moe`, `elementwise`, `lm_head`, `comm`, `overhead` |
| `operator_missing_shape_count` | Unique shapes (compute and communication) that no table answered |
| `operator_missing_shape_groups` | Number of `(table, discrete key)` groups those shapes fall into |
| `operator_missing_per_table` | Per table: missing groups, shapes and recorded misses |
| `operator_missing_shapes` | The first 200 groups (format below) |
| `module_stats` | Per operator module (when modules are active): match counts and time composition |
| `parallel_groups` | TP/EP group of rank 0 and the topology level each spans (`ep_group_size`, `ep_group_link`, ...) |
| `parallel_link_type_counts` | Communication event counts per topology level |
| `parallel_comm_match_type_counts` | Communication estimates by match type (measured, interpolated, extrapolated, analytical) |

The full missing-shape report is exported to `missing_shapes_<qps>.json`
whenever a miss was recorded. From the H100 example in section 8.8:

```json
{
  "group_count": 4, "shape_count": 16, "query_count": 16,
  "per_table": {"elementwise": {"groups": 4, "shapes": 16, "queries": 16}},
  "groups": [
    {"table": "elementwise",
     "key": {"op_name": "residual_add", "dtype": "bf16"},
     "axes": {"hidden_size": [4096, 4096], "num_tokens": [1, 513]},
     "query_count": 4, "shape_count": 4, "kinds": {"table_absent": 4},
     "reason": "table has no rows in this package",
     "examples": [{"op_name": "residual_add", "dtype": "bf16", "num_tokens": 512, "hidden_size": 4096}]}
  ]
}
```

Each group names the table, the discrete key a collection run must reproduce,
the range every interpolation axis was requested over, how many misses were
recorded and why. Estimates are memoized per worker, so a shape is recorded
once per worker rather than once per step. Measure these shapes and add them
to the package to improve coverage.

---

## 10. Known Limitations

- **Ragged batches**: tables measure padding-aligned regular batches. Ragged
  (continuous) batches are approximated by grouping requests by
  `(query_len, kv_len)` and summing the per-group latencies.
- **Prefix-cache attention**: prefill attention with cached context
  (`kv_len > query_len`) scales the measured padded-batch value by the
  analytical FLOPs ratio.
- **EP lockstep assumption**: when an expert-parallel group spans data-parallel
  replicas (`expert_parallel_scope = global`), those replicas are assumed to
  advance in lockstep, so the group's token count is `tokens *
  replicas_per_group`. `per_dp` groups never cross replicas.
- **No MoE tensor parallelism inside an EP group**: experts are split
  `expert_parallel_size` ways with `moe_tp = 1`; a `per_dp` group must equal
  the replica's TP size.
- **Shared experts**: the generic MoE path does not price shared-expert
  compute (section 4.6).
- **Borrowed module tables**: the fused-MoE rows of the B200 and B300 vLLM
  packages were measured on GB200/GB300 with SGLang and are used without a
  conversion factor (grade B); the measured scope and collector buffer
  capacities are kept in the rows and source notes.
- **No link contention in step time**: contention and queuing from pipelined
  micro-batches are not part of the per-step service time.
- <a id="gb300-derivation"></a>**GB200 and GB300 per-GPU peaks** are derived
  from NVL72 rack totals (dense per-GPU value = rack sparse total / 72 / 2; FP4
  from the dense rack figure), grade C. See the `sources` notes of
  `data/devices/gb200.yaml` and `data/devices/gb300.yaml`.
- **H20 specifications** come from third-party benchmarks (grade B).
- **Groq interconnect**: the often-quoted 480 GB/s is a derived bidirectional
  aggregate (16 links x 4 lanes x 30 Gbps x 2 directions); the catalog uses the
  ISCA 2022 per-link figure of 4 lanes x 25 Gbps (link class `groq_c2c_25g`).
