# LPX context-parallel implementation plan

Plan date: 2026-10-21.
This plan adds context parallelism (CP) to the native attention/feed-forward network (AF) simulator so that LPX long-context decode can shard the key/value (KV) cache by sequence position.
The [Phase 0 specification](lpx-context-parallel.md) defines the execution semantics, cost formulas and pre-registered expectations.
Each phase updates the status table below when it lands.

## Motivation

The published Gemma 4 31B speeds are 3,382 output tokens per second (TPS) at 10K input and 3,431 TPS at 100K input.
The [integrated LPX model](lpx-integration-20261021.md) predicts 5,578 TPS at 10K and 3,611 TPS at 100K for TP16/PP8 on 128 LP30 chips.
Tensor parallelism (TP) splits attention heads, and pipeline parallelism (PP) assigns layers to stages.
The ten Gemma global-attention layers have four KV heads with head dimension 512.
At TP16 each global KV head is replicated on four chips, and every replica reads the complete context.
The [unit-service ledger](../experiments/lpx/fidelity-results.json) gives 11.0 microseconds per global-attention core at context 101,000 and 1.2 microseconds at context 10,750.
Across ten global layers this difference is approximately 98 microseconds per token, which matches the modeled 10K-to-100K per-token difference.

[NVIDIA's long-context article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) describes "parallelism for both model weights and long-context KV cache" without disclosing the scheme.
NVIDIA's [Helix parallelism paper](https://arxiv.org/abs/2507.07120) describes a decode scheme that shards the KV cache by sequence inside the existing attention TP group.
This plan implements a Helix-style scheme as a literature hypothesis.
No reviewed source establishes that the LPX endpoint uses it.

## Decisions

| Decision | Selected option | Alternative retained for comparison |
| --- | --- | --- |
| Rank mapping | KV parallelism (KVP) factors the attention TP group: `N = TPA * KVP`, where TPA is attention TP and the FFN keeps TP `N` | CP as an additional rank dimension is out of scope |
| Layer selection | `kv_parallel_layers="full_attention"` applies KVP to layers without a window; `"all"` also applies it to windowed layers | Search scripts may derive `KVP = N / min(N, kv_heads)` per model |
| Token ownership | Round-robin blocks of `kv_append_block` tokens, 16 by default, applied to prompt and generated tokens | A block of `ceil(maximum_context / KVP)` tokens gives contiguous ownership |
| Scope | Pure-LPX colocated decode, followed by prefill | Prefill/decode handoff, external draft and Rubin attention reject `KVP > 1` |

`Layout(a_tp=TPA, a_cp=KVP)` keeps the existing `a_ranks = a_tp * a_cp * a_pp` inventory formula.
The domain configuration stores the per-stage rank count as `tensor_parallel_size = N` and the sequence factor as `kv_parallel_size = KVP`.

## Phases

### Phase 0: specification and pre-registration

Write the [specification](lpx-context-parallel.md) with semantics, ownership, communication volume, memory accounting, scope and expected results.
Record expectations before implementing timing behavior.

### Phase 1: configuration and ownership functions

Add `kv_parallel_size`, `kv_parallel_layers` and `kv_append_block` to `DomainParallelConfig`.
Validate that KVP divides TP, that the layer selection is known and that only attention domains declare `KVP > 1`.
Add `TokenSim/af/runtime/kv_sequence.py` with closed-form token ownership, peak resident tokens, rank coordinates and per-layer attention partitions.
The runtime compiler rejects `KVP > 1` until Phases 2 and 3 implement memory and execution, so no configuration can silently ignore the setting.
Tests compare closed forms with brute-force enumeration and check invalid configurations.

### Phase 2: runtime memory and admission

Shard KV heads by TPA on CP layers and attach a sequence partition to each rank's KV partition.
Charge locally owned tokens in `admission.rank_layer_kv_bytes`.
Charge Q/K/V projection weights replicated across KVP ranks in `memory.py`.
Size LP30 attention scratch from the local context and add partial-output and log-sum-exp (LSE) combine buffers in `lpx_memory.py`.
Tests check that per-rank KV sums to the unreplicated total and that replicated projection weights appear in memory reports.

### Phase 3: decode graph and costs

Add `TokenSim/af/planning/context_parallel.py` and call it from `planning/layers.py` for CP layers.
Each CP layer executes local attention, an `attention_combine` all-to-all inside the KVP group, the output projection over `heads / N` and the existing all-reduce.
Split attention components into pre-attention, core and output projection, and pass each rank's local context to the attention core.
Add LSE merge work to the LP30 vector-unit estimator.
Allow `all_to_all` in AF collectives and route its pairwise messages in the flat collective model and the LPX fabric refinement.
Place new execution code in new modules because `execution/communication.py` has 499 lines.
Add local context and CP layout to directed acyclic graph (DAG) template and service cache keys.
The implementation builds the combine node directly from the attention all-reduce's network and resources, so `_COLLECTIVE_OPERATIONS` and the execution modules need no change.
The existing endpoint and flat providers already price `all_to_all`, and the KVP fields in the attention local shape separate the estimate cache keys.
With `KVP = 1`, complete results and traces must match the current implementation field by field.

### Phase 4: LPX experiment entry points

Remove the `a_cp != 1` rejections in `scripts/lpx/config.py` and `scripts/lpx/native_config.py`.
Map `a_cp` to `kv_parallel_size` and extend `native_search.layouts` to enumerate (TPA, KVP) pairs.
Add command-line options, fidelity-configuration fields and checkpoint fingerprints for KVP, block size and layer selection.
The implementation keeps these settings in the layout, command line and checkpoint parameters and adds no fidelity-configuration section, because they describe a deployment.

### Phase 5: prefill under CP

Extend `CompletePrefill` and `OperatorTableLatencyBackend` so each prefill chunk attends across distributed KV with the same combine step.
Reuse the Phase 1 and Phase 3 helpers and keep prefill ownership identical to decode ownership.
The implementation bounds the chunk's own positions by its query count on every rank, as the [specification](lpx-context-parallel.md#implementation) describes.

### Phase 6: experiments and documentation

Verify unchanged `KVP = 1` results against the integration values 5,908.78, 5,578.41 and 3,611.17 TPS.
Run fixed TP16/PP8 layouts with KVP 1, 2 and 4 at 1K, 10K and 100K, with contiguous ownership and block-size controls.
Rerun the minimum-chip search with CP and the C2C-latency and collective-launch sensitivities.
Update the [coverage-gap analysis](lpx-coverage-gap-analysis.md), the [study guide](lpx-study-guide.md) and the [parameter evidence audit](lpx-parameter-evidence-audit.md).

Every phase runs Black, the complete test suite, `git diff --check` and the 500-line check required by [CONTRIBUTING.md](../CONTRIBUTING.md).

## Status

| Phase | Status | Record |
| --- | --- | --- |
| 0 | Complete | [Specification](lpx-context-parallel.md) |
| 1 | Complete | `TokenSim/af/runtime/kv_sequence.py`, `DomainParallelConfig`, compiler guard and tests |
| 2 | Complete | `layer_kv.py`, `admission.py`, `memory.py`, `component_storage.py`, `lpx_memory.py` and runtime tests |
| 3 | Complete | `planning/context_parallel.py`, `providers/context_parallel.py`, LP30 `lse_merge`, LPX streaming support and graph tests |
| 4 | Complete | `Layout.a_stage_ranks`, KVP mapping in `native_config.py`, `native_search.layouts` and command-line options |
| 5 | Complete | `latency/operator_table/kv_parallel.py` applied by `CompletePrefill` |
| 6 | Complete | [Results](lpx-context-parallel.md#results) and `scripts/run_lpx_context_parallel.py` |

## Exclusions

Batch-wise overlap of the combine step (Helix HOP-B) has no effect for one serial client and is excluded.
CP combined with prefill/decode handoff, external drafting, Rubin attention or non-colocated AF remains rejected until a later plan specifies its KV movement.
Ring-attention prefill with large chunks is excluded; prefill uses the decode combine step for each chunk.
