# LP30 compute-efficiency sensitivity

This experiment reruns the pure LPX Gemma 4 31B decode study with general matrix multiplication (GEMM) efficiency reduced from 0.8 to 0.3 and attention-core efficiency reduced from 0.6 to 0.1.
The device catalog defaults remain unchanged.
The reference is the 0.05 microsecond operator-floor scenario in [the timing rerun](lpx-timing-recommendations.md).

## Configuration and scope

The model uses FP8 weights, FP8 key-value (KV) cache, and BF16 activations.
Each verified workload has one serial client, five warm-up requests, and 50 measured requests.
Input lengths are 1,000, 10,000, and 100,000 tokens, with output lengths of 1,000, 1,500, and 2,000 tokens.
Time to first token is excluded by supplying resident prompt KV and the initial output token at request admission.
Output speed counts the remaining output tokens and their decode time.

| Setting | Reference | Sensitivity |
| --- | ---: | ---: |
| `gemm_mfu` | 0.8 | 0.3 |
| `attention_mfu` | 0.6 | 0.1 |
| SRAM bandwidth efficiency | 0.9 | 0.9 |
| Operator residual, microseconds | 0.05 | 0.05 |
| Additional collective launch, microseconds | 0 | 0 |
| Additional invocation, microseconds | 0 | 0 |
| Uniform chip-to-chip phase latency, microseconds | 0.35 | 0.35 |
| Native graph submission, microseconds | 4 | 4 |

The GEMM coefficient applies to the feed-forward network (FFN), attention projections, and language-model head.
The attention coefficient applies to the attention core, which performs the query-key and attention-value computations.
Elementwise efficiency and every other catalog coefficient retain their reference values.
The experiment overrides the hardware context in memory and records the complete analytical parameter dictionary in each experiment checkpoint.

## Throughput results

All speeds below are simulated output tokens per second.

| LP30 chips | Tensor parallelism | Pipeline parallelism | Compute setting | 1K | 10K | 100K |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 88 | 8 | 11 | GEMM 80%, attention 60% | 5,854.96 | 5,785.42 | 5,184.66 |
| 88 | 8 | 11 | GEMM 30%, attention 10% | 5,704.52 | 5,330.00 | 3,249.00 |
| 112 | 16 | 7 | GEMM 80%, attention 60% | 6,461.17 | 6,418.60 | 6,030.95 |
| 112 | 16 | 7 | GEMM 30%, attention 10% | 6,368.50 | 6,128.15 | 4,478.95 |

The 88-chip configuration loses 2.57%, 7.87%, and 37.33% at the three respective input lengths.
Its 100K speed falls below both the common 3,400-token target and the workload-specific 3,431-token target.
The 112-chip configuration retains a 30.54% margin over the workload-specific 100K target.
The minimum passing configuration in this search increases from 88 to 112 chips, a 27.27% increase.
Tensor parallelism 16 and pipeline parallelism seven also provide the highest worst-workload throughput among the evaluated layouts.

The grid contains 236 tensor/pipeline pairs with a ceiling of 256 LP30 chips.
The sensitivity reruns all 36 capacity-feasible pairs and reuses 200 capacity rejections from the identical-memory reference.
The final 112-chip configuration receives the serial replay and additional interpolation-probe checks described below.
The audit passes all 41 experiment checkpoints and eight additional context probes.
Its three verified workloads contain 150 measured requests, 15 warm-up requests, and 247,335 serial token-clock advances.
The full sensitivity sweep and verification take 384.73 seconds of host time.

## Why FFN efficiency has little effect here

The analytical estimator computes primitive latency as:

```text
compute_time = operations / (peak_compute * compute_efficiency)
sram_time = transferred_bytes / (peak_sram_bandwidth * memory_efficiency)
primitive_latency = max(compute_time, sram_time) + operator_residual
```

The compute coefficient specifies the compute-side limit inside this formula.
It does not impose an observed utilization on the entire module or on the complete model execution.

For one `gate_up` GEMM at tensor parallelism eight, the shape is M=1, N=5376, K=5376.
The compute term increases from 0.05872 to 0.15659 microseconds when the GEMM coefficient falls to 0.3.
The SRAM term remains 0.21416 microseconds, so the primitive latency remains 0.26416 microseconds after the 0.05-microsecond residual is added.
The measured primitive estimates show that every FFN GEMM and attention-projection GEMM remains SRAM-bound in both inspected layouts.

Separate ablations on the 88-chip layout confirm this consequence across all three workloads.
Reducing only GEMM efficiency to 0.3 reproduces the reference throughput exactly.
Reducing only attention-core efficiency to 0.1 reproduces the combined sensitivity throughput exactly.

At a sampled context of 101,000 tokens and tensor parallelism eight, the attention-core service summed along tensor rank zero across all 60 layers increases from 25.98 to 140.89 microseconds.
These primitive service sums exclude communication and do not represent refined-graph wall-clock shares.
The increase is consistent with the longer-context throughput loss.

## Interpretation limits

The predictions retain 90% effective SRAM bandwidth, fine-grained output-vector communication overlap, the 0.05-microsecond operator residual, and the uniform 0.35-microsecond communication-phase proxy.
The current streaming model assumes uniform production of terminal-GEMM output vectors.
These assumptions have not been calibrated against LP30 operator measurements or the serving configuration used for the published throughput.
Lowering two compute coefficients does not validate the retained memory, scheduling, communication, or operator-work estimates.

The capacity search reserves decode weights, protocol buffers, and one resident KV lease through a maximum context of 102,000 tokens.
The study reuses the reference search's capacity partitions because compute-efficiency changes do not change these memory requirements.
Each rerun asserts that its memory report matches the reference exactly.
The search selects one capacity-minimizing contiguous layer partition for each tensor/pipeline pair.
Any minimum card count applies to that search space and the retained modeling assumptions.

Native directed acyclic graph samples supply context-dependent token latency through adaptively checked interpolation.
The serial replay advances each token with that latency curve and checks sequential admission and KV release.
The complete resource graph is not rerun for every token of every request.
Prefill workspace, request-text diversity, and tokenizer conversion are outside this sensitivity experiment.

## Artifacts

The experiment source is [the sensitivity runner](../outputs/lpx-mfu30-attn10-20261021/run_sensitivity.py).
The full results and checkpoint references are in [the local results](../outputs/lpx-mfu30-attn10-20261021/results.json).
The primitive estimates are in [the roofline ledger](../outputs/lpx-mfu30-attn10-20261021/roofline.json).
The portable result is [the audited experiment summary](../experiments/lpx/mfu30-attn10-results.json).

```bash
.venv/bin/python \
  outputs/lpx-mfu30-attn10-20261021/run_sensitivity.py
.venv/bin/python \
  outputs/lpx-mfu30-attn10-20261021/inspect_roofline.py
.venv/bin/python \
  outputs/lpx-mfu30-attn10-20261021/audit_results.py
```
