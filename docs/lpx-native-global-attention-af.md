# Native Gemma global-attention-only AF search

## Placement and comparison

This experiment uses the native `gemma-4-31b-text` model on `lpx-experiment` at revision `83fbbf25b940b01c2c1380ee70113492b25a69be`.
Rubin holds only the ten global-attention sublayers at layers 6, 12, 18, 24, 30, 36, 42, 48, 54, and 60.
LPX holds all fifty local-attention sublayers, all sixty feed-forward networks, the embedding, and the language-model output head.
Global-attention projections, attention kernels, associated normalization, and attention output reduction execute on Rubin.
Attention and feed-forward disaggregation (AF) therefore crosses the Rubin–LPX link twenty times per generated token.
The earlier [conventional AF scan](lpx-native-gemma-af-search.md) moved all sixty attention sublayers and does not answer this placement question.

The experiment reuses native `ModelSpec` attention geometry, `OperatorTableLatencyBackend` operators, `CollectiveModel` communication, and native component and key/value (KV) cache accounting helpers.
The isolated serial composition follows the estimator used by `scripts/calibrate_lpx_aa.py`.
The full `ModelAFExecutor` does not currently expose mixed per-layer attention ownership, so this experiment implements routing in a separate prototype under `outputs/lpx-native-global-af-20261020/`.
These results are analytical estimates, and they are not measurements from physical LP30 or Rubin hardware.

## Protocol and search

Input/output lengths are 1,000/1,000, 10,000/1,500, and 100,000/2,000 tokens.
The tokenizer ratios are one, the KV cache uses FP8, and activations use BF16.
Throughput equals `(output_tokens - 1) / (last_output_at - first_output_at)`.
Time to the first token (TTFT) is excluded.
Selected layouts execute a serial decode replay with five warmup requests and fifty measured requests at concurrency one.
The replay evaluates every decode context, and prefill execution and prefill workspace validation are outside its scope.

The search considers tensor-parallel (TP) degrees 1, 2, 4, 8, 16, and 32 independently for LPX and Rubin.
Pipeline-parallel (PP) stages use contiguous memory-valid layer partitions, with up to 512 LP30 chips.
All ten Rubin sublayers fit into one Rubin PP stage for these candidates.
The scan contains 49 capacity-valid pure-LPX and global-only AF configurations across BF16 and FP8 weights.
The selected configuration maximizes the minimum throughput across all three lengths.
The scan does not establish a universal hardware minimum or a global optimum over other parallel architectures.

The memory screen includes weights, peak KV residence at 102,000 tokens, small normalization tensors, and decoder buffers.
Tied embedding and output weights occupy separate endpoint PP stages when required.
Decoder packing uses explicit nonuniform stage boundaries, which differ from the current native layout compiler's uniform partitions.
The reported decoder chip counts therefore describe prototype placements, and they do not establish a complete prefill-capable deployment.

Rubin compute assumes the existing native AF graph kernel cost of 0.2 microseconds.
Collectives retain the catalog's launch costs, topology hierarchy, algorithms, and small-message bandwidth curves.
The cross link retains the catalog's `ethernet1600Gb` assumptions, including 2 microseconds per transfer and 200 GB/s per direction.
Transfers distribute replicated hidden vectors across available source ranks and charge the largest endpoint load when TP degrees differ.
There is no communication/computation overlap between dependent operations in this single-request replay.
No efficiency fitting, hardware-rate change, or zero-cost collective assumption is applied.

## Results

| Weights / KV | Placement | Decode inventory | LPX TP / PP | Rubin TP / PP | 1K tokens/s | 10K tokens/s | 100K tokens/s |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| BF16 / FP8 | Fixed native-AA reference | 256 LP30 | 16 / 16 | — | 3372.55 | 3360.91 | 3251.48 |
| FP8 / FP8 | Fixed native-AA reference | 256 LP30 | 16 / 16 | — | 3544.40 | 3531.55 | 3410.92 |
| BF16 / FP8 | Best pure-LPX prototype | 256 LP30 | 32 / 8 | — | 3391.90 | 3383.85 | 3307.24 |
| FP8 / FP8 | Best pure-LPX prototype | 112 LP30 | 16 / 7 | — | 3551.03 | 3538.14 | 3417.07 |
| BF16 / FP8 | Best global-only AF prototype | 16 Rubin + 160 LP30 | 32 / 5 | 16 / 1 | 2359.03 | 2327.17 | 2056.20 |
| FP8 / FP8 | Best global-only AF prototype | 8 Rubin + 64 LP30 | 16 / 4 | 8 / 1 | 2498.98 | 2463.25 | 2161.72 |

All values in this table use the full generated-token interval and exact context-by-context means.
The global-only AF candidates do not reach 3,400 tokens/s under these assumptions.
The fixed FP8 pure-LPX reference reaches 3,410.92 tokens/s at 100K with the same native operator estimator.
The much smaller decoder inventory of the global-only AF prototype is a separate tradeoff from latency.

## Time composition

The selected FP8 global-only AF prototype takes 405.97 microseconds per generated token at 10K and 462.60 microseconds at 100K.

| Component | 10K microseconds/token | 100K microseconds/token |
| --- | ---: | ---: |
| LPX local-attention operators | 103.39 | 103.39 |
| LPX feed-forward operators | 81.69 | 81.69 |
| LPX reductions | 67.77 | 67.77 |
| Rubin global-attention operators | 42.03 | 98.66 |
| Rubin reductions | 68.25 | 68.25 |
| Twenty cross-device activation transfers | 41.61 | 41.61 |
| Output head and LPX pipeline transfers | 1.23 | 1.23 |

Moving only global attention reduces the cross-device handoff count from 120 to 20 per generated token compared with conventional AF.
Rubin's modeled reductions and HBM reads still exceed the global-attention cost on LPX SRAM in this single-client workload.
Global layers have four KV heads, so the native TP head-partition rule replicates KV heads when TP exceeds four.
Additional Rubin TP ranks therefore stop reducing the per-rank global KV scan once each rank owns one KV head.
The growing global KV scan increases the modeled 100K latency.
The current fixed-parameter model does not reproduce a 100K throughput increase over 10K.

## Diagnostic bounds

The following FP8 cases remove selected costs from every scanned global-only AF configuration and then select the best configuration again.
They isolate model bottlenecks and do not describe achievable serving implementations.

| Assumption | 10K tokens/s | 100K tokens/s |
| --- | ---: | ---: |
| Catalog defaults | 2463.25 | 2161.72 |
| Zero cross-device communication | 2744.58 | 2375.40 |
| Zero Rubin collective communication | 3052.65 | 2602.73 |
| Zero cross-device and Rubin collective communication | 3496.85 | 2918.86 |

Within the tested TP layouts, removing communication alone does not reach 3,400 tokens/s at 100K.
Distributed KV scanning, different kernels, or revised hardware-efficiency assumptions would require separate validation.

## Validation and reproduction

The homogeneous decomposition agrees with the existing native AA backend at eight contexts per weight precision, including both sides of the 1,024-token sliding-window boundary.
The assertion tolerance is `1e-12` relative.
The fixed TP16/PP16 context-by-context means reproduce the existing unfitted native AA command results for both weight precisions.
The prototype also checks that moving global KV away from LPX makes its persistent KV residence independent of long context growth.
Selected decode candidates execute fifty measured serial replays after five warmups, with all generated-token timings preserved in `validation.json`.
The repeated fixed lengths provide a deterministic protocol replay and do not provide independent text samples or a statistical confidence interval.
The repository unit test suite passes all 540 tests, both prototype scripts pass Black checks, and `git diff --check` passes.

```bash
.venv/bin/python \
  outputs/lpx-native-global-af-20261020/native_global.py
.venv/bin/python \
  outputs/lpx-native-global-af-20261020/validate.py
```

The output directory contains `screening.json`, `manifest.json`, `validation.json`, and `idealized-upper-bounds.json`.
The source scripts are experimental artifacts, and no production placement or simulator behavior was modified.

## LPX attention efficiency sensitivity

This follow-up holds the pure-LPX inventory at 256 LP30 chips with TP16/PP16 and retains FP8 KV cache, BF16 activations, and all other baseline parameters.
The catalog's attention compute efficiency is 0.6, so setting `attention_mfu` to 0.8 increases the assumed compute efficiency.
The efficiency parameter applies to the attention kernel's compute term and does not multiply the entire attention sublayer latency.
The decoder still pays SRAM traffic, kernel launch costs, attention projections, normalization, and communication independently.

| Weights | Attention assumption | 10K tokens/s | 100K tokens/s |
| --- | --- | ---: | ---: |
| BF16 | Baseline compute efficiency 0.6 | 3360.91 | 3251.48 |
| BF16 | Compute efficiency 0.8 | 3365.09 | 3282.14 |
| FP8 | Baseline compute efficiency 0.6 | 3531.55 | 3410.92 |
| FP8 | Compute efficiency 0.8 | 3536.16 | 3444.67 |

A separate sensitivity interprets 80% as a relative attention service rate compared with the existing model.
This multiplies the selected operation's duration by `1 / 0.8 = 1.25`, including its fixed kernel cost.
Scaling only attention kernels keeps projections, normalization, and collectives unchanged.
Scaling the whole attention sublayer includes its projections, attention kernels, normalization, and residual arithmetic, while collectives remain unchanged.

| Weights | Relative service-rate assumption | 10K tokens/s | 100K tokens/s |
| --- | --- | ---: | ---: |
| BF16 | Attention kernel service rate 0.8 | 3323.28 | 3190.55 |
| BF16 | Attention sublayer service rate 0.8 | 3030.66 | 2919.89 |
| FP8 | Attention kernel service rate 0.8 | 3490.02 | 3343.94 |
| FP8 | Attention sublayer service rate 0.8 | 3179.00 | 3057.34 |

The native compute-efficiency variants pass twenty direct comparisons with the native AA backend across both weight precisions and both sides of the sliding-window boundary.
Every row uses exact decode-context costs and a five-warmup, fifty-measured-request serial generation replay.
The script `outputs/lpx-native-global-af-20261020/attention_efficiency.py` reproduces these cases and writes `attention-efficiency.json` in the same directory.
These sensitivity runs do not change the hardware catalog or any production source file.
