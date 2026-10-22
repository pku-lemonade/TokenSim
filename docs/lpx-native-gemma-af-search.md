# Native Gemma AF layout search

The checkout is `lpx-experiment` at revision `83fbbf25b940b01c2c1380ee70113492b25a69be`.
This experiment uses the catalog's native `gemma-4-31b-text` model, compiled rank ownership, resident memory admission, and whole-model `AFService` decode execution.
The search found no conventional attention/FFN disaggregation (AF) layout near 3,400 generation token/s under the tested assumptions.
These results describe TokenSim estimates and do not establish a limit for physical Rubin and LPX hardware.

## Scope and measurement

Conventional AF places all 60 attention layers, embedding, and output processing on Rubin, with all FFN layers on LP30 chips.
Homogeneous controls place the complete model on LP30 chips.
Both paths use the native per-layer Gemma geometry, including the 1,024-token sliding windows, global attention dimensions, and shared global K/V projection.
The model weight dtype is BF16 or FP8, KV dtype is FP8, and activation dtype remains BF16.
There is one resident client, seed 42, no fitting, no speculative decoding, and no hardware parameter overrides.
Time to first token (TTFT), prefill, and prefill-to-decode KV handoff are excluded from generation speed.
Card counts below cover the decode pool.
The existing `Case(mode="af")` serving API additionally requires a complete-model prefill pool, whose cards must be counted when constructing a complete deployment.

| Input tokens | Output tokens | Midpoint decode context |
| ---: | ---: | ---: |
| 1,000 | 1,000 | 1,500 |
| 10,000 | 1,500 | 10,750 |
| 100,000 | 2,000 | 101,000 |

The screening rate is the reciprocal of one native decode step at the midpoint context.
This exploration does not execute the complete five-warm-up plus 50-measured-request AA protocol.
The selected layouts were additionally checked at nine contexts across each output interval and with eight contiguous native decode steps.
Endpoint interpolation and all seven interior checks agree within `5.7e-16` relative latency error for the 10K and 100K workloads.
The largest interpolation discrepancy at 1K is 0.044%, where local attention reaches its window limit during generation.
Contiguous execution agrees with separately executed steps within `3.6e-13` relative error.
The table rates therefore remain screening estimates with explicit native execution checks.

## Default native AF communication

The unmodified native configuration uses `flat_link` ring collectives, endpoint resources, wire occupancy, CUDA graph execution, a 0.2-microsecond device kernel overhead, and a 4-microsecond graph submission cost.
GPU-to-LPX transfers use the catalog's assumed 1,600-Gb/s link with 2-microsecond propagation latency per transfer.
The scan covers 343 distinct precision/layout combinations, comprising 336 conventional AF combinations and seven homogeneous combinations.
Attention tensor parallelism (TP) covers 1, 2, 4, 8, 16, and 32 ranks.
FFN TP covers feasible values among 1, 2, 3, 4, 6, 7, 8, 12, 14, 16, 24, 28, 32, 48, 56, 64, 96, 112, 128, 192, 224, and 256.
LP30 inventory is capped at 256, and pipeline parallelism (PP) covers the minimum feasible depth plus selected larger depths.
Memory feasibility is checked against a 102,000-token peak reservation before pricing each layout.
This is a parameter search over the stated grid, without a proof of a global optimum.

| Architecture | Weight dtype | Decode inventory | A TP / PP | F TP / PP | 10K token/s | 100K token/s |
| --- | --- | --- | --- | --- | ---: | ---: |
| Homogeneous LPX | BF16 | 256 LP30 | 16 / 16 | 16 / 16 | 2,114.86 | 2,070.99 |
| Conventional AF | BF16 | 4 Rubin + 90 LP30 | 4 / 1 | 6 / 15 | 881.37 | 839.48 |
| Homogeneous LPX | FP8 | 128 LP30 | 8 / 16 | 8 / 16 | 2,868.62 | 2,712.76 |
| Conventional AF, best at 10K | FP8 | 2 Rubin + 48 LP30 | 2 / 1 | 4 / 12 | 1,071.65 | 955.67 |
| Conventional AF, best at 100K | FP8 | 4 Rubin + 48 LP30 | 4 / 1 | 4 / 12 | 1,050.77 | 991.76 |

The best 1K AF screening estimates are approximately 886 token/s for BF16 weights and 1,085 token/s for FP8 weights.
None of the tested layouts produces increasing generation speed from 10K to 100K.

At the FP8 100K optimum, the native step takes approximately 1,008 microseconds.
Its dependency path includes approximately 296 microseconds of attention, 295 microseconds of collectives, 244 microseconds of AF transfers, 75 microseconds of FFN computation, and 22 microseconds of LM-head computation.
Normalization, residual, output-selection, graph submission, and resource scheduling account for the remaining time.
The 60 layers require 120 directional AF transfers, whose propagation latency alone contributes approximately 240 microseconds per generated token under the assumed link.
Increasing Rubin TP reduces local attention work while increasing collective costs in the default ring model.

## Catalog collective sensitivity

A separate experiment supplies the ordinary catalog `CollectiveModel` to native DAG collectives while retaining the same kernels, physical rank ownership, endpoint resources, AF cross-link, and hardware parameters.
It uses the LPX topology's hierarchical direct algorithm and Rubin's catalog-selected algorithm, including collective launch costs and message-size efficiency ramps.
This experiment is an estimator-alignment sensitivity implemented in the output script; the production AF provider remains unchanged.
The same 343 distinct combinations were evaluated.

| Architecture | Weight dtype | Decode inventory | A TP / PP | F TP / PP | 10K token/s | 100K token/s |
| --- | --- | --- | --- | --- | ---: | ---: |
| Homogeneous LPX | BF16 | 256 LP30 | 16 / 16 | 16 / 16 | 2,842.96 | 2,764.26 |
| Conventional AF | BF16 | 8 Rubin + 96 LP30 | 8 / 1 | 8 / 12 | 734.73 | 705.38 |
| Homogeneous LPX | FP8 | 144 LP30 | 16 / 9 | 16 / 9 | 2,967.72 | 2,882.07 |
| Conventional AF, best at 10K | FP8 | 1 Rubin + 48 LP30 | 1 / 1 | 8 / 6 | 845.81 | 709.82 |
| Conventional AF, best at 100K | FP8 | 4 Rubin + 48 LP30 | 4 / 1 | 8 / 6 | 798.83 | 764.26 |

The LPX collective improvement does not make the AF path faster in this sensitivity because Rubin's catalog launch costs and small-message efficiency also enter its collectives.
The resulting AF rates remain below 3,000 token/s.

## Relationship to the earlier 3,000+ results

The supplied `calibrate_lpx_aa.py` command invokes an isolated dense-pipeline composition in `scripts/lpx/aa.py`.
That composition prices each stage leader with `OperatorTableLatencyBackend` and advances a SimPy timeout for each token.
It does not invoke the native decode DAG used by this search.
Its prior unfitted BF16-weight, FP8-KV rates are 3,360.91 and 3,251.48 token/s at 10K and 100K.
The corresponding FP8-weight, FP8-KV rates are 3,531.55 and 3,410.92 token/s.

For the identical BF16 TP16/PP16 placement at 10K, the composed path takes approximately 297.54 microseconds per token, the default native DAG takes 472.85 microseconds, and the catalog-collective native DAG takes 351.75 microseconds.
Changing the native collective estimator explains 121.10 microseconds of that difference.
The remaining approximately 54.21 microseconds include additional native gathers and boundary transfers, explicit output processing, graph submission, and endpoint scheduling.
The earlier 3,000+ rates and native AF rates therefore require their executor and communication assumptions alongside the throughput value.

## Global-attention-only AF

The earlier global-attention-only proposal puts the ten global attention layers on Rubin and retains local attention and FFN work on LPX.
The current native AF configuration selects one attention domain and one AF link for every layer of a request.
An execution domain also requires every physical mapping to belong to its declared worker group.
There is no existing configuration switch for assigning local and global attention to different device families within one request.
The relevant implementation is `ModelAFExecutor._run_block`, `scripts/lpx/native_config.py`, and the worker-group validation in `TokenSim/config/af/validation.py`.
This search does not provide a native result for global-attention-only AF.
Adding that architecture requires explicit per-layer placement, weight and KV ownership, boundary transfers, and corresponding validation.

## Reproduction and evidence

Run the following commands from the repository root to reproduce the screening and selected-point checks.
The experiment scripts live in the output directory and do not modify production estimators.
The catalog script requires the completed native screening output, and the validation script requires both completed screening outputs.

```bash
.venv/bin/python \
  outputs/lpx-native-gemma-af-20261020/search_native.py
PYTHONPATH=. .venv/bin/python \
  outputs/lpx-native-gemma-af-20261020/search_catalog.py
PYTHONPATH=. .venv/bin/python \
  outputs/lpx-native-gemma-af-20261020/validate_selected.py
.venv/bin/python -m unittest \
  tests.test_layer_attention \
  tests.test_lpx_aa
```

The ten targeted native-Gemma and AA tests pass.
The branch switch preserved the previous untracked `tests/test_lpx_aa.py` at `outputs/branch-backups/lpx-to-lpx-experiment-20261020/tests/test_lpx_aa.py` because this branch tracks a different file at that path.

- [Experiment manifest](../outputs/lpx-native-gemma-af-20261020/manifest.json)
- [Native screening results](../outputs/lpx-native-gemma-af-20261020/screening.json)
- [Catalog sensitivity and TP refinement](../outputs/lpx-native-gemma-af-20261020/catalog-and-refinement.json)
- [All screening entries as CSV](../outputs/lpx-native-gemma-af-20261020/screening.csv)
- [Selected context and contiguous-step checks](../outputs/lpx-native-gemma-af-20261020/selected-validation.json)
- [Prior exact AA command recheck](../outputs/lpx-aa-command-recheck-20261020/comparison.md)
