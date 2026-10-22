# Colocated dense tensor reuse

## Behavior

The optional `colocated_dense_reuse` link setting models local reuse of complete tensors when attention (A) and the feed-forward network (F) execute on the same physical chips.
The default is `false`.
This changes simulated token latency by changing the physical dataflow.
It does not adjust hardware specifications, matrix efficiencies, operator startup, collective launch, or link bandwidth.

Each eligible tensor-parallel (TP) rank consumes its own normalized attention replica after the attention AllReduce.
The F output AllReduce remains a complete dependency barrier.
Each A rank consumes the F output replica on the same physical chip and computes a complete residual vector.
The residual remains replicated for the following layer or pipeline handoff.
The planner creates direct producer-to-consumer dependencies and full-width tensor slice records.
It omits A-to-F and F-to-A communication nodes and the residual AllGather.
It does not assign zero latency to a network transfer.

The original protocol uses a canonical attention producer for the F input, returns residual shards from a canonical F producer, and gathers those shards.
That protocol remains available with the setting disabled.
The base ownership records continue to describe the partitioned layout; each execution node describes the effective replicated layout after collectives and local reuse.

## Eligibility and memory

Reuse applies to dense layers under the `partial_return` execution protocol when all of these conditions hold:

- The link declares `colocated: true`.
- A and F bindings have a one-to-one mapping onto the same physical chips for this layer and replica pair.
- Activation, A-to-F, and F-to-A dtypes are identical.
- Both multi-rank branches use complete AllReduce outputs with full-vector payloads.

Single-rank execution uses its local producer without an AllReduce.
The pairing uses physical identities and supports permuted logical F ranks.
Multi-request blocks, multi-token blocks, pipeline stages, and the existing F batching mechanism retain their ordinary compute and scheduling paths.
Separate physical pools, unmatched data-parallel replicas, transfer precision conversions, sharded collective outputs, and `vllm_afd` retain the original protocol.
Mixture-of-experts (MoE) layers retain their original protocol, including routed communication and combine dependencies.
Dense layers inside a mixed dense/MoE model can qualify independently.

Residual compute is priced at the full hidden width on every rank.
The runtime reserves an additional `colocated_dense_residual` buffer on each attention chip in a colocated pool containing dense work when reuse is requested.
Its size is `workspace_tokens * hidden_size * activation_bytes`.
This conservative reservation remains even when a particular layer or replica pair falls back to the original protocol.
At the fixed single-token Gemma layout below, the reservation is 10,752 bytes per chip.
Model weights and resident key/value (KV) cache retain their existing accounting and admission checks.

## Configuration

Set `colocated_dense_reuse: true` on an existing colocated link in `af.links`.
The configuration rejects a non-boolean value or reuse on a link without `colocated: true`.

The experiment runner exposes `--colocated-dense-reuse` and `--no-colocated-dense-reuse`.
Add the enabled flag to the existing `scripts/reproduce_lpx.py` command and retain the experiment's normal model, configuration, workload, and output options.
The flag works with native execution and sampled replay.
The runner applies the request to colocated pools; separate A/F pools retain their original link configuration.
Programmatic callers pass `colocated_dense_reuse=True` to `scripts.lpx.native_simulation.point`, `scripts.lpx.native_config.configuration`, or the existing experiment job settings.
The result row and `run_settings` report the requested boolean.
Native trace compute events report `details.colocated_dense_reuse` when the local path is used, together with complete tensor slices and dependencies.
Replay and experiment checkpoint identities include the configuration, so enabled and disabled profiles are stored separately.

The existing output-vector overlap model is independently configurable.
Combining `--colocated-dense-reuse` with `--execution-backend sampled-replay --lpx-streaming overlap` enables both mechanisms where supported.
The [overlap model](lpx-compute-communication-overlap.md) describes its analytical timing assumptions and single-token decode restriction.
Ordinary native reuse does not have that streaming restriction.

## Fixed-layout comparison

The comparison fixes native Gemma 4 31B, 256 LP30 chips, TP16, pipeline parallelism 16, FP8 weights, FP8 KV cache, and BF16 activations.
Operator startup stays at 0.2 microseconds and collective launch stays at 0.5 microseconds.
The tokenizer input and output ratios remain 1.
One client decodes after a 10,000-token or 100,000-token input, with output lengths 1,500 and 2,000 respectively.
Time to the first token is excluded.
Each interval uses endpoint and midpoint context samples, with an asserted affine interpolation error below 0.0000001 microseconds.
The result is a deterministic model interval average, not a measured endpoint benchmark or a replay of 50 distinct text requests.

| Execution model | 10K reuse disabled, TPS | 10K reuse enabled, TPS | Gain | 100K reuse disabled, TPS | 100K reuse enabled, TPS | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native flat-link estimator | 2181.17 | 2511.14 | 15.13% | 2134.55 | 2449.54 | 14.76% |
| Catalog collective estimator | 2964.11 | 3458.71 | 16.69% | 2878.66 | 3342.92 | 16.13% |
| Catalog with reduction phase pipelining | 2972.38 | 3469.98 | 16.74% | 2886.46 | 3353.45 | 16.18% |
| Catalog with 320-byte output overlap | 3012.73 | 3525.09 | 17.01% | 2924.50 | 3404.90 | 16.43% |

Catalog pricing, reduction phase pipelining, and output overlap are separate changes from tensor reuse.
Within each row, only the reuse setting changes.
With catalog pricing, reuse saves 48.244032 microseconds per token for both context intervals.
The [latency ledger](lpx-latency-ledger.md) attributed 48.257472 microseconds to A/F traffic and residual gathers under a zero-service diagnostic.
The implemented protocol saves 0.013440 microseconds less because it charges full-width residual compute on every rank.
Native flat-link pricing saves 60.244032 microseconds because that estimator assigns a higher cost to the removed gathers.

The native graph has 4,267 events with reuse enabled and 6,247 with reuse disabled.
Across 60 layers, reuse removes 960 A-to-F transfer legs, 960 F-to-A transfer legs, and 60 residual gathers.
It retains 121 AllReduce operations, 240 pipeline transfer legs, and the logits gather.
The output-overlap model still covers the same 120 terminal projection collectives and 4,080 output vectors.

These values remain conditional on the existing analytical hardware and overlap assumptions.
The 100K estimate remains slower than 10K.
This fixed 256-chip comparison does not establish the minimum chip count or identify the deployed endpoint's weight precision.
The [native minimum-chip search](lpx-native-minimum-search.md) evaluates smaller balanced TP/PP layouts with the same reuse and overlap model.

## Reproduction and validation

```bash
.venv/bin/python scripts/model_lpx_overlap.py \
  --vector-bytes 320 \
  --output outputs/lpx-colocated-reuse-20261020/baseline

.venv/bin/python scripts/model_lpx_overlap.py \
  --colocated-dense-reuse \
  --vector-bytes 320 \
  --output outputs/lpx-colocated-reuse-20261020/local
```

The [baseline results](../outputs/lpx-colocated-reuse-20261020/baseline/results.json) and [reuse results](../outputs/lpx-colocated-reuse-20261020/local/results.json) contain the same source fingerprint, context samples, native durations, streaming coverage, and interval summaries.
The [paired comparison](../outputs/lpx-colocated-reuse-20261020/comparison.json) records the TPS gains and saved latency.

The [tests](../tests/test_colocated_dense_reuse.py) execute the planned tensor slices with independent dense matrices and RMS normalization for single-rank and two-rank cases.
They cover physical rank permutations, precision-conversion fallback, sharded-output fallback, unmatched replicas, invalid coverage, missing reduction barriers, retained residual memory, simultaneous iterations, pipeline transfers, template rebinding, replay cache separation, and unchanged MoE trace events.
All 560 repository tests passed.
Command-line native and sampled-replay runs passed the result audit with independent Rubin and pure-LPX pools and reuse enabled.

Implementation files are [local tensor planning](../TokenSim/af/planning/colocated.py), [DAG validation](../TokenSim/af/planning/reuse_validation.py), and [trace validation](../TokenSim/af/trace/stages.py).
