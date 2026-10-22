# LPX decode latency reconciliation

## Finding

The isolated composition and native full-model execution graph use nearly identical Gemma operator shapes but charge different communication and boundary work.
For a fixed 256-chip configuration, their 10K decode estimates are 3531.55 and 2181.17 tokens per second (TPS).
Every 175.307419 microseconds of the difference is accounted for below.
Agreement between these paths establishes internal accounting consistency; physical accuracy still depends on unmeasured timing assumptions.

The largest shared uncertainty is the 0.2 microsecond startup charged to each logical operator.
This contributes 192.4 microseconds per token in the isolated path and 192.8 microseconds in the native path.
The audit does not change hardware parameters, execution defaults, or supported tensor layouts.

## Configuration and measurement scope

| Field | Value |
| --- | --- |
| Revision | `83fbbf25b940b01c2c1380ee70113492b25a69be`, branch `lpx-experiment`, with the added audit scripts |
| Model | Native `gemma-4-31b-text`, 60 layers, including 50 sliding-window and 10 global-attention layers |
| Inventory | 256 LP30 chips, colocated attention and feed-forward network (FFN) domains |
| Parallelism | Tensor parallelism (TP) 16, pipeline parallelism (PP) 16 |
| Precision | FP8 weights, FP8 key/value (KV) cache, BF16 activations |
| Batch and request concurrency | One request and one token per decode step |
| 10K case | 10000 input tokens, 1500 output tokens, 1499 measured generation intervals |
| 100K case | 100000 input tokens, 2000 output tokens, 1999 measured generation intervals |
| Tokenizer conversion | Input and output ratios fixed at 1 |
| Reservation | Resident weights and KV admission checked through 102000 tokens |

The [audit entry point](../scripts/audit_lpx_latency.py) compares the representative stage backends used by [the serial AA estimator](../scripts/lpx/aa.py) with native `AFService.forward(..., phase="decode")` execution.
The helper [latency ledger](../scripts/lpx/latency_ledger.py) preserves layer multiplicities when instrumenting grouped operator queries.
The native execution graph retains all rank events and resource contention.
The ledger follows the latest-ending predecessor from the final event and accounts for service time, resource queue time, and readiness gaps along that dependency path.
Concurrent rank copies are not summed as serial latency.
The 4 microsecond host submission appears as a readiness gap because the initial embedding event does not name the submission event as a predecessor.

Each generation interval is sampled at nine contexts, including both endpoints and the midpoint.
Latency is affine over both audited context intervals to less than `2e-13` microseconds of numerical residual.
The midpoint therefore gives the interval-average latency under this model.
Eight contiguous native decode steps are also compared with separately initialized steps for the default and catalog variants at both input lengths.
Their maximum discrepancy is below `1e-10` microseconds.
This is a decode accounting experiment; it does not execute prefill, five warmup requests, or the complete 50-request benchmark.
The token ratios, FP8 precision, and fixed output lengths remain study assumptions when comparing with the published endpoint result.

## Comparable results

All times in this table are simulated mean microseconds per generated-token interval.
The two free-service rows are counterfactual attribution experiments that retain the graph and dependencies while setting selected service durations to zero.
They do not implement a valid replacement tensor-layout protocol.

| Execution path | 10K time | 10K TPS | 100K time | 100K TPS |
| --- | ---: | ---: | ---: | ---: |
| Isolated operator composition | 283.161923 | 3531.55 | 293.176101 | 3410.92 |
| Native graph, default flat ring | 458.469342 | 2181.17 | 468.483520 | 2134.55 |
| Native graph, catalog collectives | 337.369342 | 2964.11 | 347.383520 | 2878.66 |
| Catalog, free internal A/F handoffs | 322.595710 | 3099.86 | 332.609888 | 3006.53 |
| Catalog, free handoffs and residual gathers | 289.111870 | 3458.87 | 299.126048 | 3343.07 |

The default and catalog variants execute the same 6247 events per token.
The free-service variants also retain these events, with the selected durations set to zero.
The catalog variant aligns collective pricing with the isolated estimator without changing kernels, endpoint reservations, or point-to-point transfers.
Both communication estimators remain analytical assumptions.
The higher catalog TPS does not establish which estimator matches LP30 hardware.

## Exact latency bridge

The following increments apply to both context lengths.

| Increment from isolated composition to the default native graph | Microseconds per token |
| --- | ---: |
| Collective algorithm and launch model | 121.100000 |
| Internal A/F copies and their endpoint queues | 14.773632 |
| Sixty residual all-gathers, priced with the catalog model | 33.483840 |
| Boundary work and residual shape difference | 5.949947 |
| Total | 175.307419 |

The collective-model increment includes the price difference for all 121 all-reduces and all 61 all-gathers in the native graph.
The residual-gather increment therefore uses the aligned catalog price to avoid counting the algorithm difference twice.

### Collective pricing

The isolated path charges 120 all-reduces and 15 PP transfers.
The native path adds an embedding all-reduce, 60 residual all-gathers, and a logits all-gather.
Its dependency path also contains 15 PP transfers.

For a 10752-byte hidden activation across 16 ranks, the flat ring all-reduce costs `30 * 0.05 + 2 * 15 / 16 * 10752 / 1.25e6 = 1.516128` microseconds.
The catalog direct estimator charges 0.616128 microseconds for the same all-reduce.
This difference contributes `121 * 0.9 = 108.9` microseconds.
The remaining 12.2 microseconds come from the 61 all-gather price differences.

The catalog uses the declared tray hierarchy and an unmeasured 0.5 microsecond collective launch assumption.
The flat estimator uses ring rounds with the declared 0.05 microsecond link latency.
Published aggregate bandwidth does not verify either complete collective latency.

### Colocated A/F handoffs and residual ownership

The [copy planner](../TokenSim/af/planning/copies.py) selects a canonical producer for replicated tensors.
For this colocated layout, each layer sends attention output from that producer to all 16 FFN recipients, including its own physical endpoint.
The FFN return distributes shards from a canonical producer to the attention ranks.
The resulting residual ownership then requires an all-gather before the following layer.

| Internal A/F cost on the recorded dependency path | Microseconds per token |
| --- | ---: |
| A-to-F copy service | 3.516096 |
| F-to-A copy service | 3.032256 |
| A-to-F endpoint queue | 7.741440 |
| F-to-A endpoint queue | 0.483840 |
| Total | 14.773632 |

These costs follow from the current generic AF protocol even though both domains occupy the same physical chips.
A colocated dense-TP protocol could reuse equivalent local replicas after all-reduce and keep residual ownership replicated.
Such a change must establish correct producers, tensor coverage, and normalization semantics before removing copies or gathers.
Setting durations to zero in this audit only measures their contribution.

### Boundary accounting

After aligning collectives and removing the diagnostic internal-copy and residual-gather charges, 5.949947 microseconds remain.

| Boundary difference relative to isolated composition | Microseconds per token |
| --- | ---: |
| Host submission readiness | 4.000000 |
| Embedding lookup | 0.200159 |
| Embedding all-reduce | 0.616128 |
| Logits all-gather | 0.943216 |
| Greedy token selection | 0.203884 |
| Sharded versus full-width FFN residual addition | -0.013440 |
| Total | 5.949947 |

Final normalization and the language-model output projection already exist in both paths.
Their query shapes and costs match.
The only operator-multiset differences are embedding lookup, token selection, and 60 residual additions with width 336 in the native path versus 5376 in the isolated path.
Attention, query/key/value projections, FFN projections, and the remaining normalization and activation queries match exactly.

## Compute, SRAM, and startup budget

For each analytical operator, charged latency is `max(compute_time, memory_time) + startup`.
The roofline columns below assign each operator's winning term to either compute or SRAM bandwidth.
Adding raw compute and memory times would double-count work overlapped inside an operator.

| Operator budget | Isolated 10K | Native 10K | Isolated 100K | Native 100K |
| --- | ---: | ---: | ---: | ---: |
| Logical operators on the serial path | 962 | 964 | 962 | 964 |
| Startup, microseconds | 192.400000 | 192.800000 | 192.400000 | 192.800000 |
| Compute-bound roofline, microseconds | 1.476883 | 1.476883 | 11.491061 | 11.491061 |
| SRAM-bound roofline, microseconds | 14.470656 | 14.461259 | 14.470656 | 14.461259 |
| Total operator service, microseconds | 208.347539 | 208.738142 | 218.361717 | 218.752320 |

The common startup charge represents 68% of isolated 10K latency and 57% of catalog-native 10K latency.
It represents approximately 92% of operator service at 10K.
Changing matrix multiplication efficiency alone therefore has limited influence on this particular 256-chip configuration.
At 100K, the compute-bound attention term grows by 10.014178 microseconds per token.
All audited variants consequently predict lower TPS at 100K.
A context-independent fixed per-token overhead cannot reverse this ordering.

## Physical interpretation and model priorities

NVIDIA describes compiler-scheduled compute, explicit data movement, and coordinated SRAM/C2C execution in the [LPX architecture article](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/).
Its [long-context article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) describes sending 320-byte result vectors while matrix multiplication is still executing.
These descriptions support a model with compiled execution regions and partial-result readiness.
They do not establish the region boundaries, their startup costs, or complete collective latency for this Gemma deployment.
Treating 0.2 microseconds as a separately measured startup for every logical operator has no verified basis.
The 4 microsecond `cuda_graph` submission assumption also lacks an LP30-specific measurement boundary.

| Item | Confidence in the audit result | Confidence in LP30 physical timing |
| --- | --- | --- |
| Operator shapes and charged event counts | High, checked against both implementations | Conditional on precision and partition choices |
| Exact latency bridge | High, algebra and replay assertions | Conditional on the existing analytical estimators |
| 0.2 microseconds per logical operator | High confidence that the code charges it | Low, unmeasured assumption |
| Flat ring versus catalog direct collective | High confidence in the implemented difference | Low, algorithm and startup need evidence |
| Internal colocated A/F traffic | High, explicit rank and endpoint records | Current simulator protocol; deployment mapping unknown |
| Fine-grained compute/communication overlap | Absent within the native operator boundaries audited here | Supported qualitatively by NVIDIA; magnitude unknown |
| Minimum chip count reproducing published TPS | Not established by this experiment | Requires a physically justified common execution model |

An optional [colocated dense-TP reuse protocol](lpx-colocated-dense-reuse.md) implements local replica consumption and full-width residual computation.
Its paired experiment saves 48.244032 microseconds per token with catalog collective pricing, including the extra residual compute service.
A shared LPX-specific collective estimator and measurement boundaries for startup remain modeling priorities.
Compiled-region startup and partial-result communication overlap need explicit measurement boundaries and sensitivity ranges.
They must preserve dependencies and resource limits; applying a global `max(total_compute, total_communication)` would assume overlap between dependent operations.
The pure-LPX and global-attention-on-Rubin searches should use this common execution model after these choices are justified.
Unknown coefficients should remain labeled sensitivity parameters instead of being selected solely to hit the TPS target.

The published medians are 3382 TPS at 10K and 3431 TPS at 100K, using a standardized tokenizer, according to the NVIDIA long-context article linked above.
The present study fixes tokenizer ratios at 1 and does not establish the endpoint's weight precision or partitioning.
Matching one TPS value under these assumptions cannot determine a unique minimum chip count or validate the 100K ordering.

## Reproduction and validation

```bash
.venv/bin/python scripts/audit_lpx_latency.py \
  --samples 9 \
  --output outputs/lpx-latency-ledger-20261020
```

The [result ledger](../outputs/lpx-latency-ledger-20261020/ledger.json) contains source hashes, hardware assumptions, interval samples, dependency-path events, operator keys and multiplicities, exact reconciliation, and contiguous-step checks.
The audit completed locally in 87.33 seconds.
This elapsed time is the host's audit runtime and is separate from simulated token latency.
The [accounting tests](../tests/test_lpx_latency_ledger.py) cover weighted query counts, unlinked host readiness, resource queues, complete latency reconciliation, and restoration of patched estimators.
The complete unit suite passed with 543 tests.
