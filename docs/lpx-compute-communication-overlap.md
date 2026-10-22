# LPX output-vector compute and communication overlap

## Evidence and scope

NVIDIA describes compiler scheduling at 320-byte vector granularity and sending matrix output slices while the matrix operation is still executing.
The primary source is [How NVIDIA Groq 3 LPX Unlocks Ultrafast Interactivity at Long Context](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/), in the section on fine-grained computation and communication overlap.
The mechanism has high evidence confidence.
The article does not establish a measured operator startup, a measured collective launch, the precise reduction routes, or a complete compiler schedule.

This implementation refines a captured native decode directed acyclic graph (DAG) into explicit output-vector dependencies.
It covers the attention output projection and the dense feed-forward network (FFN) down projection followed by a tensor-parallel (TP) AllReduce on LP30.
Other compute operators, attention kernels, attention/FFN copies, residual gathers, pipeline transfers, and model boundaries remain in the captured native graph.
Global attention placed on Rubin does not receive LPX streaming behavior.
The current native planner does not express mixed per-layer Rubin ownership, so the earlier global-attention-only AF prototype remains a separate experiment.
This model does not implement a compiler, distributed key/value (KV) scanning, or overlap through every attention operation.

The model is experimental and opt-in through the sampled replay execution backend.
Native execution, the isolated calibration backend, and hardware defaults retain their existing behavior.
The prior [latency ledger](lpx-latency-ledger.md) describes the differences between the native graph and isolated operator composition.
The independent [colocated dense reuse setting](lpx-colocated-dense-reuse.md) can remove internal A/F copies and residual gathers from the graph captured by this model.
The fixed-layout tables below keep that setting disabled to isolate the overlap mechanism.

## Timing and dependency rules

The model retains all original operator service, including each operator's fixed startup.
For each eligible rank, all components preceding the output projection, and the output projection's own startup, form a nonstreaming prefix.
The remaining projection service is its existing analytical roofline work.
Output bytes are assumed to become available uniformly during that work.
For a prefix duration `P`, streamable work `W`, output size `N`, and cumulative output position `b`, the output readiness is `compute_start + P + W * b / N`.
This production schedule is an analytical assumption with medium confidence in the qualitative behavior and low confidence in exact timing.
The current attention component list places value normalization after the output projection in its accounting order; the overlap model conservatively includes that normalization in the prefix.

A vector's reduction waits until every TP rank has produced that vector.
The collective pays one launch delay after all ranks produce the leading vector.
The launch delay can overlap production of later vectors.
Each vector then executes reduce-scatter and all-gather phases with a dependency between those phases.
Both phases share the existing endpoint transmit and receive resources.
Injection occupies those resources, while propagation releases them for other vectors.
The following normalization or other whole-buffer consumer waits until every vector finishes both phases.
The trailing vector retains its actual byte count and must align to activation elements.
At BF16 activation precision, 320 bytes represent 160 elements.

The full-message estimate comes from the existing catalog `CollectiveModel`.
For group size `p`, each phase injects `N * (p - 1) / p` bytes per endpoint over all vectors.
The total injection is `2 * N * (p - 1) / p` bytes per endpoint.
The whole-message wire service is divided among vectors without repricing a small-message efficiency curve or charging a new launch per vector.
Measured collectives cannot be decomposed through this interface.
Streaming TP groups must fit one catalog level with a direct collective algorithm.
The algorithm, routing, propagation, and reduction service remain analytical assumptions with low hardware confidence.
The direct estimate treats each vector's endpoint traffic as uniformly distributed fluid traffic.
The 320-byte setting controls output readiness; it does not establish a measured packet size or a cycle-accurate mapping of partial reductions onto physical links.
Packet alignment, owner-specific hotspots, and finite internal reduction capacity require a physical routing schedule before this assumption can be validated.

The four controlled modes distinguish separate modeling changes:

| Mode | LPX collective pricing | Reduction phase pipeline | Early producer output |
| --- | --- | --- | --- |
| `off` | Existing native estimator | Existing native behavior | Disabled |
| `catalog` | Existing topology catalog | Disabled | Disabled |
| `phase-only` | Existing topology catalog | Enabled | Disabled |
| `overlap` | Existing topology catalog | Enabled | Enabled |

The catalog alignment changes the native flat-link ring estimate to the topology's direct estimate.
Its gain is separate from the incremental overlap gain.
A vector size covering the whole output exactly reproduces the catalog control's latency in the tested isolated-request layout.

## Configuration and use

The existing runner accepts these options:

```bash
.venv/bin/python scripts/reproduce_lpx.py \
  --execution-backend sampled-replay \
  --micro-batch 1 \
  --lpx-streaming overlap \
  --lpx-vector-bytes 320
```

Supply the experiment's normal model, configuration, workload, and output options in addition to these flags.
Supported streaming producers are analytical dense single-token decode operations.
Prefill retains its existing profile and records `streaming_skipped: prefill` in profile provenance.
An incompatible multi-token or multi-request decode micro-batch fails explicitly.
Concurrent jobs may arbitrate compute resources at vector boundaries; this scheduling assumption needs separate validation for concurrent serving.
The reported experiment uses one client.
Full execution trace retention is unavailable in the existing sampled replay backend.

Programmatic callers can configure the existing replay factory:

```python
from scripts.lpx.replay_driver import ReplayConfig

execution_factory = ReplayConfig(
    workload,
    streaming={
        "vector_bytes": 320,
        "overlap": True,
        "early_output": True,
        "collective_launch_us": None,
    },
)
```

Pass this factory to `scripts.lpx.native_simulation.point` as `execution_factory`.
The existing experiment runner also accepts `replay_streaming` with the same dictionary when `execution_backend` is `sampled-replay`.
`collective_launch_us: None` retains the catalog default.
The optional `--lpx-collective-launch-us` override changes the assumed launch delay of LPX catalog collectives in this experiment.
It does not alter operator startup, hardware files, or other experiments.
Setting `early_output: False` provides the phase-only control.
Setting `overlap: False` provides the catalog control.
Omitting `streaming`, or using `--lpx-streaming off`, preserves the existing execution model.

Cache identities include the requested streaming settings and code identity.
Modeled profiles retain `native_duration_s` and add `modeled_duration_s` with model assumptions and coverage counts.
Their exported method is `modeled_native_operator_dags`.
The model cannot be mistaken for an unchanged replay by consumers that check this provenance.

## Fixed-layout experiment

The experiment fixes 256 LP30 chips with TP16 and pipeline parallelism (PP) 16.
It uses native Gemma 4 31B, FP8 weights, FP8 KV cache, BF16 activations, and one decode client.
The original 0.2 microsecond operator startup, 0.5 microsecond collective launch, catalog efficiency assumptions, and link assumptions remain unchanged.
Input/output lengths are 10,000/1,500 and 100,000/2,000 tokens.
Time to the first token (TTFT) is excluded.
Each context interval is checked at its endpoints and midpoint for affine latency before reporting its interval mean.
This deterministic context sweep is not a fifty-request text benchmark or a statistical confidence interval.
The model uses native model tokens and does not establish equality with the benchmark's standardized tokenizer.

Reproduce the primary comparison with:

```bash
.venv/bin/python scripts/model_lpx_overlap.py \
  --output outputs/lpx-streaming-model-20261020
```

The result is [results.json](../outputs/lpx-streaming-model-20261020/results.json).
The native graph contains 6,247 events for one decode pass.
The 320-byte model streams 120 collectives and refines 1,920 rank-local producer nodes.
Its 4,080 collective vectors produce a refined graph of 93,967 nodes.
The counts include all TP ranks; throughput still describes one user.

| Model | 10K microseconds/token | 10K tokens/s | 100K microseconds/token | 100K tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Native flat-link control | 458.469 | 2181.17 | 468.484 | 2134.55 |
| Native graph with catalog collectives | 337.369 | 2964.11 | 347.384 | 2878.66 |
| 320-byte reduction phases; whole producer readiness | 336.430 | 2972.38 | 346.445 | 2886.46 |
| 320-byte reduction phases and early producer output | 331.925 | 3012.73 | 341.939 | 2924.50 |
| 640-byte reduction phases and early producer output | 332.092 | 3011.22 | 342.106 | 2923.07 |
| 10,752-byte whole-output control | 337.369 | 2964.11 | 347.384 | 2878.66 |

The incremental 320-byte overlap gain over catalog pricing is 1.64% at 10K and 1.59% at 100K.
The reduction phase pipeline saves 0.938880 microseconds per token.
Early producer output saves another 4.505916 microseconds per token.
The total saving is 5.444796 microseconds per token in both context intervals.
The 121.1 microsecond difference between native flat-link and catalog controls belongs to collective pricing and cannot be attributed to streaming.
The earlier isolated-composition result of about 3,532/3,411 tokens/s uses a different execution path with fewer native layout costs.

The retained native operator startup budget is 192.8 microseconds per token along the single-request path.
The terminal projections expose only a small roofline work interval after their startup.
Streaming those outputs therefore has limited benefit under the current fixed-overhead assumptions.
This implementation does not reproduce the published 100K-over-10K ordering or establish a minimum chip count capable of matching the published throughput.

## Collective launch sensitivity

The official compiler schedule removes some runtime coordination, but its residual collective startup is not published.
The catalog's 0.5 microsecond launch is an unmeasured assumption.
This sensitivity changes that assumption independently of vector overlap and leaves operator startup at 0.2 microseconds.
The 0.1 microsecond case is an illustrative hypothesis; the zero-launch case is a diagnostic bound.
Neither case is an LP30 measurement or a calibrated replacement default.

```bash
.venv/bin/python scripts/model_lpx_overlap.py \
  --output outputs/lpx-streaming-startup-sensitivity-20261020 \
  --vector-bytes 320 \
  --collective-launch-us 0.1 0
```

The result is [startup sensitivity results.json](../outputs/lpx-streaming-startup-sensitivity-20261020/results.json).

| Collective launch assumption | Catalog 10K tokens/s | Streaming 10K tokens/s | Catalog 100K tokens/s | Streaming 100K tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Existing 0.5 microseconds | 2964.11 | 3012.73 | 2878.66 | 2924.50 |
| Hypothetical 0.1 microseconds | 3779.73 | 3859.15 | 3641.88 | 3715.56 |
| Diagnostic zero launch | 4058.95 | 4090.41 | 3900.41 | 3929.45 |

Reducing the collective launch assumption can move this fixed inventory beyond 3,400 tokens/s without changing the operator efficiencies.
The catalog controls show that most of this change comes from the launch assumption itself.
At zero launch, later vectors often wait for producer output, so the incremental early-output benefit shrinks.
This separates a plausible missing coordination mechanism from an unsupported numerical calibration.
All tested fixed-parameter cases retain lower modeled throughput at 100K than at 10K.

## Validation

The tests compare a two-rank streamed reduction with an independent SimPy schedule.
They cover a slower producer rank, a shared injection resource blocked by unrelated traffic, partial vectors, alignment errors, and whole-buffer consumer barriers.
They verify byte conservation, unchanged worker service, a single collective launch, cache round-trips, and rejection of unsupported measured or multilevel decomposition.
The native catalog control matches an independently captured native run using the catalog estimator.
All 551 repository tests passed, and a complete single-client serving smoke test passed with modeled replay provenance.
The command-line smoke test also passed the result audit for independent Rubin and pure-LPX pools.

Implementation files are [streaming geometry](../TokenSim/comm/streaming.py), [native DAG refinement](../scripts/lpx/streaming.py), and [regression tests](../tests/test_lpx_compute_streaming.py).
