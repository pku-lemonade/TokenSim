# Shared-KV assistant MFU search at 100K context

Progress date: 2026-11-02.
This study extends the [maximum-MFU search](lpx-maximum-mfu-search.md) with a complete draft-plus-verification cost model.
The non-speculative incumbent remains 3.03914% model FLOP utilization (MFU) under the declared uncalibrated LP30 model.
Acceptance-dependent results below are conditional predictions.
The repository contains measured 100K GPU accepted-prefix distributions.
The current LP30 FP8-KV and FP8-assistant projections still lack matching-precision acceptance measurements and a certified global hardware optimum.

## Model structure and storage

The bundled [assistant configuration](../experiments/lpx/gemma4-31b-assistant-config.json) is copied from [Google's pinned revision](https://huggingface.co/google/gemma-4-31B-it-assistant/raw/627c5ec1458b9086b841a91e0512fd31fd2fbbf1/config.json).
Its parameter ledger totals 469,518,596 parameters, matching the official model metadata.
The assistant has hidden width 1,024, three sliding-window layers and one global-attention layer.
It reads the target model's existing KV states and receives target embeddings plus recurrent hidden state.
The [reference implementation](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4_assistant/modeling_gemma4_assistant.py) applies a 10,752-to-1,024 pre-projection and a 1,024-to-5,376 post-projection.
The assistant's own tied embedding/output matrix has width 1,024 and differs from the target's width-5,376 embedding matrix.
All 262,144 assistant vocabulary entries are evaluated.
The pinned checkpoint disables centroid-based vocabulary masking.

| Parameter group | Parameters |
| --- | ---: |
| Tied assistant embedding and LM head | 268,435,456 |
| Attention Q and O projections | 83,886,080 |
| Feed-forward matrices | 100,663,296 |
| Pre-projection | 11,010,048 |
| Post-projection | 5,505,024 |
| Layer norms, final norm, query norms and layer scalars | 18,692 |
| Total | 469,518,596 |

The memory ledger shards matrix weights and replicates normalization weights across assistant tensor-parallel ranks.
BF16 matrices require two assistant chips in these probes; FP8 matrices fit on one.
All normalization weights and activations remain BF16, and the shared target KV remains FP8.
Three independent scratch slots share the same assistant weights.
Each slot holds its arena through actual draft completion, including queueing and communication.
The ledger separately charges communication scratch, per-request recurrent feedback state and a 5,000,000-byte reserve.
Each transformer chip receives an additional 1,048,576 bytes of reserve per assistant slot for remote-attention scratch.
Charging that allocation to every transformer chip is conservative because shared-cache attention uses the final stage.

## Explicit execution

The target uses TP16/PP10/DCP4 with six transformer layers per stage.
Its three tied embedding/output chips remain separate from the assistant pool.
The scratch policy reserves K+1 verification query rows for K draft tokens.
The native memory allocator admits nine requests in the supplied probes, with target KV reserved through 102,000 tokens.

Each assistant step looks up the proposed token's target embedding and sends it to the assistant pool.
The assistant executes its pre-projection, four Q/O attention blocks, four feed-forward blocks, final normalization, recurrent post-projection, full-vocabulary LM head and distributed greedy selection.
Query states travel to the existing target KV owners.
The three sliding-window layers read target layer 58; global attention reads target layer 59.
QK, softmax and AV execute on those target chips and contend with target verification on the same physical compute resources.
Global attention returns FP32 partial outputs and log-sum-exp states for merging at the assistant pool.
The program sends query states and partial outputs across the fabric instead of transferring the full KV cache.

The target verifies K+1 query rows with its native transformer operator graph and complete target LM head.
The feedback path copies all verifier hidden rows to the assistant, prices vector selection and retains one accepted recurrent row per request.
Each draft step includes 4 us of serialized host service, and target verification retains its native launch service.
External control, physical synchronization, prefill and stochastic sampling remain uncalibrated or excluded.
The source programs are [assistant operators](../scripts/lpx/assistant_pool.py) and the [experiment driver](../scripts/search_lpx_assistant.py).

## Acceptance-conditional replay

For an independent per-position acceptance probability alpha, the expected committed length is:

```text
expected_committed = 1 + alpha + ... + alpha^K
aggregate_TPS = completed_verification_rounds_per_second * expected_committed
MFU = aggregate_TPS * useful_target_matmul_FLOPs_per_token / all_allocated_FP8_peak
```

Every allocated transformer, target-head and assistant chip contributes to the denominator.
Draft work, rejected positions and duplicated operations receive no extra useful-work credit.
The current useful target work is 129.261109248 GFLOP per committed token at the representative 101,000-token context.
An empirical mean committed length can replace the independent-acceptance formula without assuming independence.

The table uses two warm-up rounds and eight measured rounds for each nine-request deployment.
The last column reports the relative throughput change between measurement halves.
It indicates finite-window convergence, not hardware uncertainty.

| K | Assistant matrix dtype / TP | Total chips | Verification rounds/s | MFU at alpha=0.7 | MFU at alpha=0.9 | MFU at alpha=0.95 | Break-even alpha | Half-window change |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | BF16 / 2 | 165 | 33,649.5 | 3.6420% | 4.0705% | 4.1776% | 41.86% | -0.288% |
| 3 | FP8 / 1 | 164 | 22,050.2 | 3.5777% | 4.8573% | 5.2399% | 59.25% | -0.429% |
| 1 | FP8 / 1 | 164 | 34,045.6 | 3.7073% | 4.1435% | 4.2525% | 39.36% | 0.711% |
| 3 | BF16 / 2 | 165 | 21,644.0 | 3.4905% | 4.7390% | 5.1122% | 60.89% | -2.368% |

Break-even alpha is the independent acceptance probability required to exceed the non-speculative 3.03914% incumbent at the measured round rate.
The initial BF16 K=3 probe has appreciable measurement-half drift.
Longer phase/pacing probes and K=7 observations are recorded below; the initial table remains a finite-window observation.
No acceptance-dependent row replaces the non-speculative incumbent without a representative acceptance measurement.
FP8 assistant quantization can change acceptance and therefore needs its own measurements.
Native target-stage scratch lifetimes, physical bank layout and executable compiler schedules also remain unverified.

Two longer FP8 K=3 pacing probes discard four warm-up rounds and measure sixteen rounds.
They use the same frozen program snapshots and nine resident requests.

| Issue interval (us) | Verification rounds/s | Conditional TPS at alpha=0.9 | Conditional MFU at alpha=0.9 | Half-window change |
| ---: | ---: | ---: | ---: | ---: |
| 43.0 | 22,371.2 | 76,934.5 | 4.9280% | 0.761% |
| 45.5 | 21,917.2 | 75,373.4 | 4.8280% | 0.743% |

The 43 us point improves the observed conditional rate, while its measurement-half difference still requires convergence checks.
The BF16 K=3 pacing probes also finish: 44 us gives 21,669.0 verification rounds/s with 1.729% half-window change, and 46 us gives 21,521.3 rounds/s with 0.233% change.
A K=3 FP8 run with four scratch slots and nine requests uses an uneven fixed round-robin mapping.
Its eight-warm-up, 32-measured-round replay reaches 22,466.6 rounds/s at a 42 us issue interval, with 0.0473% half-window change.
At alpha=0.9 this corresponds to a conditional 4.94906% MFU.
The default K=7 FP8 probe reaches 11,166.3 rounds/s.
A 70 us pacing control reaches 11,194.1 rounds/s, with -0.00364% half-window change and conditional 4.08377% MFU at alpha=0.9.
These K=7 observations remain below the supplied K=3 observations at that acceptance assumption, and do not exclude other K=7 placements or schedules.
A K=4 FP8 control at 48 us reaches conditional 4.93747% MFU at alpha=0.9, with 0.490% half-window change.
The narrow K=3/K=4 difference needs further convergence and acceptance evidence.

## Measured 100K acceptance already in the repository

The earlier statement that no measured 100K acceptance was available omitted existing repository evidence.
The [completed GPU measurements](lpx-gpu-measurements.md#completed-fp8-comparison) include all 16 books at both document question styles, without capped, empty or repetition-flagged outputs.
The [FP8 target raw turns](../experiments/lpx/gpu-triton-fp8-mtp-k3-outputs.jsonl) use a BF16 assistant and BF16 KV.
The [counter audit driver](../scripts/analyze_lpx_mfu_acceptance.py) recomputes acceptance, emitted-step identities, terminal trimming, preemptions and output-quality flags from those raw turns.
It checks source hashes against companion coverage receipts where available.
The audit retains flagged turns and keeps a turn mean distinct from a verification-round-weighted mean.

| GPU target / assistant / KV | K | 100K style | Turns | Round-weighted committed length |
| --- | ---: | --- | ---: | ---: |
| FP8 / BF16 / BF16 | 3 | Quotes | 16 | 2.770602 |
| FP8 / BF16 / BF16 | 3 | Summary | 16 | 2.698749 |
| BF16 / BF16 / BF16 | 1 | Quotes | 16 | 1.805067 |
| BF16 / BF16 / BF16 | 1 | Summary | 16 | 1.783099 |
| BF16 / BF16 / BF16 | 7 | Quotes | 16 | 3.539381 |
| BF16 / BF16 / BF16 | 7 | Summary | 16 | 3.313931 |

Ten audited groups also retain the BF16 K=3 and K=5 measurements.
Every audited group contains all 16 books and passes the recorded output-quality and counter checks.
The current closest-precision K=3 LP30 projections match the FP8 target and BF16 assistant, while changing KV from measured BF16 to modeled FP8.
Using the measured round-weighted lengths gives:

| BF16 assistant issue interval (us) | Quotes projected TPS / MFU | Summary projected TPS / MFU | Half-window change |
| ---: | --- | --- | ---: |
| 44 | 60,036.0 / 3.82231% | 58,479.1 / 3.72318% | 1.729% |
| 46 | 59,627.0 / 3.79626% | 58,080.6 / 3.69781% | 0.233% |

These values are transports of measured GPU acceptance onto a fixed LP30 timing DAG, with an explicit KV-precision mismatch.
They are not measured LP30 throughput.
The higher 4.95% FP8-assistant figure uses alpha=0.9 and additionally changes assistant precision relative to these measurements.
The empirical document results do not establish that assumption.
FP8-assistant acceptance, FP8-KV acceptance and full request scheduling remain validation requirements.

Two additional four-slot BF16 controls finish after the milestone commit.
K=3 at a 43 us issue interval reaches 21,246.3 rounds/s with -1.852% half-window change.
Transporting the measured FP8-target K=3 lengths gives 3.74775% MFU for quotes and 3.65055% for summaries, with the KV-precision mismatch retained.
K=5 at its default 55.517220 us interval reaches 14,872.4 rounds/s with -0.484% change.
Its available BF16-target accepted lengths give 3.14205% and 2.99148% MFU, with both target-weight and KV-precision mismatches.
These controls do not improve the supplied empirical-acceptance transport comparison.

The [vLLM, AMD and Embedded LLM primary report](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus) records Gemma 31B assistant mean committed lengths from 1.91 to 1.96 at K=1 and from 3.39 to 3.70 at K=3 across its listed tasks.
Its example Gemma serving commands use a 32,768-token maximum model length on different hardware and software.
Those results motivate acceptance experiments and do not supply 100K LP30 acceptance.
Its per-position values describe accepted-prefix survival; adding them to one yields mean committed length.
Treating them as independent conditional probabilities and multiplying them would misread that table.

## Reproduction

The default assistant configuration is the bundled pinned file.
The CLI supports assistant matrix precision, tensor width, scratch slots, draft length, resident requests and issue interval.
The cache-owner mapping currently requires the official four-layer assistant and a six-layers-per-stage PP10 target layout.
The supplied probes use TP16/DCP4.
Custom tensor widths must preserve the shared-cache head geometry.
The remote scratch calculation includes FP8 cache tiles, BF16 conversions, score rows, queries and partial outputs, rounded to whole MiB per slot.

```bash
python scripts/search_lpx_assistant.py \
  --draft-tokens 3 \
  --draft-dtype bf16 \
  --draft-tp 2 \
  --slots 3 \
  --resident-requests 9 \
  --warmup-rounds 4 \
  --measured-rounds 16 \
  --output outputs/lpx-maximum-mfu/assistant-k3
```

Each result records assistant configuration hashes, parameter counts, rank memory ledgers, native transformer admission, per-cohort completion counts, convergence and conditional scenarios.
Gzip program snapshots preserve every dependency, physical resource and arena lifetime with input and node-body fingerprints.
The [portable search record](../experiments/lpx/maximum-mfu-search-progress.json) retains compact receipts and current finite search coverage.
