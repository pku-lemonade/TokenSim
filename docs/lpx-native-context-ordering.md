# Native Gemma context-ordering sensitivity

## Research objective and decision criteria

The research objective is to estimate the smallest feasible LP30 deployment and its parallel configuration that can reproduce the reported generation performance under an evidence-supported execution model.
The search must report chip counts, tensor parallelism (TP), pipeline parallelism (PP), precision, memory feasibility, topology, and the measurement boundary.
The heterogeneous candidate uses attention/feed-forward disaggregation (AF) with only global attention on Rubin.
Its LP30 and Rubin inventories must be reported separately.
Decode-pool inventory and the hardware required for a complete prefill-capable deployment must also be distinguished.

The 10K/100K ordering is a model diagnostic alongside absolute throughput.
An unmeasured per-request pause or an independently fitted parameter at each context length cannot establish the required chip count.
The ordering alone cannot establish which physical mechanism caused the published difference.
Unknown hardware and serving parameters require explicit sensitivity ranges, with resulting inventory estimates labeled as conditional.
If endpoint throughput is used to fit a parameter, that same endpoint cannot provide independent validation of the fit.

The earlier [global-attention-only search](lpx-native-global-attention-af.md) selected the configuration with the highest minimum throughput across the tested lengths.
Its FP8 pure-LPX candidate uses 112 LP30 chips at TP16/PP7 and predicts 3,538.14 and 3,417.07 tokens/s at 10K and 100K under the isolated composition.
This candidate does not establish the minimum inventory meeting a specified performance target.
The [native execution comparison](lpx-native-gemma-af-search.md) records substantial latency differences between isolated operator composition and the complete execution graph.
The contribution of each modeled dependency, transfer, collective, and invocation cost must be justified before either execution path can support a reliable minimum-card conclusion.
The inventory search must then minimize resources subject to throughput and feasibility constraints, using a consistent execution model and declared uncertainty assumptions.

## Scope

This experiment asks which parameter changes can make generation at 100K input faster than generation at 10K input.
It uses native `gemma-4-31b-text` operators on `lpx-experiment`, FP8 weights and KV cache, BF16 activations, and tokenizer ratios of one.
The pure-LPX reference has 256 LP30 chips at tensor parallelism (TP) 16 and pipeline parallelism (PP) 16.
The global-only attention/feed-forward disaggregation (AF) reference has 64 LP30 chips at TP16/PP4 and 8 Rubin GPUs at TP8/PP1.
Only the ten global-attention sublayers execute on Rubin in that reference.
The [placement report](lpx-native-global-attention-af.md) documents the native-operator prototype and its limitations.

Input/output lengths remain 10,000/1,500 and 100,000/2,000 tokens.
Generation throughput excludes time to the first token (TTFT).
The baseline pure-LPX rates are 3,531.55 and 3,410.92 tokens/s.
The baseline global-only AF rates are 2,463.25 and 2,161.72 tokens/s.
The fitted reversal ratio is `3431 / 3382 = 1.014488`, so 100K is approximately 1.45% faster than 10K.
This ratio is a numerical target inherited from the earlier study, and it does not provide new hardware evidence.

## Shared static parameters

The scan evaluates 100 parameter profiles across both placements, producing 200 cases with zero reversals.
Each profile applies exactly the same settings at both input lengths.

| Parameter | Tested values |
| --- | --- |
| GEMM compute efficiency, independently per device | 0.1, 0.3, 0.6, 0.8, 1 |
| Attention compute efficiency, independently per device | 0.05, 0.1, 0.3, 0.6, 0.8, 1 |
| Memory bandwidth efficiency, independently per device | 0.1, 0.3, 0.6, 0.9, 1 |
| Effective operator startup, independently per device | 0, 0.05, 0.1, 0.2, 0.5, 1 microseconds |
| Collective startup, independently per device | 0, 0.1, 0.5, 1, 2, 5 microseconds |
| Memory bandwidth multiplier, independently per device | 0.25, 0.5, 2, 4, 16 |
| Peak compute multiplier, independently per device | 0.25, 0.5, 2, 4, 16 |
| Link propagation latency multiplier | 0, 0.1, 0.5, 2, 5 |
| Link bandwidth multiplier | 0.25, 0.5, 2, 4, 16 |
| Cross-device startup | 0, 0.1, 0.5, 1, 2, 5, 10 microseconds |
| Global-attention-only memory efficiency | 0.005, 0.01, 0.05, 0.1, 0.3, 1 |

Zero startup and above-catalog compute or bandwidth values are diagnostic controls.
The scan does not claim those controls are deployable hardware configurations.
The fixed layouts remain resident because these controls do not change stored tensor sizes.

The current decoder sums context-independent operators and nondecreasing attention costs.
Local attention is already bounded by its 1,024-token window at both tested input lengths.
Global-attention cost grows with context.
Within this fixed-layout cost model, changing a shared nonnegative compute efficiency, bandwidth, or startup constant cannot make 100K faster than 10K.
The script checks that even the earliest sampled 100K interval is no faster than the latest sampled 10K interval for every fixed profile.

## Length-dependent numerical hypotheses

The following cases deliberately retain the 10K baseline and change only the indicated parameter at 100K.
The table targets a 1.45% throughput reversal.

| Placement | Effective parameter | 10K value | Required 100K value | Predicted 10K / 100K tokens/s |
| --- | --- | ---: | ---: | --- |
| Pure LPX | LPX operator startup | 0.200000 microseconds | 0.185387 microseconds | 3531.55 / 3582.71 |
| Pure LPX | LPX collective startup | 0.500000 microseconds | 0.382849 microseconds | 3531.55 / 3582.71 |
| Global-only AF | LPX operator startup | 0.200000 microseconds | 0.127581 microseconds | 2463.25 / 2498.94 |

For pure LPX, the operator-startup change is a 7.31% reduction and the collective-startup change is a 23.43% reduction.
For global-only AF, the LPX operator-startup change is a 36.21% reduction.
These hypotheses require a context-dependent execution schedule or effective service cost.
Setting either startup to one shared value at both lengths does not produce the reversal.
The model provides no measurements showing that longer context causes these changes.

Pure LPX reaches equal throughput when only its 100K operator startup decreases from 0.2 to 0.189590 microseconds.
A smaller value produces a strict reversal.
The corresponding equality threshold for only the 100K collective startup is 0.416549 microseconds.
Global-only AF reaches equal throughput when only the 100K LPX operator startup falls to 0.134307 microseconds.

The native pure-LPX composition contains 962 modeled operator-startup terms per generated token.
The global-only AF composition contains 862 such terms on LPX and 100 on Rubin.
These counts explain the numerical sensitivity and do not measure physical device launch events.

Two independent endpoint fits can also match the absolute numerical targets of 3,382 and 3,431 tokens/s.
The fits change only the LPX effective operator startup between lengths.

| Placement | Effective startup at 10K | Effective startup at 100K | Fitted 10K / 100K tokens/s |
| --- | ---: | ---: | --- |
| Pure LPX | 0.213016 microseconds | 0.198216 microseconds | 3382 / 3431 |
| Global-only AF | 0.072059 microseconds | 0.001468 microseconds | 3382 / 3431 |

The pure-LPX endpoints differ by approximately fifteen nanoseconds per modeled operator.
The AF endpoints require a much larger reduction, with the 100K LPX startup approaching zero.
Exact context-by-context generation replays reproduce these numerical fits.
Fitting two independent endpoint parameters to two target values is not independent model validation.

Several bounded changes fail even when applied only to 100K.
Pure LPX with attention compute efficiency raised from 0.6 to 1 reaches only 3,453.74 tokens/s, below its unchanged 10K rate of 3,531.55.
Pure LPX with memory efficiency raised from 0.9 to 1 reaches only 3,427.84 tokens/s.
Raising its GEMM compute efficiency from 0.8 to 1 does not change throughput under this memory-bound decode composition.
Global-only AF with cross-device propagation latency reduced from 2 microseconds to zero reaches only 2,366.33 tokens/s, below its unchanged 10K rate of 2,463.25.
Zeroing only the LPX collective startup reaches 2,453.41 tokens/s for global-only AF and still falls short.

Another constructed reversal reduces global-attention bandwidth efficiency only at 10K.
Pure LPX requires 10K efficiency 0.047507 versus 100K efficiency 0.9, producing 3,362.21 versus 3,410.92 tokens/s.
Global-only AF requires Rubin global-attention efficiency 0.081879 at 10K versus 0.85 at 100K, producing 2,130.84 versus 2,161.72 tokens/s.
These approximately nineteenfold and tenfold efficiency differences are much larger than the effective-startup adjustments.
They require independent evidence before they can explain a hardware measurement.

## One identical per-request pause inside generation

A new shared per-request pause after the first token can reverse measured rates because the two workloads use different output lengths.
This hypothesis uses the same pause at both lengths and does not require a context-dependent hardware parameter.
The pause occurs once per request, with a repetition count independent of the number of generated tokens.
The 10K workload has 1,499 measured generation intervals, and the 100K workload has 1,999.
The longer output therefore amortizes the same pause over more intervals.

```text
measured_generation_time = sum(decode_intervals) + post_first_token_pause
measured_generation_tps = (output_tokens - 1) / measured_generation_time
mean_interval_with_pause = mean_decode_interval + post_first_token_pause / (output_tokens - 1)
```

| Placement | Pause at equal throughput | Pause for a 1.45% reversal | Predicted 10K / 100K tokens/s at the reversal |
| --- | ---: | ---: | --- |
| Pure LPX | 60.015 milliseconds | 89.352 milliseconds | 2917.41 / 2959.68 |
| Global-only AF | 339.368 milliseconds | 396.770 milliseconds | 1491.07 / 1512.68 |

A pause greater than the equality threshold makes 100K faster under the assumed unequal output lengths.
The same pause before the first output token is excluded from generation throughput and cannot cause this effect.
An equal-output-length control removes the reversal because the pause is then amortized over the same number of intervals.
This mechanism does not preserve the approximately 3,400 tokens/s absolute rates without additional parameter changes.

Per-token operator startup, collective startup, and recurring AF transfers are paid repeatedly during decoding.
Applying the same per-token cost at both lengths cannot create this output-length amortization effect in the current fixed-layout model.
An AF handoff could contribute to the proposed pause only if it introduces an additional one-time delay inside the measured generation window.

This per-request pause is an analytical hypothesis in the local sensitivity experiment.
The native decoder does not automatically inject the pause, and no production configuration or hardware catalog parameter was changed to add it.
No hardware measurement currently establishes the existence, timing, or magnitude of such a pause.
These thresholds describe a possible measurement mechanism and do not verify the cause of the published 10K/100K ordering.

## Interpretation and validation

Effective operator startup is the smallest relative adjustment among the tested context-dependent parameters that reproduces the target ordering.
The result identifies a sensitive assumption in this simulator and does not establish a physical explanation for the published ordering.
The present evidence cannot distinguish a scheduling change, a measurement-window effect, or differences between the actual request sets.
Whole-model fixed defaults remain unchanged.

The startup reversal candidates execute every decode context in five warmup and fifty measured serial generation replays.
Exact replay means agree with the parameter-search predictions at relative tolerance `1e-9`.
The script asserts fixed-profile ordering, bounded-fit results, equal operation counts across lengths, and the output-length pause control.
The experiment reuses native operator formulas and the previously validated homogeneous decomposition.

```bash
.venv/bin/python \
  outputs/lpx-native-global-af-20261020/context_ordering.py
```

Results are saved to `outputs/lpx-native-global-af-20261020/context-ordering.json`.
The source prototype and data are local experiment artifacts, and no production source or hardware catalog file is modified.
