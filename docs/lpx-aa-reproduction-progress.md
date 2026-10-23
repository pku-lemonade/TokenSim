# Progress on reproducing the AA Gemma 4 31B LPX results

Record date: 2026-10-23.
This page summarizes the work to reproduce the Artificial Analysis (AA) Gemma 4 31B results on NVIDIA Groq 3 LPX (LP30) hardware with TokenSim.
Each section links to the detailed record that holds the method, the expectations recorded before each run and the complete results.

## Goal

The published AA endpoint reaches a median output speed of 3,382 tokens per second (TPS) at 10K input tokens and 3,431 TPS at 100K input tokens.
The goal is to reproduce both values, ideally with the smallest LP30 deployment.
The published 100K speed is 1.45% higher than the 10K speed, so the endpoint's speed does not fall with context length.

TP denotes tensor parallelism, PP pipeline parallelism and KVP the Helix-style KV-parallel size, in which several ranks share the KV cache of one global-attention layer.
All results use the bundled [fidelity configuration](../experiments/lpx/fidelity-study.json), whose LP30 unit rates and efficiencies are uncalibrated assumptions.

## Summary of outcomes

| Step | Record | Outcome |
| --- | --- | --- |
| KV parallelism for decode and prefill | [LPX context parallelism](lpx-context-parallel.md) | 100K speed rises approximately 15%; minimum inventory falls from 128 to 96 chips |
| TP8 with KV parallelism | [TP8 section](lpx-context-parallel.md#tp8-with-kv-parallelism) | No gain at TP8; TP16/PP6 on 96 chips remains the smallest layout that clears all targets |
| Placement plausibility | [Placement plausibility record](lpx-placement-plausibility.md) | Six reasons why a dedicated batch-one deployment is unlikely to be the published one |
| Batched serving | [Batched-serving control](lpx-batched-serving.md) | Background traffic explains the 10K level but not the flat context dependence; no match in 156 points |
| Speculative decoding | [Speculative-decoding control](lpx-speculative-decoding.md) | Speculation shared with background traffic matches both AA values within 5% |

## KV parallelism

The simulator now supports Helix-style KV parallelism on LP30 decode and prefill.
On TP16/PP8 with 128 chips, `KVP = 2` changes the speeds at 1K, 10K and 100K from 5,908.78, 5,578.41 and 3,611.17 TPS to 5,759.05, 5,560.63 and 4,167.93 TPS.
The 100K-to-10K speed ratio rises from 0.647 to 0.750, while the published ratio is 1.015.
KV parallelism helps only when vector-unit KV conversion bounds global attention, which holds at TP16 and not at TP8.
TP16/PP6 on 96 chips with KV parallelism is the smallest layout found that clears the 1K, 10K and 100K speed targets.

## Why the dedicated batch-one placement is unlikely

The simulated dedicated deployment runs one request per forward pass without speculative decoding.
It is faster than published at 10K and slower than published at 100K, and its speed always falls with context length.
The [placement plausibility record](lpx-placement-plausibility.md) gives six reasons why this is unlikely to be the published deployment.
One reason is that the same Gemma4-31B-LPX deployment name is reported at 10,996 TPS on a 16K-input coding example at Hot Chips 2026, which one committed token per forward pass cannot reach.
The record also holds the sources and checksums of that slide.

## Batched serving

The [batched-serving control](lpx-batched-serving.md) places the AA request in each decode forward together with up to 15 background requests.
About seven short background requests slow the 10K speed to the published value.
Background requests add the same time at 10K and 100K, so the AA request's own 100K attention increment of 28.6 to 97.7 microseconds remains.
When the 10K speed matches, the 100K speed stays 10% to 26% below the published value.

## Speculative decoding

The [speculative-decoding control](lpx-speculative-decoding.md) prices native verification forwards that check `K` draft tokens plus one target token.
One shared acceptance rate matches the AA pair only when the 100K-to-10K step-time ratio is at most 1.0896.

Speculation at batch size one cannot match the AA pair, because the smallest step-time ratio is 1.154.
Speculation shared with background traffic does match.
One-token drafting with seven background requests under `KVP = 4` places both AA values within 5% at an acceptance of 0.51 to 0.74, on 128 or 256 chips.
Each step then commits 1.50 to 1.75 tokens, so a 428 to 491-microsecond batched step gives the published per-token speed.

The 10,996 TPS coding example needs at least two draft tokens and an acceptance of at least 0.67.
Within the tested limit of 16 query rows per forward, the AA match and the coding example therefore need different draft lengths.
The limit comes from the study's live-scratch reservation and from host memory, because a 32-row forward exceeded 4.5 GB in the streaming refinement.

## Current position

The simulator reproduces both published AA values within 5% under one fitted serving regime: speculative decoding with one draft token, seven background requests and KV parallelism on 128 or 256 chips.
The acceptance rates and background loads are fitted to the published speeds and are not measured.
The match shows that the cost model can produce the published pair under a plausible serving regime, and it does not identify the regime that NVIDIA used.
No result calibrates an LP30 operator cost against measured hardware.

## Open work

| Item | Purpose |
| --- | --- |
| Lift the 16-row forward limit | Test whether one draft length, such as `K = 3` with more background requests, matches both the AA pair and the 10,996 TPS coding example |
| GroqChip v1 validation | Test the operator cost-modeling method against measured results from the previous Groq chip |
| Native token conversion | Replace the assumed one-to-one conversion between Gemma tokens and AA `o200k_base` tokens |

## Commits

The work is on branch `lpx-experiment`, pushed to `origin/lpx-context-parallel`.

| Commit | Content |
| --- | --- |
| `f39fce1` | Context-parallel plan and KV-parallel configuration |
| `befd8a0` | KV-parallel decode and prefill with results |
| `69e767e` | Placement plausibility record |
| `bfa944b` | TP8 KV-parallel confirmation |
| `15e86e9` | Batched-serving control |
| `0aec3d4` | Speculative-decoding control |
