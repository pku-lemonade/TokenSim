# Progress on reproducing the AA Gemma 4 31B LPX results

Record date: 2026-10-25.
This page summarizes the work to reproduce the Artificial Analysis (AA) Gemma 4 31B results on NVIDIA Groq 3 LPX (LP30) hardware with TokenSim.
Each section links to the detailed record that holds the method, the expectations recorded before each run and the complete results.

## Goal

The published AA endpoint reaches a median output speed of 3,382 tokens per second (TPS) at 10K input tokens and 3,431 TPS at 100K input tokens.
The goal is to reproduce both values, ideally with the smallest LP30 deployment.
The published 100K speed is 1.45% higher than the 10K speed, so the endpoint's speed does not fall with context length.
A simulated point matches when both speeds lie within 5% of the published values.

TP denotes tensor parallelism and PP pipeline parallelism.
Decode context parallelism (DCP) splits one request's key/value (KV) cache by token position across the chips of a TP group, following NVIDIA's Helix parallelism.
The scripts and earlier records call the DCP size `KVP`.
The attention group is the TP group of `N` chips that computes one global attention layer, factored as `N = TPA * KVP`, where TPA ranks split the four global KV heads.
All results use the bundled [fidelity configuration](../experiments/lpx/fidelity-study.json), whose LP30 unit rates and efficiencies are uncalibrated assumptions.

## Summary of outcomes

| Step | Record | Outcome |
| --- | --- | --- |
| DCP for decode and prefill | [LPX context parallelism](lpx-context-parallel.md) | 100K speed rises approximately 15%; minimum inventory falls from 128 to 96 chips |
| TP8 with DCP | [TP8 section](lpx-context-parallel.md#tp8-with-kv-parallelism) | No gain at TP8; TP16/PP6 on 96 chips remains the smallest layout that clears all targets |
| Placement plausibility | [Placement plausibility record](lpx-placement-plausibility.md) | Six reasons why a dedicated batch-one deployment is unlikely to be the published one |
| Batched serving | [Batched-serving control](lpx-batched-serving.md) | Background traffic explains the 10K level but not the flat context dependence; no match in 156 points |
| Speculative decoding | [Speculative-decoding control](lpx-speculative-decoding.md) | Speculation shared with background traffic matches both AA values within 5% on TP16 |
| Attention-group size | [Attention-group control](lpx-attention-group.md) | TP32 with `KVP = 8` matches without speculation, and with `K = 3` speculation it also reaches the coding example, at attention matrix efficiency 0.30 or more |

## DCP

The simulator supports Helix-style DCP on LP30 decode and prefill.
On TP16/PP8 with 128 chips, `KVP = 2` changes the speeds at 1K, 10K and 100K from 5,908.78, 5,578.41 and 3,611.17 TPS to 5,759.05, 5,560.63 and 4,167.93 TPS.
The 100K-to-10K speed ratio rises from 0.647 to 0.750, while the published ratio is 1.015.
DCP helps only when duplicated global KV heads make vector-unit KV conversion the bound, which holds at TP16 and not at TP8.
TP16/PP6 on 96 chips with DCP is the smallest layout found that clears the 1K, 10K and 100K speed targets.

## Why the dedicated batch-one placement is unlikely

The simulated dedicated deployment runs one request per forward pass without speculative decoding.
It is faster than published at 10K and slower than published at 100K, and its speed always falls with context length.
The [placement plausibility record](lpx-placement-plausibility.md) gives six reasons why this is unlikely to be the published deployment.
One reason is that the same Gemma4-31B-LPX deployment name is reported at 10,996 TPS on a 16K-input coding example at Hot Chips 2026, which one committed token per forward pass cannot reach.
The shared name does not establish that the coding example and the AA runs used the same configuration, so this summary treats the coding example as a consistency check.

## Batched serving

The [batched-serving control](lpx-batched-serving.md) places the AA request in each decode forward together with up to 15 background requests.
About seven short background requests slow the 10K speed to the published value.
Background requests add the same time at 10K and 100K, so the AA request's own 100K attention increment of 28.6 to 97.7 microseconds remains.
When the 10K speed matches, the 100K speed stays 10% to 26% below the published value.
Background requests represent any context-independent time per step, so a match shows how much such time a deployment needs and does not show that other users shared the endpoint.

## Speculative decoding

The [speculative-decoding control](lpx-speculative-decoding.md) prices native verification forwards that check `K` draft tokens plus one target token.
One shared acceptance rate matches the AA pair only when the 100K-to-10K step-time ratio is at most 1.0896.

Speculation at batch size one on TP16 cannot match the AA pair, because the smallest step-time ratio is 1.154.
On TP16, speculation shared with background traffic does match.
One-token drafting with seven background requests under `KVP = 4` places both AA values within 5% at an acceptance of 0.51 to 0.74, on 128 or 256 chips.
Each step then commits 1.50 to 1.75 tokens, so a 428 to 491-microsecond batched step gives the published per-token speed.
The 10,996 TPS coding example needs at least two draft tokens on TP16, so no single TP16 draft length covered both within the tested limit of 16 query rows per forward.

## The remaining 100K cost

A critical-path breakdown of TP16/PP8 with `KVP = 4` at attention matrix efficiency 0.50 places the whole 28.35-microsecond 10K-to-100K increment in the attention core of the 10 global layers.
At 100K each global core spends 3.17 microseconds on the vector unit, mostly converting FP8 KV entries to BF16, and 1.35 microseconds on the matrix unit.
The vector unit therefore bounds the core above efficiency 0.30, which is why higher matrix efficiency did not reduce the increment.
Without duplicated KV heads, each chip's attention work is the layer's work divided by `N`, so DCP inside a fixed TP16 group cannot reduce it further.

## Attention-group size

The [attention-group control](lpx-attention-group.md) doubles `N` with TP32/PP4 on the same 128 chips.
With `KVP = 8` the increment falls to 14.2 microseconds at efficiency 0.30 and 0.50, below the 25.2-microsecond bound that one-token decode allows.
Two short background requests then place both AA values within 5% without speculation, at 3,435 to 3,452 TPS and 3,274 to 3,290 TPS.
TP32/PP8 on 256 chips matches with one 16,000-token background request.
`KVP = 2` and `KVP = 4` on TP32 keep replicated KV heads and do not match, and TP32 without DCP does not fit in SRAM.

On TP32/PP4 with `KVP = 8` at efficiency 0.50, speculation matches at 14 of 17 batched points and at 7 serial points.
`K = 3` matches the AA pair with one or three background requests at an acceptance of 0.18 to 0.41.
Without background and draft cost, the same layout reaches the 10,996 TPS coding example at `K = 3` with an acceptance of 0.870.
One draft length therefore covers both on one layout.
No TP32 point matches at efficiency 0.10, the study value.

## Ablation

The table lists every tested combination and whether it matches both AA values within 5%.

| DCP | Background | Speculation | Attention group | Result |
| --- | --- | --- | --- | --- |
| none | none | none | TP16 | No: 5,578 and 3,611 TPS |
| `KVP = 4` | none | none | TP16 | No: 5,533 and 4,152 TPS |
| none | 7 requests | none | TP16 | No: 100K 2,542 TPS when 10K matches |
| `KVP = 4` | up to 15 requests | none | TP16 | No: best 3,331 and 3,041 TPS |
| none or `KVP = 4` | none | `K` from 1 to 7 | TP16 | No: step-time ratio at least 1.154 |
| none | 1 to 7 requests | `K` of 1, 3 or 7 | TP16 | No: step-time ratio at least 1.249 |
| `KVP = 4` | 7 requests | `K = 1` | TP16 | Yes: 6 points, acceptance 0.51 to 0.74 |
| none | any | none | TP32 | Infeasible: resident KV exceeds SRAM |
| `KVP = 2` or `KVP = 4` | 0 to 7 requests | none | TP32 | No: increment 25.7 µs or more |
| `KVP = 8` | none | none | TP32 | No: 10K speed 12% too high |
| `KVP = 8` | 1 or 2 requests | none | TP32 | Yes: 6 points at efficiency 0.30 or 0.50 |
| `KVP = 8` | none to 7 requests | `K = 1` or `K = 3` | TP32 | Yes: 21 points at efficiency 0.50 |
| `KVP = 8` | any | any | TP32, efficiency 0.10 | No |

Every match uses DCP and some context-independent work per step, from background requests, verification of draft tokens or both.
Speculation is required on TP16 and optional on TP32.
TP32 is required for a match without speculation and depends on an attention matrix efficiency of at least 0.30.

## Current position

The simulator reproduces both published AA values within 5% under several fitted serving regimes on 128 or 256 chips.
Every regime combines DCP with additional context-independent work per step.
On TP32/PP4 with `KVP = 8`, three-token drafting with zero to three background requests matches the AA pair and also reaches the 10,996 TPS coding example.
No regime makes the 100K speed exceed the 10K speed, and every match sits near the lower edge of the 100K band.
The acceptance rates and background loads are fitted to the published speeds and are not measured.
The TP32 regimes also depend on the assumed attention matrix efficiency and on FP8 KV conversion running on the vector unit.
The matches show that the cost model can produce the published pair, and they do not identify the regime that NVIDIA used.
No result calibrates an LP30 operator cost against measured hardware.

## Open work

| Item | Purpose |
| --- | --- |
| GroqChip v1 validation | Test the operator cost-modeling method against measured results from the previous Groq chip |
| LP30 FP8 attention path | Establish whether KV conversion runs on the vector unit, which decides whether a larger attention group helps |
| Measured acceptance | Replace the fitted acceptances with a Gemma drafter run on AA-length prompts and coding prompts |
| Lift the 16-row forward limit | Test longer drafts and heavier background traffic in one forward |
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
| `d90008a` | Progress summary |
