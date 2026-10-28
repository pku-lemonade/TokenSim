# Per-step cost audit of the closest AA setup

Record date: 2026-10-28.
The [article-condition sweep](lpx-article-sweep.md) found one 256-chip layout close to both Artificial Analysis (AA) speeds: TP32/PP8 with decode context parallelism (DCP) of 8 at attention matrix efficiency 0.30 or more.
At efficiency 0.50 its decode step takes 267.26 µs at the AA 10K context and 281.46 µs at the AA 100K context.
At the 1.006 Gemma tokens per AA token measured on AA-like book answers in the [API-outputs record](lpx-api-outputs.md), both AA values fall within 5% only if each step takes 13 to 24 µs longer.
The [tray-latency record](lpx-tray-latency.md) showed that a cross-tray latency of about 0.80 to 0.85 µs instead of 0.75 µs supplies that time.
This record lists every per-step cost of that step, finds the costs that the sweep sets to zero or leaves out, and measures how much time each would add.

## Inventory of one decode step

One native decode step at the 10K context, captured with the sweep's settings, holds 8,515 events.
Its critical path, the chain of latest-finishing predecessors from the last event, holds 393 events.
The inventory below was taken before the expectations were recorded, so the expectations use its counts.

| Cost | In the step | On the critical path | Setting in the sweep |
| --- | --- | --- | --- |
| Host submission of the step | Yes, one `cuda_graph_submit` event | Yes, a 4.00 µs wait before the embedding lookup | Included |
| Embedding lookup and its all-reduce | Yes | Yes | Included |
| 60 layers of attention, normalization, feed-forward network (FFN) and residual work | Yes | Yes, 254 compute operators including the head | Included |
| 121 all-reduces, 10 DCP all-to-all combines and one logits all-gather | Yes | Yes, 132 collectives | Included, with the collective launch cost set to 0 µs |
| 7 handoffs between pipeline stages | Yes | Yes | Included |
| Final normalization, language-model head and token selection (sampling) | Yes | Yes | Included |
| Per-operator instruction floor | Priced per operator when nonzero | Would apply to each compute operator | 0 µs; the robustness and placement records use 0.05 µs |
| One invocation charge per step | Priced per captured step when nonzero | Would gate the whole step | 0 µs; the robustness and placement records use 1 µs |
| Collective launch | Priced per collective when nonzero | Would apply to each collective | 0 µs; the LP30 catalog value is 0.5 µs |
| Return of the selected token from the last pipeline stage to the first | No | Not modeled; the step ends at token selection on stage 8 | Missing |
| Delivery of the token to the host for streaming | No | Not modeled | Missing; it delays the next step only if the host must receive each token before submitting the next step |

The step therefore has three explicit fixed-cost settings at zero and one device transfer that it omits.
The [latency ledger](lpx-latency-ledger.md) describes the 0.5 µs collective launch and the per-operator startup as unmeasured assumptions.

## Method

[`scripts/audit_lpx_step_costs.py`](../scripts/audit_lpx_step_costs.py) rebuilds the closest setup with the sweep's configuration, fabric and DCP settings at attention efficiency 0.50.
It captures the decode step at the AA 10K and 100K contexts while it varies one setting at a time:

| Setting | Values |
| --- | --- |
| Collective launch | 0, 0.1, 0.25 and 0.5 µs |
| Per-operator instruction floor | 0, 0.025 and 0.05 µs |
| Invocation charge per step | 0 and 1 µs |

It also captures the combination used by the robustness and placement records, an instruction floor of 0.05 µs and an invocation of 1 µs with zero collective launch, and that combination with the catalog launch of 0.5 µs.
For each point it reports the step, its growth over the sweep's step, the AA errors at 1.006 Gemma tokens per AA token and the extra time per step that would still be needed.
For the omitted token return it reports the service time of the seven pipeline handoffs on the critical path and the fabric tier between the first and the last stage.

## Expectations recorded before the run

1. Each microsecond of collective launch adds 125 to 140 µs to the step at both contexts, with the two contexts within 1 µs of each other, so a launch of about 0.10 to 0.19 µs supplies the 13 to 24 µs.
2. Each microsecond of instruction floor adds 230 to 400 µs to the step, so a floor of 0.05 µs adds 11.5 to 20 µs.
3. An invocation charge of 1 µs adds 1.0 µs to the step.
4. With a floor of 0.05 µs, an invocation of 1 µs and zero launch, both AA values fall within 5% without extra time.
5. With the catalog launch of 0.5 µs added to that combination, both AA values are more than 5% too slow.
6. Returning the token from the last pipeline stage to the first costs 0.3 to 1.3 µs, less than 10% of the 13 to 24 µs.

## Results

The run took 44 seconds with a peak resident memory of 153 MB.
[`experiments/lpx/step-cost-audit-results.json`](../experiments/lpx/step-cost-audit-results.json) holds every step, the inventory and the stage placement.

```bash
.venv/bin/python scripts/audit_lpx_step_costs.py \
  --summary experiments/lpx/step-cost-audit-results.json
```

The AA columns count 1.006 Gemma tokens per AA token.
The last column gives the constant extra time per step that would still place both AA values within 5%.

| Point | Collective launch | Instruction floor | Invocation | 10K step | 100K step | Growth | AA 10K | AA 100K | Extra time needed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Sweep | 0 | 0 | 0 | 267.26 µs | 281.46 µs | — | 3,719 (+10.0%) | 3,532 (+2.9%) | 12.7 to 23.5 µs |
| Launch 0.1 | 0.1 µs | 0 | 0 | 280.46 µs | 294.66 µs | 13.2 µs | 3,544 (+4.8%) | 3,374 (−1.7%) | 0 to 10.3 µs |
| Launch 0.25 | 0.25 µs | 0 | 0 | 300.26 µs | 314.46 µs | 33.0 µs | 3,311 (−2.1%) | 3,161 (−7.9%) | No match |
| Launch 0.5 | 0.5 µs | 0 | 0 | 333.26 µs | 347.46 µs | 66.0 µs | 2,983 (−11.8%) | 2,861 (−16.6%) | No match |
| Floor 0.025 | 0 | 0.025 µs | 0 | 293.16 µs | 307.36 µs | 25.9 µs | 3,391 (+0.3%) | 3,234 (−5.7%) | No match |
| Floor 0.05 | 0 | 0.05 µs | 0 | 319.06 µs | 333.26 µs | 51.8 µs | 3,116 (−7.9%) | 2,983 (−13.1%) | No match |
| Invocation 1 | 0 | 0 | 1 µs | 268.26 µs | 282.46 µs | 1.0 µs | 3,706 (+9.6%) | 3,519 (+2.6%) | 11.7 to 22.5 µs |
| Robustness settings | 0 | 0.05 µs | 1 µs | 320.06 µs | 334.26 µs | 52.8 µs | 3,106 (−8.2%) | 2,974 (−13.3%) | No match |
| Robustness settings and catalog launch | 0.5 µs | 0.05 µs | 1 µs | 386.06 µs | 400.26 µs | 118.8 µs | 2,575 (−23.9%) | 2,483 (−27.6%) | No match |

Every setting adds the same time at 10K and 100K.

The seven pipeline handoffs on the critical path each take 0.359 µs.
Stage 1 holds global ranks 0 to 31 and stage 8 ranks 224 to 255.
If ranks map to chips in order, the first and the last stage lie in different 128-chip groups of the fabric's middle tier and communicate over the 1.15 µs rack tier.
The handoff between stages 4 and 5 also crosses that boundary and takes the same 0.359 µs, because the model prices a handoff as one chip-to-chip hop of 0.35 µs plus wire time, without the tier latency.

1. Held: each microsecond of collective launch adds 132 µs at both contexts, one per collective on the critical path.
   A launch of 0.096 to 0.178 µs supplies the 12.7 to 23.5 µs.
2. Missed: each microsecond of instruction floor adds 1,036 µs, against the expected 230 to 400 µs.
   The floor applies to every priced kernel component of an operator, and the critical path holds 1,036 components, such as 470 in the 60 attention operators, against 254 operators.
   A floor of 0.012 to 0.023 µs supplies the 12.7 to 23.5 µs.
3. Held: an invocation of 1 µs adds 1.0 µs.
4. Missed: with the floor of 0.05 µs and the invocation of 1 µs, the step grows by 52.8 µs and both AA values are 8.2% and 13.3% too slow.
5. Held: adding the catalog launch of 0.5 µs makes them 23.9% and 27.6% too slow.
6. Held: one handoff-priced token return costs 0.36 µs, and a return at the 1.15 µs rack-tier latency about 1.2 µs, which is 1.5% to 9.4% of the 12.7 to 23.5 µs.

## Interpretation

The closest setup runs with three explicit fixed costs at zero.
Small nonzero values of any of them supply the 12.7 to 23.5 µs: a collective launch of about 0.10 to 0.18 µs over 132 collectives, an instruction floor of about 0.012 to 0.023 µs over 1,036 kernel components, or a cross-tray latency of about 0.80 to 0.85 µs from the [tray-latency record](lpx-tray-latency.md).
The gap therefore lies within the uncertainty of fixed-cost assumptions that no public source measures, and it does not require a mechanism outside the model.
The audit cannot tell which cost supplies it.

The same numbers bound the fixed costs that the setup tolerates.
The values used by other records, an instruction floor of 0.05 µs and a collective launch of 0.5 µs, each make the setup more than 5% too slow at 100K, and together they make it 24% to 28% too slow.
The match holds only if LP30's fixed costs total at most about 24 µs per step across 132 collectives and 1,036 kernel components, about 0.02 µs per component or 0.18 µs per collective.
A statically scheduled chip that issues instructions without a runtime launch path could meet that budget, but no LP30 measurement confirms it.

The modeled step leaves out the selected token's return from the last pipeline stage to the first, which adds about 0.4 to 1.2 µs, and the token's delivery to the host.
The 4 µs host submission already sits on the critical path.
A host that had to receive each token before submitting the next step would add its round trip to every step, so the match also requires that loop to cost no more than the remaining budget.
The 2.7 to 2.8 ms host loop of the [GroqRack check](groqrack-llama2-7b-validation.md) would make the setup about ten times too slow, so the LPX deployment cannot run such a loop per token.

Every fixed cost adds the same time at both contexts, so none of them explains AA's 1.45% higher speed at 100K.
