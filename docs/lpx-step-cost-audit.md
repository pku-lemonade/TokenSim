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

`scripts/audit_lpx_step_costs.py` rebuilds the closest setup with the sweep's configuration, fabric and DCP settings at attention efficiency 0.50.
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

Pending.
