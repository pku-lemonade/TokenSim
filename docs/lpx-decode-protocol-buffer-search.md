# Decode protocol buffers and request-slot scheduling

An external-head protocol-buffer projection admits fifteen resident requests on 163 LP30 chips with two active request frames per pipeline stage.
Its longest replay reaches 52,911.881644 aggregate tokens per second (TPS), or 3.410066841% model floating-point operation utilization (MFU).
Fourteen shorter screens vary slot assignment, incoming-buffer count and resource arbitration without exceeding that result.
The comparison reference for these buffer screens is 3.419600500% experimental shared-cache decode MFU, with model quality and physical compiler support unverified.
The later [periodic resource-calendar study](lpx-periodic-resource-calendar-search.md) reaches 3.557996616% with fourteen requests and a larger charged scratch policy.
This study does not certify a global maximum.

## Workload and ownership

The workload uses Gemma 4 31B text decode at a representative context of 101,000 tokens and reserves key-value (KV) storage for 102,000 tokens.
Each of fifteen independent clients produces one committed token per forward.
The transformer uses tensor parallelism (TP) 16, decode context parallelism (DCP) 4 and pipeline parallelism (PP) 10, with six layers per stage.
Its 160 chips share no weights or execution resources with the three-chip tied embedding and output-head pool.
All 163 chips contribute their peak to the MFU denominator.
The hardware constraint requires pure LP30 and imposes no minimum TPS.
Weights and the experimental shared global cache use eight-bit floating point (FP8), while activations use bfloat16 (BF16).
The shared-cache representation retains its existing quality caveat.
Prefill and request replacement remain outside these resident-prefix decode measurements.
The [complete-request serving study](lpx-complete-serving-mfu.md) reports its separate prompt-plus-output metric.

```text
useful_FLOP_per_token = 129261109248
peak_FLOP_per_chip_per_s = 315e15 / 256
MFU_percent = 100 * aggregate_TPS * useful_FLOP_per_token / (163 * peak_FLOP_per_chip_per_s)
```

## Protocol storage projection

The external-head setup backcharges full-vocabulary communication storage to every transformer chip.
The extra charge is `4 * (262144 - 5376) = 1027072` bytes per transformer chip.
The projection removes this extra charge after setup and retains the native hidden-row protocol workspace of 21,504 bytes per transformer chip.
The complete vocabulary, matrix weights, logits computation, distributed greedy selection and head communication storage remain in the three-chip head pool.
Each head chip retains 1,048,576 communication bytes and the 5,000,000-byte safety reserve.
The largest modeled head allocation is 476,392,516 bytes.
The repository's default memory policy remains unchanged.

The protocol audit records 305 positive-payload native communication events before boundary replacement.
These comprise one embedding reduction, sixty attention reductions, sixty feed-forward reductions, forty global-attention combines and 144 pipeline handoffs.
The native embedding reduction is elided when the external embedding program replaces it, leaving 304 such transformer events in the composed graph.
Ordinary protocol events require at most 21,504 bytes for two payload copies.
Global-attention combines require up to 32,832 bytes for two copies and retain their separately charged per-frame attention arrays.
The audit preserves each event's payload, wire type, buffer owner and endpoints.

The [frame-workspace study](lpx-frame-workspace-search.md) supplies the persistent-frame geometry used here.
Each active frame reserves 366,080 bytes per transformer chip.
This includes 78,464 bytes of attention state, 29,568 bytes of dense feed-forward state, sixteen BF16 hidden rows and four additional full-precision transport rows.
The native shared scratch allocation of 2,179,712 bytes remains enabled.
Each incoming slot adds one 10,752-byte BF16 hidden row.
The active-frame lease spans its pipeline handoff or head copy, and copy completion precedes slot release.
The head pool uses an exclusive boundary arena.
These accounting and lifetime constraints remain a compiler-unverified projection.

| Active slots per stage | Incoming slots per stage | Additional reservation per transformer chip | Minimum headroom after KV watermark | Native resident admission |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 753,664 bytes | 31,636 bytes | 15 requests |
| 2 | 3 | 764,416 bytes | 20,884 bytes | 15 requests |
| 2 | 4 | 775,168 bytes | 10,132 bytes | 15 requests |

The two-active/two-incoming policy sets the transformer reserve to 5,753,664 bytes, including the unchanged 5,000,000-byte safety allocation.
Native block rounding and the one-percent KV watermark remain enabled.
The watermark-adjusted headroom determines capacity for additional buffers.
The larger free-byte value in the raw resident report includes unused watermark space and cannot all be assigned to those buffers.
Two active frames and five incoming rows exceed this supplied footprint budget by 620 bytes.
That five-row result is a footprint calculation, without a separate native graph capture.

The unchanged vocabulary-backcharge policy admits fourteen requests.
A capacity sensitivity calculation requires a uniform discount of at least 241,772 bytes per chip to admit fifteen under the supplied partition.
Removing the 1,027,072-byte charge and adding 753,664 bytes of frame storage gives a net discount of 273,408 bytes.
The difference of 31,636 bytes agrees with the watermark-adjusted native footprint.

## Scheduling controls

All screens preserve the native operator dependencies, costs, physical resources and token feedback.
Fixed client affinity selects the same frame profile for each client.
Rotating assignment selects profiles by the global forward-submission sequence and allows one outstanding forward per client.
The driver uses the canonical cohort measurement and feedback logic.
A deterministic control confirms that its fixed-affinity wrapper reproduces the canonical metrics exactly and that rotating submissions select the expected profiles.

| Policy | Incoming slots | Issue factors tested | Screens | MFU range |
| --- | ---: | --- | ---: | ---: |
| Fixed client affinity, repository arbitration | 2 | 1.001, 1.08, 1.16 | 3 | 3.352770815%–3.380594132% |
| Rotating submission slots, repository arbitration | 2 | 1.001, 1.08, 1.16 | 3 | 3.410066841% |
| Rotating submission slots, repository arbitration | 3 | 1.001, 1.08, 1.16 | 3 | 3.410066841% |
| Rotating submission slots, repository arbitration | 4 | 1.001, 1.08, 1.16 | 3 | 3.410066841% |
| Rotating submission slots, private work-conserving arbitration | 2 | 1.001, 1.08 | 2 | 3.410066841% |

Each screen uses four warm-up outputs and sixteen measured outputs per client, totaling sixty warm-up and 240 measured outputs.
Issue factors multiply the isolated forward duration divided by fifteen.
The private arbitration prototype permits a ready node to bypass an older blocked node when all of its required resources are free.
A controlled three-job example confirms that this bypass changes execution as intended.
The full decode graphs receive no throughput gain under the two tested issue intervals.
The repository replay engine retains its existing arbitration policy.
The private engine source and its distinct fingerprint are preserved in the experiment record.

The asymmetric-slot profiles rename only the active and incoming arena resources.
Checks compare all 96,774 node rows per profile and preserve every other node field.
Three incoming slots use six joint profiles, while four incoming slots use four joint profiles.
These profiles rotate the active and incoming leases independently without changing the physical chip resources.

## Long replay and resource occupancy

The longer repository-engine replay uses eight warm-up and 32 measured outputs per client at issue factor 1.001.
It measures 480 output tokens over 9.071686455 milliseconds, with 3,527.458776 TPS per client.
The two measurement halves differ by approximately `1.12e-13` relative.
All 615 submitted forwards complete after draining, including warm-up and forwards outstanding at the measurement boundary.
The measurement window records resource holds that cross its boundaries.

| Resource class | Measured occupancy |
| --- | ---: |
| Exclusive three-chip boundary arena | 40.5731% |
| Active-frame arenas | 61.3582%–75.5720% |
| Incoming-row arenas | 17.7222%–89.0582% |
| Vector execution resources across all chips | 21.3053%–62.9872% |

No measured exclusive resource exceeds full occupancy.
These occupancies describe this periodic schedule and do not establish an achievable global throughput bound.
The head arena's minimum dependency hold is 7.668057 microseconds, compared with an observed aggregate output period of 18.899347 microseconds.
Its measured occupancy does not support treating the head pool as a continuously saturated bottleneck in this schedule.
The fixed-program necessary bound remains 4.259145148% MFU and leaves a gap above the measured result.
That bound permits ideal scheduling and does not demonstrate an attainable execution.

## Evidence and remaining scope

The [compressed experiment record](../experiments/lpx/decode-protocol-buffer-search.json.gz) preserves the protocol audit, current memory policy, native admission receipts, fourteen screens, long replay, resource statistics, scheduling controls and reproduction sources.
It records the older frame ledger as geometry evidence and identifies its historical reserve as superseded for this projection.
Full local graph paths and file fingerprints accompany the portable results.
The [A/F ratio study](lpx-pure-lp30-af-ratio-bounds.md) supplies the current physical-group code identity and incumbent replay controls.

The fifteen-request projection does not improve the 3.419600500% experimental shared-cache decode reference.
Physical SRAM bank placement, compiler scheduling and matching shared-cache quality require validation.
Revised tensor lifetimes, alternate batching, other placements and the complete-request objective remain open search directions.
Earlier batch exclusions retain their supplied memory envelopes and do not automatically cover this protocol-buffer projection.
