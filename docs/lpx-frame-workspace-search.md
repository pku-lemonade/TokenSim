# Shared attention tiles and persistent request frames

The pure-LP30 search separates temporary attention tiles from state retained by an in-flight pipeline request.
The workload remains one-token decode at a fixed 101,000-token context, with key-value (KV) capacity reserved for 102,000 tokens.
The objective has no minimum throughput and includes every allocated LP30 chip in the model floating-point operation utilization (MFU) denominator.
Weights and the experimental shared global cache use eight-bit floating point (FP8), while activations use bfloat16 (BF16).
Shared-cache model quality and the proposed physical memory allocation remain unverified.

## Storage projection

The earlier [integrated-boundary prototypes](lpx-integrated-boundary-search.md) duplicate the entire native attention scratch arena for an additional active request.
Their additional reservation is 2,190,464 bytes per transformer chip, including an additional dense residual.
The new projection retains the entire original native scratch arena and adds separate persistent frame storage.
Only physical and decoded KV data tiles, assembled key tiles and tile scales share temporary storage between requests.
The captured monolithic attention operation holds its worker resource until its tile use completes.
When the terminal output projection is streamed, all tile use finishes within its monolithic prefix before output-vector production begins.
Queries, scores, online accumulators, partial outputs, log-sum-exp values, append quantization, feed-forward state and transport state remain charged to the owning frame.
No compute duration or native tensor dependency is shortened by this storage projection.

For tensor parallelism 16 and decode context parallelism 4, a global attention operation reserves 2,179,712 bytes of native scratch.
Its reusable data tiles account for 2,101,248 bytes, leaving 78,464 bytes charged to each frame.
The frame budget also retains the complete native dense feed-forward workspace, sixteen full hidden BF16 rows and four additional hidden FP32 transport rows.
These extra rows deliberately preserve independent tensor copies within a frame.
Layers execute sequentially within each request, so a stage reuses its frame allocation across its owned layers.

| Additional allocation per transformer chip | Bytes |
| --- | ---: |
| Attention state per active frame | 78,464 |
| Complete dense feed-forward workspace per active frame | 29,568 |
| Sixteen hidden BF16 rows per active frame | 172,032 |
| Four hidden FP32 transport rows per active frame | 86,016 |
| Total per active frame | 366,080 |
| Two active frames | 732,160 |
| Two incoming hidden BF16 rows | 21,504 |
| Total additional reservation | 753,664 |

The native communication allocation, dense residual, proxy boundary weights, 5,000,000-byte safety reserve, KV block rounding and 1% watermark remain enabled.
The full-vocabulary communication backcharge also remains enabled in the transformer proxy.
The colocated embedding and language-model head retain their complete additional weights, live buffers and conservative communication allocation.
An exclusive head arena protects those tensors while head compute shares the physical transformer worker and vector resources.
Two exclusive active arenas and two exclusive incoming arenas per stage bound the number of retained request frames.
An explicit entry copy completes before an incoming arena is released.
The active arena remains held through its outgoing hidden copy or pipeline handoff.

## Capacity and replay

With a single tied matrix on the final tensor-parallel group, the combined memory ledger admits thirteen resident requests on 160 LP30 chips.
The ten pipeline stages contain `[6, 6, 6, 6, 6, 7, 7, 6, 6, 4]` layers.
Minimum projected memory headroom is 836,914 bytes per chip after every charged reservation.
An independent reconstruction compares every stage and rank against the native block and watermark arithmetic.
Fourteen resident requests have no feasible ten-stage contiguous partition under this storage policy.
The fixed-program necessary MFU bound is 3.826404529%; it supplies no executable hardware performance guarantee.

A separate exact capacity sensitivity check finds that thirteen requests permit at most 1,590,578 additional uniform bytes per chip in this placement family.
The 192-chip deployment with a tied matrix on the initial group permits only 39,862 additional uniform bytes if it must retain seventeen requests.
Both checks reject one additional byte at their stated limits across every allowed contiguous partition at the requested pipeline depth.
The new 753,664-byte policy therefore retains sixteen requests in the 192-chip family.
These thresholds constrain the supplied uniform reservation and ownership rules.
They do not exclude different lifetimes, nonuniform allocations or other request schedules.

The finite screen evaluates nine issue intervals with two active/input slots and two intervals for each of three and four slots.
Each screen discards 52 warm-up outputs and measures 208 committed output tokens.
Initial client phases use the same spacing as the subsequent global issue interval.

| Active/input slots | Additional bytes per chip | Minimum headroom, bytes | Best finite MFU | Issue interval, microseconds |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 753,664 | 836,914 | 3.370783576% | 19.000000000 |
| 3 | 1,130,496 | 460,082 | 3.359516043% | 17.200000000 |
| 4 | 1,507,328 | 83,250 | 3.359516043% | 17.200000000 |

The selected longer control uses 2 active/input slots and an issue interval of 19.000000000 microseconds.
It discards 104 warm-up outputs and measures 416 outputs at 51,335.041357 aggregate tokens per second, corresponding to 3.370475880% MFU.
The relative throughput change between its measurement halves is -0.193021%.
The supplied storage and scheduling controls preserve the existing 3.419600500% experimental shared-cache decode record.
The additional resident request and larger number of frame slots do not supply a universal throughput guarantee.

## Physical scope and evidence

NVIDIA describes compiler scheduling of compute, data movement and SRAM bank conflicts in its [LPX deterministic-execution article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-deterministic-execution-drives-power-efficient-high-interactivity-inference-on-nvidia-vera-rubin/).
That description motivates explicit lifetime and resource checks but does not validate this frame allocation or supply its physical bank placement.
The simulation uses coarse exclusive worker resources, while physical LP30 compilation requires a schedule of the individual units and memory banks.
The storage projection assumes each analytical attention prefix completes its tile reads before releasing its worker.
Finer overlap between attention prefixes would require additional tile arenas or stricter explicit tile lifetimes.

The [compressed experiment record](../experiments/lpx/frame-workspace-search.json.gz) retains the capacity sensitivities, exact reservation thresholds, per-frame allocation ledger, native admission receipts, per-rank combined memory witnesses, all thirteen finite screens and the longer replay.
Each of the two base programs preserves all 96,536 native transformer nodes and their dependency edges.
All non-host native compute durations and resource bindings remain unchanged.
An explicit host submission of four microseconds replaces the captured proxy submission.
The lifetime audit verifies 800 streamed and 160 unstreamed attention prefixes per program, ten active arena gates and ten incoming gates.
Every one of 69,552 compute nodes per base program reserves its physical vector resource.
The additional-slot audit compares 678,069 node rows and confirms that only active/input arena names change.
Original worker and fabric resources, dependencies, durations and release indices remain identical under that renaming.
Program bodies remain at the local paths identified by their fingerprints in the artifact.
Thirteen focused lifetime and vector-resource tests pass, and pinned Black formatting and checking both leave 580 files unchanged.
The source identity is unchanged from the vector-resource correction, so the existing full-suite result remains the applicable code baseline.

Alternative partitions, client phases, mixed batch widths, heterogeneous stage widths, lower-precision caches, offload and prefetch remain open.
A global maximum has not been certified.
