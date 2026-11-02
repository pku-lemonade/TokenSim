# Concurrent GPU global cores with LP30 projections

## Result

This experiment composes a closed-loop decoder for Gemma 4 31B with global attention cores on B200 graphics processing units (GPUs).
All query/key/value (QKV) and output projections, sliding-window attention, feed-forward networks (FFNs), normalization and residual work remain on LP30.
A shared three-LP30 pool holds the tied embedding and output-head weights.
The workload uses independent requests at context 101,000, separate eight-bit floating-point (FP8) key/value (KV) caches reserved through 102,000 tokens, FP8 weights and bfloat16 (BF16) attention.
Model FLOPs utilization (MFU) credits 129,261,109,248 useful matrix floating-point operations per output token and counts every allocated GPU and LP30 at its dense FP8 peak.
Prefill and complete-request MFU remain separate metrics.

The highest measured short-window point reaches 2.90936% MFU.
Its longer replay reaches 2.90028% MFU and 114,733.14 aggregate tokens per second (TPS), with 128 measured outputs for each of 58 requests.
The two halves of that measurement differ by 0.77%, so the record retains the finite-window result and its drift.
The allocation contains 80 B200 GPUs and 123 LP30s, totaling 203 accelerators.
It remains below the 3.03914% existing-format decode incumbent and the 3.41960% experimental shared-cache incumbent.
The latter requires separate model-quality validation.

This is an analytical whole-buffer prototype with a projected memory ledger.
The production mixed-ownership compiler, physical SRAM bank placement, actual cross-device transport and quantized-output equivalence are unverified.
Its negative comparisons do not exclude different kernels, finer operator streaming, heterogeneous stage widths or other ownership schemes.

## Work ownership and request affinity

Each of the ten global layers receives a separate GPU pool.
A pool contains `replicas` independent head tensor-parallel (TP) groups.
Every request cohort keeps the same replica assignment through all ten global layers and across decode steps.
GPU count is therefore `10 * gpu_tp * replicas`.
GPU memory is charged for all request cohorts mapped to the busiest replica.
For example, three four-request cohorts distributed over two replicas require eight requests' KV capacity at the busiest replica.
Dividing twelve requests evenly into six per replica would undercount its storage.

LP30 contiguous pipeline-parallel (PP) stages contain all model layers and retain all projection weights.
A global layer executes LP30 normalization, QKV projection and rotary position encoding, sends query heads and the new KV entry to its GPU pool, runs the GPU attention core, and returns attention outputs to LP30 for the output projection.
Its post-attention normalization, residual path and FFN then execute on LP30.
Sliding-window layers execute entirely on LP30.
All 60 QKV projections, all 60 output projections, all 60 FFNs and all normalization/residual compositions remain priced.
Only the ten global attention cores move to GPUs.

The native provider supplies rank-local costs and geometry.
The GPU core uses its analytical generation-attention estimate with 0.2 microsecond kernel overhead.
LP30 compute and collectives use the existing supplied efficiency, fabric-port and propagation assumptions.
Native costs are analytical estimates and do not establish a hardware implementation.

## Memory projection and operator lifetimes

The capacity projection starts from a native one-layer-per-stage runtime with the existing transformer-only boundary proxy.
It preserves each layer's weights, auxiliary normalization weights and bounded-window KV ledger.
It removes global KV and its blocks from LP30 and charges that cache to the corresponding GPUs.
The separate-cache ledger counts payload bytes without additional FP8 scale metadata, matching the current native model.
Native block rounding, the 1% allocator watermark, the 5 MB safety reserve and conservative communication/live-scratch reservations remain on LP30.
The full native global-attention scratch reservation is retained even though the global core runs remotely.
This conservative reservation supplies reusable scratch for the remaining projections and staging operations.

Every resident request also receives permanent activation slots on every LP30 stage.
Per-rank slots contain four full hidden vectors and two maximum-width QKV bundles in BF16.
They cover request input, residual, partial/reduced output and both directions of communication staging while compute scratch is reused.
The TP4 point charges 63,488 bytes per request per transformer LP30, adding 3,682,304 bytes for 58 resident requests.
The boundary pool receives additional request-specific hidden slots.
GPU memory includes its catalog runtime reserve, resident global KV and request-specific query/KV/output staging.

Each LP30 stage has one exclusive reusable operator-scratch arena.
A local attention composition holds that arena through its reduction and post-attention normalization.
The FFN holds it through its reduction and residual composition.
A global pre-attention composition releases scratch after the outbound transfer finishes sending.
Its GPU wait retains the request's permanent activation slots.
The returning output acquires scratch for the LP30 output projection and reduction.
This permits other requests to compute during a GPU wait without granting unbounded transient activation storage.

The selected TP4/PP30 transformer has two layers per stage.
Its minimum projected SRAM headroom is 1,836,924 bytes after the existing reserve, native cache admission rules and added request slots.
The projection does not prove that the resulting buffers fit a physical compiler's bank placement.

## Communication assumptions

Query heads are redistributed from LP30 TP to GPU TP in BF16.
Each global KV head uses one canonical LP30 sender, avoiding redundant transmissions from LP30's replicated grouped-query KV projections.
GPU head TP above the KV-head count receives an append at every required physical KV replica.
Returned attention heads are redistributed to the LP30 output-projection owners.
Each direction's bytes are independently conserved and tested.

The synthetic cross-device fabric grants one 200 GB/s uplink per 16-LP30 tray and one per eight-GPU node, with separate transmit and receive resources.
Endpoint transfers sharing an uplink serialize against that common resource.
Each transfer round prices its largest shared-resource byte load and 2 microseconds of propagation.
LP30 scratch releases after the wire interval, while the preallocated GPU/request staging retains in-flight data through propagation.
These logical uplinks do not establish that LP30 exposes a physical Ethernet port or that a real GPU/LP30 bridge supplies this bandwidth and latency.
Any concrete bridge, host relay or fabric expansion path needs its own measured transport evidence.

LP30 pipeline handoffs and the embedding/output-head pool use the existing routed chip-to-chip model.
Studies allowing inventories beyond one LP30 rack explicitly set inter-rack propagation to 2 microseconds as a sensitivity assumption.
The host submission resource prices 4 microseconds once per cohort forward.
Cross-device signaling, transport implementation and compiler synchronization remain uncalibrated.

## Coverage and bounds

| Study | Evaluated cases | Admitted cases | Maximum inventory | Best observed MFU |
| --- | ---: | ---: | ---: | ---: |
| LP30 TP4/8/16, eight PP values, GPU TP2/4 and one to four replicas | 170 | 114 | 503 total accelerators | 2.88796% in the short screen |
| Adjacent PP values 12 through 35 at LP30 TP4/8 | 144 | 120 | 403 total accelerators | Bounds only; maximum bound 2.97527% |
| Batch two at LP30 TP2/4/8 and PP15/30/60 | 24 | 15 | Up to the 512-card limit | 2.73611% |
| LP30 TP2 at PP30/45/60, GPU TP2/4 | 24 | 16 | Up to the 512-card limit | 2.60196% |

The studies overlap in their parameter coverage.
The numbers count evaluated cases in each study and do not claim that every row is a distinct hardware layout.
The main and adjacent-PP studies repeat exactly after validation, excluding wall-clock timing and the changed source-identity field.
All 375 case results across the archived studies also repeat after the cache-validation fix, excluding wall-clock timing, source identities and saved-profile input identities.
The five retained frozen program bodies pass their content-hash checks.
Two main-screen points reach the 256-request capacity-search limit.
Their fixed GPU-pool throughput bounds already fall below the incumbent, independently of additional resident requests.
The study does not claim to have exhausted their cache capacity.

The necessary throughput bound combines common LP30 resource demand, each symmetric global GPU pool's service and the request-latency limit.
LP30 resource demand includes minimum arena-lifetime holds.
The bound verifies that all replica profiles have identical common-resource demand and does not assume equal completion rates across replicas.
GPU cross-uplink contention is relaxed in this bound.
The TP4/PP30, two-GPU-replica point is limited to 117,699.43 TPS and 2.97527% MFU by its GPU pools under the declared native costs.
Its common LP30 resource limit is 139,664.38 TPS and its 58-request dependency-path limit is 127,871.40 TPS.
No issue schedule for those same programs and allocation can exceed the steady bound.
Short completion windows can slightly exceed a steady bound because of work crossing the measurement boundary.

## Issue phases and relaxed controls

At 58 requests, the 64-output-per-client burst replay reaches 2.87615% MFU.
Testing fixed issue intervals of 8.4, 8.5, 8.6 and 9.0 microseconds gives a short-window maximum of 2.90936% at 8.6 microseconds.
The longer 16-warmup/128-measured-round replay at that interval yields 2.90028% MFU and approximately 1,978 TPS per resident request.
The run measures 7,424 output tokens and preserves the same projected capacity.
Reducing residency to 16, 32 or 48 requests lowers measured MFU in the supplied controls.

The three-replica point uses 120 B200 GPUs and the same 123 LP30s.
Its unpaced normal control reaches 2.12058% MFU.
Removing LP30 transformer-collective cost raises that observation to 2.73936%; removing only cross-device propagation gives 2.26017%.
Removing both gives 2.87440% observed MFU and a 3.30094% necessary bound.
These controls retain other traffic and compute costs and are explicit relaxations.
They do not provide an achieved hardware result or update the incumbent.
The gap identifies communication/arena dependencies and scheduling as additional targets for a finer producer/consumer model.

## GPU weight-cache evidence and remaining search

The [NVIDIA Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html), checked on November 3, 2026, assigns 126 MB of level-two (L2) cache to the GB200 GPU and describes persistence controls.
The product label in that statement does not by itself verify the available B200 cache set-aside or access-policy limits.
The [CUDA L2 cache-control guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html) requires respecting the device's queried persistence limits and the cache shared by concurrent kernels.
A complete global layer's TP4 FP8 QKV/output-projection matrices occupy 46,792,704 bytes per GPU.
Persistent-weight execution remains a separate ownership/kernel hypothesis requiring verified capacity, a priced L2 access path and an explicit placement across global layers.

The next search includes finer LP30 output streaming, heterogeneous attention/FFN widths and weight placement, GPU context parallelism, persistent GPU weights, and sliding-KV prefetch/offload.
The broader speculative and shared-cache searches also remain open.
No whole-system or global-hardware maximum has been established.

## Reproduction and validation

```bash
python scripts/search_lpx_global_core.py \
  --lp-tp 4 \
  --pp 30 \
  --gpu b200_sxm \
  --gpu-tp 4 \
  --replicas 2 \
  --max-cards 256 \
  --warmup 16 \
  --measured 128 \
  --spacing-us 8.6 \
  --pace-us 8.6 \
  --save-profiles \
  --output outputs/lpx-global-core-paced
```

`--resident 0` selects the maximum admitted request count below `--request-limit` for the chosen LP30 layout.
`--batch`, `--cross-latency-us`, `--cross-bandwidth` and `--host-submit-us` expose the named assumptions.
`--relax-lp-collectives` removes transformer LP30 collective cost as a control while retaining the boundary pool's reductions.
`--bounds-only` skips concurrent replay and retains projected capacity and necessary bounds.
`--save-profiles` writes reconstructable native-cost program snapshots for each GPU replica.
The [portable record](../experiments/lpx/global-core-serving-search.json.gz) retains case results, source/configuration identities, exact-repeat checks, control reproduction code and selected frozen program bodies.
The 22 focused tests pass, including a regression that validates request-count types before looking up cached capacity results.
The full suite runs 873 tests with 871 passes and the two unchanged macOS environment failures involving `/tmp` normalization and unavailable NVIDIA process facilities.
All 579 Python files pass the pinned Black check.
