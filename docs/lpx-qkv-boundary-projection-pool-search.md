# LP30 QKV projection placement in the boundary pool

Progress date: 2026-11-04, Asia/Shanghai.
The boundary-pool projection experiment supports 57 resident requests under its declared memory policy, but its measured schedules do not improve the [3.832974589% analytical decode record](lpx-qkv-projection-broadcast-search.md).
A longer 448-chip replay reaches 2.959402320% model floating-point operation utilization (MFU) and 126,207.375 tokens per second (TPS).
The objective remains pure LP30 hardware, MFU priority and no minimum TPS.
The objective-wide maximum remains unproven.

## Workload and placement

The experiment retains Gemma 4 31B decode at 101,000 positions and key-value (KV) storage reserved through 102,000 positions.
It retains eight-bit floating-point weights and KV, bfloat16 activations, the experimental value/rotary shared-cache layout and the existing split-dot attention costs.
Useful credit remains 129,261,109,248 floating-point operations per committed output token.
Every allocated chip remains in the peak denominator, including the complete tied embedding and language-model head pool.
[Complete-request serving](lpx-complete-serving-mfu.md) retains its separate workload and records.

The 440 transformer chips retain ten repetitions of a one-layer local stage on four chips, a four-layer local stage on eight chips and a one-layer global stage on 32 chips.
The preceding experiment retained one query/key/value (QKV) projection matrix on each global KV-head group's projection root.
This experiment places all forty retained matrices on chips that also hold the complete embedding and language-model head.
A matrix's boundary chip is `(4 * global_stage + kv_head_group) % pool_width`.
Global output projections, feed-forward networks, normalization storage and original buffer charges remain.

## Capacity and scratch lifetimes

The fresh native weight ledger identifies 24,772,608 bytes per complete QKV matrix.
Removing 320 global copies releases 7,927,234,560 bytes across the transformer.
Forty retained matrices occupy 990,904,320 bytes in the boundary pool, with another 45,058 bytes of copied parameters per matrix.
Every used boundary chip reserves 524,288 bytes of shared projection scratch and a 1,027,072-byte message guard.
The separate 5 MB safety reserve and complete boundary frames remain charged.
The capacity screen covers pool widths from three through 64 for batch sizes one and two; both require at least five pool chips under the specified placement.

Each global KV-head group distributes its 6,375 whole sixteen-position blocks cyclically across eight ranks.
The physical owner of a block is `block % 8`, and its local position is `(block // 8) * 16 + position % 16`.
Each request and rank receives a separately rounded one-percent watermark allowance.
The independent ledger admits 56 and 57 requests and rejects 58 under this policy.
Minimum global-chip headroom is 140,094 bytes at 57 requests.
Native admission agrees after explicitly recalculating replica block budgets from the modified physical states.
This modified-state admission does not establish default AFService compilation of the custom placement and ownership policy.

For batch-one trials, a third complete active and input frame on each four-layer local stage adds 1,628,192 bytes per affected chip.
Fresh native admission and the watermark-adjusted ledger retain 7,992,920 bytes of minimum headroom for 57 requests on those stages.
The local four-layer stages use three frame slots; the other rotating stages use two.
Six program variants implement the combined rotation.

Separate ingress banks cover each matrix's incoming hidden-state transfer.
The boundary chip acquires shared projection scratch after arrival and holds it through outgoing transmission.
Each incoming message contains 10,752 hidden-state bytes and a sixteen-byte descriptor.
Each outgoing message contains 8,192 query bytes, 1,280 KV bytes and a sixteen-byte descriptor.
Native costs include packing, receiving copies, control work, the original normalization/projection/rotary prefix, routed transmission and propagation.
The twelve-lane fanout uses 96 outgoing lanes for eight recipients.
Global active frames enclose the added operations, and the original batch-two attention-core gate remains.

## Numerical controls and finite search

Four synthetic full-shape projection fixtures use ten distinct layer parameter sets each.
The controls reproduce identical projection and prefix payloads after hidden-state transport.
They check 4,800 matrix slots across the passing pool layouts and all 1,824 request/rank KV cells for the 57-request layout.
Sixteen cached-attention cases cover four KV heads and context lengths of 1, 12,641, 101,000 and 102,000.
The largest absolute error against the independent full-context reference is approximately `6.12907e-8`, below the `1e-5` tolerance.
A fresh rerun reproduces every numerical result exactly, excluding host execution time.
These synthetic controls do not establish trained-model quality or LP30 compiler execution.

The dependency and resource bounds below apply to the specified captured graphs and memory policies.
They are necessary upper bounds, and they do not imply an attainable schedule.

| Batch | Pool chips | Total chips | Requests | Local four-layer frames | MFU upper bound |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 448 | 57 | 2 | 3.664759087% |
| 1 | 8 | 448 | 57 | 3 | 3.962274264% |
| 2 | 5 | 445 | 56 | 2 | 3.480308135% |
| 2 | 6 | 446 | 56 | 2 | 3.520047655% |
| 2 | 7 | 447 | 56 | 2 | 3.516008885% |
| 2 | 8 | 448 | 56 | 2 | 3.511037473% |
| 2 | 12 | 452 | 56 | 2 | 3.471755393% |
| 2 | 16 | 456 | 56 | 2 | 3.444588828% |

Every tested batch-two graph is bounded below the current decode record.
The batch-one graph with a third local frame retains a gap between its bound and measured performance.
A 445-chip batch-one control keeps QKV projection on global chips and has a corresponding 3.988986225% bound.

The retained order studies contain 892 specified linear programs: 66 single-layer calendar orders, 512 alternative single-layer orders, 310 four-layer software orders and four complete-pipeline orders.
The artifact contains 127 successful primal/dual certificates and 765 solver-reported infeasible cases without Farkas certificates.
These results do not exclude other resource orders.
The best tested single-layer period is approximately 1.825080 microseconds, and the best tested three-frame four-layer period is approximately 8.496599 microseconds.
These component results do not assign complete-pipeline throughput.

## Request-feedback replays

Every later forward waits for its request's preceding token completion.
All submitted jobs drain, including jobs remaining after the measurement window.
Each long replay discards eight warmup rounds and measures 32 rounds, with a 6.1-microsecond minimum issue interval.
The unanchored native scheduler determines actual contention and completion timing.

| Placement and measurement | Requests | Measured tokens | Drained jobs | MFU | Aggregate TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Boundary projections, two warmup and four measured rounds | 57 | 228 | 399 | 2.948595679% | 125,746.512 |
| Boundary projections, eight warmup and 32 measured rounds | 57 | 1,824 | 2,337 | 2.959402320% | 126,207.375 |
| Global projections, two warmup and four measured rounds | 56 | 224 | 392 | 3.033182239% | 128,487.602 |
| Global projections, eight warmup and 32 measured rounds | 56 | 1,792 | 2,296 | 3.011306690% | 127,560.940 |

The long boundary-projection replay's relative throughput drift between measurement halves is approximately +0.09174%.
The global-projection control's corresponding drift is approximately -0.12554%.
Every measured physical-resource utilization remains at or below one.
Neither supplied schedule improves the separate 56-request batch-two QKV-sharing record.

## Evidence and continuation

The [portable evidence archive](../experiments/lpx/qkv-boundary-projection-pool-search.json.gz) retains the experiment's JSON records, numerical controls, native graph bodies, parent graphs, order seeds, constraint arrays, successful certificates, replay traces and helper sources.
The [validation receipt](../experiments/lpx/qkv-boundary-projection-pool-validation.json) records reconstructed capacity, graph preservation, added-operation lifetimes and costs, every specified order and replay accounting.
Run the archive validator from the checkout after installing the project dependencies.
The command extracts the three validation helpers into a fresh temporary directory and loads replay helpers and graph data from the archive.

```bash
python - <<'PYCODE'
import gc
import gzip
import json
import runpy
import sys
import tempfile
from pathlib import Path

gc.disable()
artifact = Path("experiments/lpx/qkv-boundary-projection-pool-search.json.gz")
data = json.loads(gzip.decompress(artifact.read_bytes()))
with tempfile.TemporaryDirectory(prefix="lpx-pool-validation-") as directory:
    root = Path(directory)
    for name in ("archive", "certificates", "validate"):
        filename = f"lpx_qkv_pool_{name}.py"
        (root / filename).write_text(data["reproduction_sources"][filename])
    del data
    sys.path.insert(0, str(root))
    sys.argv = ["lpx_qkv_pool_validate.py", "--replay-control"]
    runpy.run_path(str(root / sys.argv[0]), run_name="__main__")
PYCODE
```

The current native simulator source identity remains `9dfe67e50e9ab76edafd96f37033622e8ae62b2abb3a03f0d233a81920e2407e`.
The existing best-result dictionaries remain unchanged.

The unfinished session's pool-width screen is now recorded with longer measurements.
Other pool assignments, guarded early projection preparation, different cyclic resource orders and different transformer inventories remain open.
Hardware compilation, physical SRAM banks, fixed physical routes, multi-rack timing and experimental-cache model quality remain unverified.
The batch-one upper-bound gap and the separate complete-serving objective remain unresolved.

The continuation also ran the project formatter, format check and all 898 unit tests.
Two unchanged GPU-tool tests fail on this macOS host: `test_watch_paths_require_unique_names` assumes that `/tmp` remains unresolved, and `test_stop_refuses_foreign_process` invokes the unavailable `nvidia-smi` executable.
The other 896 tests pass.
