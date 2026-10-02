# AgentX / InferenceX native simulation

TokenSim's AgentX path replays the SemiAnalysis coding traces against the full
DeepSeek-V4-Pro model. It models request trees, MTP, native CSA/HCA attention,
MegaMoE, mHC, DP lockstep, hybrid KV caching, SimpleCPU and Mooncake transfer
lifecycles. The simulated LLM does not generate text or require a model server.

The reference audit and decisions are recorded in
[the research plan](agentx-research-plan.md). Stage reports cover the native
[MegaMoE](agentx-native-megamoe-report.md),
[attention/MTP](agentx-native-attention-report.md), [mHC](agentx-native-mhc-report.md),
[Mooncake](agentx-mooncake-report.md), and [curve](agentx-native-curve-report.md)
work. Older proxy results remain in their original directories for historical
comparison and are not valid native measurements.

## Native matrix

The versioned matrix is `data/experiments/agentx.yaml`. It requires the full
393-trace pool, a warmup, a 3,600-second profile, a 30-second grace period and
the reference cancellation window. A one-hour run is simulated time; it does
not require every request in the trace pool to finish.

Check and inspect the matrix before running it:

```bash
python scripts/run_agentx_experiments.py --action check
python scripts/run_agentx_experiments.py --action list
python scripts/run_agentx_experiments.py \
  --action expand --output /tmp/agentx-expanded.json
```

Run the two native checkpoints and retain each simulation seed separately:

```bash
python scripts/run_agentx_experiments.py \
  --action run --group checkpoints --seeds 0,1,2 \
  --result-root results/agentx-matrix/native-checkpoints
python scripts/run_agentx_experiments.py \
  --action plot --result-root results/agentx-matrix/native-checkpoints \
  --output results/agentx-matrix/native-checkpoints/summary.png
```

Run the configured TP8, TP4/EP4 and DEP8 curve points with three MTP seeds:

```bash
python scripts/run_agentx_experiments.py \
  --action run --group curve --seeds 0,1,2 --max-parallel 2 \
  --result-root results/agentx-matrix/native-curve
python scripts/run_agentx_experiments.py \
  --action plot --group curve \
  --result-root results/agentx-matrix/native-curve \
  --output results/agentx-matrix/native-curve/summary.png
```

`summary.csv` keeps seed rows and adds a mean/min/max/standard-deviation row
when at least two seeds are present. It records total and per-GPU throughput,
p90 TTFT/ITL/E2E, GPU and CPU hit rates, topology, cache backend and any
checked-in reference error. Points without an independently verified reference
artifact remain explicitly uncomparable.

Use `scripts/run_agentx_native.py` only when a single matrix point and backend
must be inspected. The `simple_cpu`, `hbm` and `mooncake` choices are written
to the run manifest. Mooncake uses the native shared-page adapter and records
its shared pool, DMA pinning, queue contention, asynchronous load/save and
cancellation counters.

## Data and scope

The workload is read directly from `dataset/agentx/traces.jsonl`; TokenSim
implements the AIPerf trace parser and dependency replay internally. Recorded
prefix hashes, nested streams, SPAWN/JOIN dependencies, session routing and
input/output lengths are preserved. The dataset is intentionally gitignored.

Operator data is imported from the checked-in AIConfigurator sources with the
original metadata and licenses. The native attention tables are FlashMLA
measurements while the historical serving reference uses FlashInfer and an
MXFP4 indexer cache; this backend/layout mismatch is retained in provenance.
MegaMoE uses the matched GB200/GB300 table as a documented cross-hardware
estimate, and shared-expert fusion is an explicit approximation. These limits
are reported with every native result and are not silently folded into a
reference value.

## Validation

Run the full logical suite with:

```bash
pytest -q
```

The curve validator checks every completed point's request count, final cache
state, cancellation cleanup and DP time conservation:

```bash
python scripts/validate_agentx_curve.py \
  results/agentx-matrix/native-curve \
  --output results/agentx-matrix/native-curve/validation.json
```

The results are a simulator reproduction, not an official InferenceX
submission. Absolute throughput remains subject to the documented operator
table and FlashMLA/FlashInfer differences; use the per-stage reports and
component time breakdown when interpreting the remaining reference error.
