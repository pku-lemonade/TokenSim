# 32-user DeepSeek-V3 decode experiment

The repository runner executes the historical 32-user workload on the current attention/FFN (AF) runtime.
It runs a CPU discrete-event simulation without model weights or physical GPUs.
Run it from a clone of the `afd` branch after installing the repository dependencies.

```bash
python -m pip install -e .
python scripts/run_v3_user_qps.py \
  --qps 4 \
  --users 32 \
  --requests 32 \
  --unique-users \
  --context 2048 \
  --output-tokens 400 \
  --a-tp 4 \
  --f-ep 32 \
  --max-batch 8 \
  --af-batch-size 8 \
  --seed 42 \
  --busy-policy queue \
  --output outputs/u32-qps4
```

The listed arguments are the defaults except for the required new output directory.
The equivalent short command is `python scripts/run_v3_user_qps.py --output outputs/u32-qps4`.
Existing output directories are rejected to preserve previous evidence.
Multiple values after `--qps` run independent points sequentially.

## Fixed model and hardware assumptions

| Setting | Value |
| --- | --- |
| Model | Complete DeepSeek-V3 671B MLA, 61 layers, BF16 |
| Attention workers | 4, tensor parallelism (TP) = 4 |
| FFN workers | 32, TP = expert parallelism (EP) = 32, expert TP = 1 |
| Other parallelism | Data parallelism = pipeline parallelism = 1 |
| Device performance | A100 SXM 80 GB compute and HBM bandwidth |
| Attention memory | Base A100 capacity |
| FFN memory | 1536 GiB virtual capacity per worker |
| Placement | Five nodes with eight slots each; dedicated A slots 0–3, F slots 8–39 |
| Node communication | Exclusive FIFO domains; local 300 GB/s, inter-node TX and RX 50 GB/s each |
| Fixed communication latency | Zero |
| Host timing | Disabled |
| Estimation | `analytical_only`, eager execution, A-side router, partial-return protocol |

The FFN capacity override preserves A100 compute and memory bandwidth.
It describes hypothetical storage capacity and does not describe a physical 1536 GiB A100.
Static plans and pure service estimates can be cached; each iteration executes real AF DAGs and acquires communication resources again.
The runner uses the current repository implementation and records source hashes.
Equivalent workload settings do not guarantee identical timing estimates across different simulator revisions.

## Workload and metric definitions

The total offered arrival rate is a Poisson process with seed 42.
An independent random stream with seed 43 shuffles the user IDs, assigning one request to each of the 32 users.
Users have independent KV state initially containing 2048 tokens.
Prefill is assumed complete and its time is excluded.
Each request produces 400 output tokens through 400 decode steps, giving 12800 output tokens for the full run.
The scheduler admits arrivals during computation and serves up to eight distinct users per batch with round-robin token scheduling.
Iterations are serialized, users may be idle, and a batch can contain fewer than eight requests.
The simulator drains all requests after the last arrival.
The finite experiment starts from empty queues and has no warmup or steady-state claim.

Time to first token (TTFT) is arrival to the first output, including queueing.
Time between tokens (TBT) uses all 32 × 399 adjacent output intervals.
Request latency is arrival to the last output.
For each request, latency equals TTFT plus its 399 TBT intervals.
Throughput is 12800 divided by simulated time from zero to the last completion, including idle time and drain.
Percentiles use nearest rank, `ceil(p*N)`.
Results label mean, P95 and maximum separately and report milliseconds for latency fields.
`wall_seconds` measures host execution time and is separate from simulated latency.

## Validation and outputs

To validate and export all default settings and arrivals without running decode, add `--dry-run` and choose a new output directory.
Dry runs write `status: planned` and do not produce completed results.
For a short real execution, reduce only the request count and output length explicitly:

```bash
python scripts/run_v3_user_qps.py \
  --requests 2 \
  --output-tokens 2 \
  --output outputs/u32-smoke
```

The smoke command retains the full model and hardware assumptions but has a smaller workload.
It does not replace the full 32-request run.

The output root contains `experiment.json`, source/configuration hashes and completed-point `results.json` and `results.csv`.
Each `qps-*` directory contains `arrivals.json`, `requests.json`, `tokens.json`, `batches.json`, `queue.json`, `progress.json`, `summary.json` and `validation.json`.
Model, cluster, runtime-plan, effective AF configuration, hardware and communication metadata are saved alongside them.
The runner recomputes metrics from the saved records and validates token accounting, request ordering, per-user serialization, context growth, batch limits and empty final user and communication queues.
`sample-operators.trace.json.gz` records the initial occurrence of each actual batch size with full 61-layer operator coverage.
Subsequent iterations execute with operator tracing disabled; every token remains in `tokens.json`.
Failures retain partial records and `failure.json` and do not create a successful summary.
