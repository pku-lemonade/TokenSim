# V3 Poisson-arrival experiments

Run from the repository root with the TokenSim Python dependencies installed.
These are CPU event simulations, not checkpoint inference on GPUs.

## Poisson QPS sweep

PowerShell:

```powershell
foreach ($a in 2, 4, 8) {
    python scripts/run_v3_poisson.py --a-tp $a --f-ep 32 `
        --qps 1 4 8 16 32 64 128 --requests 64 --max-batch 32 --seed 42 `
        --output "outputs/v3-poisson-${a}a32f-20260920"
    if ($LASTEXITCODE -ne 0) { throw "A=$a failed" }
}
python scripts/plot_v3_poisson_comparison.py
```

Linux shell:

```sh
for a in 2 4 8; do
    python scripts/run_v3_poisson.py --a-tp "$a" --f-ep 32 \
        --qps 1 4 8 16 32 64 128 --requests 64 --max-batch 32 --seed 42 \
        --output "outputs/v3-poisson-${a}a32f-20260920" || exit 1
done
python scripts/plot_v3_poisson_comparison.py
```

The comparison script reads the three directories above. Its main plot is
`outputs/v3-poisson-comparison-20260920/throughput-vs-qps.png`:
offered requests/s on the x axis, output tokens/s on the y axis, with no power
normalization. Throughput includes time from simulation start through draining
the final request. Each individual run also produces its own plots and summaries.

The runner resumes completed QPS cases in the same output directory. Use a new
`--output` directory for an independent rerun. The comparison script currently
uses the dated default directories above; individual-run plots support any output
directory.

## Workload and interpretation

- Full DeepSeek-V3 671B, BF16, 61 layers plus model boundaries.
- A TP=2/4/8; F EP=32 with expert TP=1; Router on A.
- A100 SXM 80GB device model and NVLink3 network model; analytical-only timing.
- Every F worker has a hypothetical 1536 GiB capacity override; compute and
  bandwidth retain their base-device values.
- Each request begins with 2048 cached tokens and generates one decode token.
  Prefill, multi-token generation and serving CPU overhead are excluded.
- Exponentially distributed interarrival times produce Poisson arrivals.
- FIFO batching admits up to 32 pending requests when the previous iteration
  finishes, with AF microbatch size 8 and no deliberate batching timer.
- Arrivals continue during service; the finite queue drains after the last arrival.
- Queue latency is arrival-to-admission; service latency includes contention after
  admission. Per-request completion follows its own AF block.
- 64 requests and one seed per rate are a short pilot, not a steady-state capacity
  or tail-latency measurement. Realized arrival rates differ from nominal QPS.

Each QPS directory contains request arrivals/completions, batch records, queue
samples, hardware/cluster/runtime plans, dependency validation and a compressed
Chrome-format operator trace (`operators.trace.json.gz`). Decompress the trace
if the viewer does not accept gzip. Large generated results are ignored by Git.

## Fixed-batch topology sweeps

```sh
python scripts/run_v3_f_count_sweep.py
python scripts/run_v3_a_count_sweep.py
```

These run the earlier fixed-batch experiments: 1A with F=2/4/8/16/32, or
A=2/4/8/16/32 with F=32. Both use batch=1/4/16/32, AF block=8 and a single decode
step. They are not QPS tests; their energy plots retain the explicit assumption
of 250 W per virtual card.

## Focused tests

```sh
python -m unittest discover -s tests -p test_v3_poisson.py
python -m unittest discover -s tests -p test_deepseek_v3_operator_table_adapter.py
```

Legacy test migration and full-suite validation remain outstanding in the
OpenSpec migration task list; passing these focused tests is not a full-suite pass.
