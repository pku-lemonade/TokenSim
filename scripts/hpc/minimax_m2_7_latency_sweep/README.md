# MiniMax-M2.7-229B latency sweep

These scripts use the historical PSLA and TransformerRoofline interfaces from the HBF snapshot, commit `0b8d751`.
Run them from a checkout of that revision.
See [current HBF examples](../../../docs/hbf.md) for the merged runtime and independent storage/device configurations.

Launches 1,000 MiniMax-M2.7-229B full-trace simulations as four one-node c96
Slurm array tasks. Each trace owns 250 simulations and uses all 96 cores: 58
cores execute three sequential simulations and 38 execute two.

The matrix includes HBF1-48GiB latency variants, HBF2 latency variants, and
legacy non-batched SSD capacities `ssd12`, `ssd24`, and `ssd48` for H100 and
B200. It excludes all `hbf1-r*-w*` variants and explicitly sets
`ssd_io_model: legacy`, so Mooncake SSD-batching configurations are not run.

Run on `hpc-cpu` from a clean source snapshot:

```bash
scripts/hpc/minimax_m2_7_latency_sweep/launch.sh "$PWD" "$RUN_ROOT"
```

Completed JSON files are written below:

```text
$RUN_ROOT/results/1-latency-sweep/{H100,b200}/minimax-m2.7-229b/
```
