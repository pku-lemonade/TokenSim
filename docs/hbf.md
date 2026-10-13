# HBF storage and experimental device examples

High-bandwidth flash (HBF) is an optional storage tier for uniform attention key/value (KV) blocks in a Mooncake store.
Selecting HBF storage leaves the cluster's accelerator specifications unchanged.
The optional implementation lives in the independently installable `examples/storage/hbf/` package and uses the [storage backend plugin interface](storage-backends.md).
The HBF model and memory-variant parameters originate from the public HBF snapshot, commit `0b8d751`, and its [HBF characterization paper](https://arxiv.org/abs/2608.11668).
TokenSim simulates these devices without accessing physical HBF hardware.

## Ordinary GPUs with HBF storage

Install the example plugin from the repository root.

```bash
python -m pip install -e examples/storage/hbf
```

Use a normal cluster configuration and select HBF through `--kv_transfer_config`.
This example uses eight standard H100 GPUs, a 1 TiB HBF store and an assumed sequential bandwidth of 128 GiB/s in each direction.

```bash
python benchmark.py \
  --model data/models/llama-7b.yaml \
  --cluster data/clusters/8_h100/tp8.json \
  --latency_backend analytical \
  --batching paged-attn \
  --block_size 16 \
  --workload_type qwen_jsonl \
  --dataset_path dataset/qwen_example.jsonl \
  --request_count 3 \
  --kv_transfer_config data/kv_transfer/mooncake_store_hbf.json \
  --verbose none \
  --results_path results/hbf-example
```

The analytical backend keeps this example small; standard GPUs can also use their measured operator tables.
The trace supplies prefix hashes for store writes and reuse.
Reads from HBF occur when a reusable prefix has been evicted from the GPU cache.

## Independent GPU memory variants

The device catalog contains experimental memory variants of H100, H200 and B200.
Each entry preserves the base GPU's compute and interconnect specifications and changes its memory kind, capacity and bandwidth.
Memory and analytical-efficiency assumptions carry evidence grade D.
Base-device aliases such as `H100` continue to select the ordinary GPU.

| Device ID | GPU memory | Capacity per GPU | Bandwidth per GPU |
| --- | --- | --- | --- |
| `h100_sxm_hbf1` | GDDR7 | 32 GiB | 1.792 TB/s |
| `h100_sxm_hbf1_48gib` | GDDR7 | 48 GiB | 1.344 TB/s |
| `h100_sxm_hbf2` | HBM3e | 48 GiB | 1.5 TB/s |
| `h200_sxm_hbf1` | GDDR7 | 32 GiB | 1.792 TB/s |
| `h200_sxm_hbf2` | HBM3e | 48 GiB | 1.5 TB/s |
| `b200_sxm_hbf1` | GDDR7 | 64 GiB | 3.584 TB/s |
| `b200_sxm_hbf1_48gib` | GDDR7 | 96 GiB | 2.688 TB/s |
| `b200_sxm_hbf2` | HBM3e | 96 GiB | 4 TB/s |

Select a variant in the worker group's `hardware` field.
`data/clusters/8_h100/tp8_hbf1.json` provides an eight-GPU HBF1 example.
GPU variants can use HBF storage, SSD storage or no external store.
They have separate device IDs and no imported operator measurements, so the default `table_first` policy uses analytical estimates.
Explicitly requesting the base GPU's operator backend fails when the variant has no corresponding package.
Add measurements under the variant's own device ID when they become available.

## Storage timing and configuration units

The connector's `offload` object selects `tokensim_hbf:HBFBackend` and supplies plugin-owned options in `config`.
The example configurations name the instance `hbf`.

```json
{
  "offload": {
    "backend": "tokensim_hbf:HBFBackend",
    "name": "hbf",
    "config": {
      "capacity_bytes": 1099511627776,
      "seq_read_bandwidth_bytes_per_s": 137438953472,
      "seq_write_bandwidth_bytes_per_s": 137438953472,
      "read_latency_us": 12,
      "write_latency_us": 120
    }
  }
}
```

Capacity uses `capacity_bytes`.
Bandwidth fields use bytes per second: `seq_read_bandwidth_bytes_per_s`, `seq_write_bandwidth_bytes_per_s`, `random_4k_read_bandwidth_bytes_per_s` and `random_4k_write_bandwidth_bytes_per_s`.
Fixed media latency uses `read_latency_us` and `write_latency_us`.
The plugin validates these fields and supplies their defaults.
Each operation costs its fixed latency plus transferred bytes divided by effective bandwidth.
Missing random 4 KiB bandwidth values default to 80% of sequential bandwidth.
The effective bandwidth interpolates logarithmically between 4 KiB and 10 MiB operation sizes and remains at the endpoint values outside that interval.

HBF uses the same store admission, prefix matching and ownership rules as SSD.
Writes become visible after their modeled completion time, and in-progress reads pin their objects against eviction.
Synchronous saves include media latency in the caller's blocking time.
The uniform connector waits for I/O completion and rejects `transfer_overlap`, independently of the selected backend.
Results export HBF counters under `storage_metrics[master]["hbf"]` and describe the configured media in `mooncake_offload_profiles`.
The generic dictionary replaces the fixed `mooncake_hbf_*` result fields.

Grouped KV pages currently use the native DRAM page store.
The runtime rejects HBF and SSD offload with `cache_layout=grouped`.
Supporting flash for grouped pages requires extending that page-storage interface.

## Migration from the public HBF snapshot

The checkout includes one hardware-independent HBF storage configuration, `data/kv_transfer/mooncake_store_hbf.json`.
Set storage capacity and bandwidth in its plugin options for the system being modeled.
The snapshot's `*_capacity_gb` and `*_bw_gbps` fields used binary GiB units, so the migrated values multiply those inputs by `2**30`.
GPU memory bandwidth in the device examples follows the snapshot's decimal TB/s values and uses `10**12` bytes/s per TB/s.
`accelerator_memory_preset`, `target_hardware` and `hbf_preset` have been removed from storage configurations.
Move the intermediate flat `hbf_*` options into `offload.config` and remove their `hbf_` prefixes.
Select `tokensim_hbf:HBFBackend` through `offload.backend`; the core rejects the old flat options.
Choose the corresponding device ID in the cluster to reproduce a GPU memory variant.

The merged runtime uses main's device catalog and operator latency model.
Its hardware, model accounting and scheduling corrections can change results relative to the original HBF snapshot.
Uniform external prefix hits follow main's trailing-block rule, so full-prompt reuse leaves one block for computation.
The retired TransformerRoofline backend and PSLA configuration interface remain available in commit `0b8d751`.
Historical experiment configurations and runners are available in that snapshot.
