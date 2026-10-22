# LP30 and Rubin simulation

TokenSim includes separate catalog entries for Groq LP30 and NVIDIA Rubin, together with the Gemma 4 31B text architecture.
The native attention/feed-forward network (AF) runtime checks physical rank ownership, weights, key/value (KV) storage and communication workspaces before scheduling model operations.
The bundled hardware efficiencies and timing constants are analytical assumptions without measured LP30 calibration.

## Run native decode

Run the complete Gemma text model on 256 LP30 ranks with tensor parallelism (TP) 16 and pipeline parallelism (PP) 16:

```bash
python scripts/run_lpx.py \
  --model gemma-4-31b-text \
  --device groq_lp30 \
  --tp 16 \
  --pp 16 \
  --dtype fp8 \
  --kv-cache-dtype fp8 \
  --batches 1 4 \
  --context 1024 \
  --steps 2 \
  --output outputs/lpx
```

Run a Rubin comparison with an explicit inventory:

```bash
python scripts/run_lpx.py \
  --model gemma-4-31b-text \
  --device rubin \
  --tp 8 \
  --pp 1 \
  --dtype fp8 \
  --kv-cache-dtype fp8 \
  --batches 1 4 \
  --context 1024 \
  --steps 2 \
  --output outputs/rubin
```

Both commands use the shared `AFExperimentRunner` and production operator-table providers with analytical fallback selected explicitly.
The entry point measures fixed-batch decode at the requested context.
Its rates count native model tokens and exclude prefill, time to first token, external arrivals and end-to-end request latency.
It does not generate text, evaluate numerical quality or reproduce an undisclosed benchmark deployment.

`summary.json` records the model, device assumptions, runtime plan and timing summaries.
Each result identifies its checkpoint directory under `points/`, containing `manifest.json`, `result.json` and an optional trace.
Matching completed points are reused; `--no-resume` reruns them.
Use `--trace validate` to check events without saving them or `--trace off` to disable event collection.
The trace format and filtering options match the [AF runner](afd.md#configure-and-run).

| Argument | Behavior |
| --- | --- |
| `--model` | Catalog ID or a model YAML path; defaults to `gemma-4-31b-text`. |
| `--device` | `groq_lp30` or `rubin`; defaults to `groq_lp30`. |
| `--tp` | Tensor-parallel ranks per pipeline stage; defaults to 16. |
| `--pp` | Pipeline stages; defaults to 16. |
| `--pipeline-layer-counts` | Optional positive layer counts, one per stage, summing to the model's layer count. |
| `--dtype` | Optional weight format override; activations retain the model's declared format. |
| `--kv-cache-dtype` | Optional KV storage format override. |
| `--batches` | Positive fixed decode batch sizes; defaults to one. |
| `--context` | Positive resident context length; defaults to 1,024 tokens. |
| `--steps` | Positive decode iterations; defaults to one. |
| `--seed` | Nonnegative routing seed; defaults to 42. |
| `--collective-algorithm` | `ring`, `tree`, `direct` or `auto`; defaults to `ring`. |
| `--execution-mode` | `eager` or `cuda_graph`; defaults to `cuda_graph`. |
| `--colocated-dense-reuse` | Reuses eligible complete local activation replicas between dense attention and FFN work. |
| `--data-root` | Catalog directory; defaults to the repository's `data/` directory. |
| `--output` | Result directory; defaults to `outputs/lpx`. |

The runner uses a colocated complete-model layout with one data-parallel replica.
For heterogeneous attention/FFN placement, use the [standard AF configuration](afd.md) with separate execution domains or the `Layout` and `configuration` helpers in [lpx_runtime.py](../scripts/lpx_runtime.py).
Unsupported context parallelism and colocated hybrid expert TP/EP layouts are rejected before rank remapping.

## Model and memory behavior

The [Gemma catalog entry](../data/models/gemma-4-31b-text.yaml) records per-layer windows, head dimensions, KV head counts and shared K/V projections.
Shared projections still retain separate transformed K and V tensors in the cache.
Native decode and complete-model prefill use each stage's owned layers, including explicit nonuniform pipeline partitions.
Cached prefill accounts for reads from the retained KV state without multiplying fixed launch costs by a floating-point operation ratio.
Operator-specific weight formats propagate to prefill and decode.

LP30 weight, activation and KV estimates use the catalog's on-chip static random-access memory (SRAM) capacity and bandwidth.
Compilation reserves weights and protocol buffers, and admission checks the maximum KV context on every physical rank.
This model excludes external DRAM weight streaming and compiler-level SRAM bank allocation.

Two optional mappings in a device's `analytical` section enable additional accounting:

- `lpx_pipeline` selects independent matrix, vector, exponential, shuffle and SRAM service estimates.
- `lpx_memory` reserves reusable scratch space and enforces configured prefill and context bounds.

The defaults are defined by [LP30Pipeline](../TokenSim/operator_data/analytical/lp30.py) and [LP30Memory](../TokenSim/af/runtime/lpx_memory.py).
They are uncalibrated hypotheses and are disabled in the bundled catalog.
For a custom catalog, the following mapping enables them with explicit assumptions:

```yaml
analytical:
  source_id: local-lp30-assumptions
  grade: D
  lpx_pipeline:
    activation_dtype: bf16
    vector_ops_per_s: 2.0e+13
    exp_ops_per_s: 1.0e+12
    shuffle_bytes_per_s: 1.0e+13
    vector_efficiency: 0.5
    shuffle_efficiency: 0.8
    vector_bytes: 320
    attention_tile_keys: 256
    fill_drain_us: 0.05
    schedule: tiled
  lpx_memory:
    prefill_chunk: 16
    maximum_context: 102000
    attention_tile_keys: 256
    tile_buffers: 2
    reserve_bytes: 5000000
```

The pipeline activation format must match the model's activation format.
Keep the attention tile size consistent between the two mappings.
The `tiled` service estimate overlaps unit work analytically; `serialized` sums that work.
Neither mode implements a compiler schedule or vector-level network overlap.
Live scratch reserves the maximum sequential attention, FFN or output arena, plus auxiliary weights and the explicit reserve.
Admission and forward calls reject contexts or query batches exceeding these reservations.

## Catalog evidence and limits

The [LP30 device](../data/devices/groq_lp30.yaml), [Rubin device](../data/devices/rubin.yaml), [link catalog](../data/topologies/links.yaml) and [LPX topology](../data/topologies/groq_lpx256.yaml) retain source identifiers and evidence grades.
The [evidence grading guide](../TokenSim/hardware/README.md) defines these grades.

| Catalog quantity | Recorded value | Interpretation |
| --- | --- | --- |
| LP30 SRAM | 500,000,000 bytes per chip; 150 TB/s peak bandwidth | Vendor specification recorded in the catalog; sustained operator bandwidth is unmeasured. |
| LP30 FP8 matrix peak | 315 PFLOPS divided by 256 chips | Derived per-chip peak; separate matrix and vector formats require their own rates. |
| LP30 chip injection | 1.25 TB/s per direction | Equal-direction split of the recorded 2.5 TB/s bidirectional aggregate; the 96 ports do not multiply this budget. |
| LP30 C2C latency | 0.05 microseconds per modeled hop | Unmeasured topology assumption. |
| Rubin-to-LP30 link | 200 GB/s per direction; 2 microseconds propagation | Unmeasured heterogeneous-transport assumption. |
| Processor power | 450 W for LP30; 1,800 W for Rubin | Assumed chip power, excluding hosts, racks and facility overhead. |

The source index includes [NVIDIA's LPX product description](https://www.nvidia.com/en-us/data-center/lpx/), [LPX architecture description](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) and [Rubin product description](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/).
Earlier Groq generations have separate catalog identities and cannot supply LP30 calibration measurements.
The native flat-link policies are generic analytical collectives; `direct` assumes a nonblocking fabric and does not establish a particular hardware or compiler implementation.
The retained simulator does not establish a reproduced NVIDIA Figure 11 frontier or the hardware allocation of an Artificial Analysis endpoint.
