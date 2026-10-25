# TokenSim/hardware/

This module loads device specifications, interconnect definitions and the hierarchy connecting devices, nodes and clusters.

The catalog replaces the `TransformerRoofline` hardware table.
Every numeric field carries a `source_id` and evidence grade so that official specs, project assumptions, and calibrated values can be told apart in results.

## Directory structure

```
hardware/
  __init__.py       lazy re-exports; HardwareContext loaded on first access
  _yaml.py          YAML loading helpers (require, positive_number)
  context.py        HardwareContext — bundles all catalogs + lazy operator packages
  links.py          LinkClass / LinkCatalog — interconnect link types
  topology.py       TopologyLevel / TopologySpec / TopologyPlacement / TopologyCatalog
  placement.py      build_topology_placement() — maps workers to topology indices
  device/
    __init__.py     re-exports all public symbols (import path unchanged)
    dtypes.py       canonical dtype names, byte sizes, compute-pipeline mapping and fallbacks
    sourced_value.py  SourcedValue — a number with provenance
    spec.py         DeviceSpec — one accelerator's full specification
    catalog.py      DeviceCatalog — alias-aware collection of DeviceSpec
```

## Evidence grades

Every `SourcedValue` stores a number, its source and an evidence grade describing how it was obtained.

| Grade | Name | Definition | Example |
|-------|------|-----------|---------|
| **A** | Official spec or first-party measurement | Value from vendor datasheet, whitepaper, or your own benchmark on the target hardware. | H100 HBM3 bandwidth 3.35 TB/s (NVIDIA datasheet) |
| **B** | Public benchmark | Third-party result that is published, citable, and reproducible. Not measured by this project. | MLPerf submission throughput, nccl-tests community results |
| **C** | Calibrated estimate | Derived by fitting a formula against measured data (grade A or B). The number is not directly observed but is constrained by observations. | `memory_efficiency = 0.80` fitted from A100 TRT-LLM GEMM table |
| **D** | Analytical default | An analytical or architectural default parameter used when device-specific measurements or calibrations are not supplied. | `gemm_mfu = 0.70` default for an uncalibrated GPU |

Apply the following rules when assigning evidence grades.

- Results should report the lowest grade among their inputs.
- `calibrate` upgrades D-grade analytical parameters to C.
- Imported AIConfigurator measurements of individual operators are marked grade A, and complete module measurements are marked grade B.
- Use grade D when a value's evidence is unclear, and revise the grade when supporting measurements become available.

## Analytical parameters (`DEFAULT_ANALYTICAL_PARAMETERS`)

When measured tables cannot answer a query, the analytical model uses efficiency parameters to estimate execution time from floating-point operation counts (FLOPs) and transferred bytes.
The roofline formula takes the slower of compute and memory transfer time, then adds overhead.

```
latency = max(flops / (peak * mfu), bytes / (bandwidth * mem_eff)) + overhead
```

Parameters are grouped by device family: `nvidia_gpu`, `groq_tsp` and `generic`.
GEMM means matrix multiplication, MoE means mixture-of-experts computation, and a collective is a communication operation involving a group of workers.

| Parameter | Meaning | nvidia_gpu default | groq_tsp default |
|-----------|---------|-------------------|-----------------|
| `gemm_mfu` | Fraction of peak compute a large GEMM achieves | 0.70 | 0.80 |
| `attention_mfu` | Fraction of peak compute for attention kernels | 0.45 | 0.60 |
| `moe_mfu` | Fraction of peak compute for grouped/fused MoE | 0.50 | 0.70 |
| `memory_efficiency` | Fraction of peak memory bandwidth achieved | 0.85 | 0.90 |
| `elementwise_memory_efficiency` | Memory bandwidth efficiency for elementwise ops | 0.70 | 0.90 |
| `kernel_launch_us` | Fixed per-kernel launch overhead (microseconds) | 4.0 | 0.2 |
| `collective_launch_us` | Software launch cost of one collective (microseconds) | 8.0 | 0.5 |
| `gemm_small_m_knee` | Tensor core tile size; m below this wastes compute | 64.0 | 1.0 |

All family defaults are grade D.
Device YAML files override individual parameters in `analytical:` and identify their source and evidence grade with `source_id` and `grade`.
These are exposed as `DeviceSpec.analytical_source_id` and `DeviceSpec.analytical_grade`.
The NVIDIA GPU files use AIConfigurator's grade C corrections: `memory_efficiency 0.8`, `kernel_launch_us 3.0` and `collective_launch_us 10.0`.
`operator_data.cli calibrate` fits the parameters from measured data.

## Reserved device memory

`memory.reserved_bytes`, a `SourcedValue` with default 0, reserves device memory for NCCL communication buffers, the CUDA context and framework workspace.
`DeviceSpec.usable_memory_bytes` and `CacheConfig` subtract this reserve before sizing the attention key/value (KV) cache.
The NVIDIA GPU files use AIConfigurator's 392 MB + 3.5 GB figure (grade C); the Intel B60 file uses its 500 MiB value.

## Compute-pipe fallbacks

`DeviceSpec.peak_compute_for(dtype)` selects the compute precision for a stored data type using the rules in `dtypes.py`.
Weight-only formats (`int8_wo`, `int4_wo`, `mxfp4_wo`) use fp16 compute, `int4_a8` and `fp8_block` use fp8, and `mxfp4` uses fp4.
When the device has no compute units for a precision, `COMPUTE_PIPE_FALLBACKS` tries the next supported precision.
The fallback sequences are fp4 -> fp8 -> fp16, int4 -> int8 -> fp16, bf16 <-> fp16 and tf32 -> fp32.
fp8 has no fallback, so a device without fp8 tensor cores reports an error for fp8 GEMM queries.
