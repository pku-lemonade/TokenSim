# TokenSim/hardware/

Hardware catalog: device specs, link classes, hierarchical topologies.

Replaces the old monolithic `TransformerRoofline` hardware table. Every
numeric field carries a `source_id` and evidence grade so that official specs,
project assumptions, and calibrated values can be told apart in results.

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
    dtypes.py       canonical dtype names, byte sizes, compute-pipeline mapping
    sourced_value.py  SourcedValue — a number with provenance
    spec.py         DeviceSpec — one accelerator's full specification
    catalog.py      DeviceCatalog — alias-aware collection of DeviceSpec
```

## Evidence grades

Every `SourcedValue` carries a grade that says how trustworthy the number is.

| Grade | Name | Definition | Example |
|-------|------|-----------|---------|
| **A** | Official spec or first-party measurement | Value from vendor datasheet, whitepaper, or your own benchmark on the target hardware. | H100 HBM3 bandwidth 3.35 TB/s (NVIDIA datasheet) |
| **B** | Public benchmark | Third-party result that is published, citable, and reproducible. Not measured by this project. | MLPerf submission throughput, nccl-tests community results |
| **C** | Calibrated estimate | Derived by fitting a formula against measured data (grade A or B). The number is not directly observed but is constrained by observations. | `memory_efficiency = 0.80` fitted from A100 TRT-LLM GEMM table |
| **D** | Project assumption | A default with no hardware-specific evidence. May be reasonable but has not been validated for this device. | `gemm_mfu = 0.70` as a starting point for an untested GPU |

Rules:
- Results should report the lowest grade among their inputs.
- `calibrate` upgrades D-grade analytical parameters to C.
- Importing AIConfigurator measured tables produces A-grade rows.
- When a value grade is unclear, use D; upgrading later is safe, downgrading risks false confidence.

## Analytical parameters (`DEFAULT_ANALYTICAL_PARAMETERS`)

When no measured operator table covers a query, the analytical model uses these
efficiency parameters to convert theoretical FLOPs/bytes into a latency
estimate via the roofline formula:

```
latency = max(flops / (peak * mfu), bytes / (bandwidth * mem_eff)) + overhead
```

Parameters are grouped by device family (`nvidia_gpu`, `groq_tsp`, `generic`):

| Parameter | Meaning | nvidia_gpu default | groq_tsp default |
|-----------|---------|-------------------|-----------------|
| `gemm_mfu` | Fraction of peak compute a large GEMM achieves | 0.70 | 0.80 |
| `attention_mfu` | Fraction of peak compute for attention kernels | 0.45 | 0.60 |
| `moe_mfu` | Fraction of peak compute for grouped/fused MoE | 0.50 | 0.70 |
| `memory_efficiency` | Fraction of peak memory bandwidth achieved | 0.85 | 0.90 |
| `elementwise_memory_efficiency` | Memory bandwidth efficiency for elementwise ops | 0.70 | 0.90 |
| `kernel_launch_us` | Fixed per-kernel launch overhead (microseconds) | 4.0 | 0.2 |
| `gemm_small_m_knee` | Tensor core tile size; m below this wastes compute | 64.0 | 1.0 |

All defaults are grade D. Device YAMLs can override individual parameters in
their `analytical:` section, and `operator_data.cli calibrate` can fit them
from measured data (producing grade C values).
