# Operator-data sources and licenses

NVIDIA AIConfigurator measurements are imported under Apache-2.0 from commit `f254959eb89e2f206b8f9a77051644d7c1cbdb89`.
The retained license is `third_party/licenses/AIConfigurator-Apache-2.0.txt`, and attribution is in `NOTICE`.
Each `generation_meta.yaml` identifies the sources, upstream directories, collection versions, hardware, framework and operations included in each measured time.

The bundled data includes matrix multiplication (GEMM), prompt and decode attention, mixture-of-experts (MoE) compute, group communication (collectives) and transfers between expert workers.
It also includes complete sparse-attention modules, mHC modules that mix residual streams and MegaMoE modules that combine expert computation with token transfers.
MegaMoE tables use SGLang measurements in each backend's package.
The B200 and B300 vLLM packages explicitly use GB200 and GB300 measurements.
Rows retain their evidence grades and the buffer capacities used during measurement.
Module source IDs are registered by the importer.

V100 NCCL curves derive from the project's original nccl-tests measurements.
Groq analytical tables are generated project data.
Local profiling runs use `aiconfigurator-collector:` source IDs and retain records of how the measurements were collected.

See [AIConfigurator usage](../../docs/aiconfigurator.md) for importing the recorded revision, validating data and selecting which operations a timing estimate includes.
The importer does not convert unsupported measurement types into other operator types.
