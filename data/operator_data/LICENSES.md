# Operator-data sources and licenses

NVIDIA AIConfigurator measurements are imported under Apache-2.0 from commit
`f254959eb89e2f206b8f9a77051644d7c1cbdb89`. The retained license is
`third_party/licenses/AIConfigurator-Apache-2.0.txt`; attribution is in `NOTICE`.
Each `generation_meta.yaml` records its source IDs, exact upstream directories,
collection versions, hardware, framework and timed scope.

The bundled data includes ordinary GEMM, context/generation attention, MoE,
collectives and expert communication, plus complete sparse-attention, mHC and
MegaMoE module tables. B200/B300 fused-expert measurements borrow GB200/GB300
SGLang sources explicitly. Source grades and collection capacities remain
attached to the rows. Module source IDs are registered by the importer.

V100 NCCL curves derive from the project's original nccl-tests measurements.
Groq analytical tables are generated project data. Local collector runs carry
`aiconfigurator-collector:` source IDs and retain their collection provenance.

See [AIConfigurator usage](../../docs/aiconfigurator.md) for pinned import,
validation and cost-scope instructions. Unsupported measurement families are
not silently converted to another operator type.
