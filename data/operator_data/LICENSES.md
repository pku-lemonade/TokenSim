# Third-party data in `data/operator_data`

| Package | Origin | License | Notes |
| --- | --- | --- | --- |
| `a100_sxm_80g/trtllm`, `h100_sxm/trtllm`, `h200_sxm/trtllm`, `gb300/trtllm` | [NVIDIA AIConfigurator](https://github.com/ai-dynamo/aiconfigurator), `aic-core/src/aiconfigurator_core/systems/data/<system>/`, commit `f254959eb89e2f206b8f9a77051644d7c1cbdb89` | Apache-2.0 | Measured TensorRT-LLM / NCCL kernel latencies, converted from milliseconds to microseconds by `TokenSim/operator_data/importers/aiconfigurator.py`. The `source_id` of every row points to the exact upstream file. |
| `v100_sxm2/nccl` | TokenSim legacy `TransformerRoofline/allreduce_v100.xlsx` (repository history) | Project data | nccl-tests all_reduce_perf on a V100 node, fp32, 2/4/8 GPUs. |

Regenerate the AIConfigurator packages with:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/ai-dynamo/aiconfigurator.git /tmp/aiconfigurator
git -C /tmp/aiconfigurator sparse-checkout set aic-core/src/aiconfigurator_core/systems/data/a100_sxm
python -m TokenSim.operator_data.cli import-aiconfigurator \
  --system-dir /tmp/aiconfigurator/aic-core/src/aiconfigurator_core/systems/data/a100_sxm \
  --device a100_sxm_80g --backend trtllm --upstream-commit "$(git -C /tmp/aiconfigurator rev-parse HEAD)"
```
