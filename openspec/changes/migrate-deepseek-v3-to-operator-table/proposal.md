## Why

合并 `afd-data` 分支后，TokenSim 的运行时已切换到 `ModelSpec`、`HardwareContext` 和 operator-table 后端，但 DeepSeek-V3 仍停留在旧的 JSON 模型描述和旧 `operator_data` provider API。结果是模型目录无法发现 V3，现有 MLA/完整 V3 脚本在导入阶段失败，无法验证 671B MoE + MLA 的 AF 仿真。现在需要完成一次有边界、可追溯的适配，让 V3 能通过新后端运行并保留现有 AF 语义。

## What Changes

- 将 DeepSeek-V3 671B 的模型描述转换为可被 `ModelCatalog` 加载的 `ModelSpec` YAML。
- 扩展 `ModelSpec` 对 MLA 所需结构字段的表达和校验，包括 KV/Q low-rank、RoPE/NoPE/V head dimensions 以及 MLA attention type。
- 将 DeepSeek-V3 的 dense layer、MoE layer、routed/shared expert、top-k、词表和 BF16 配置完整映射到新模型规格。
- 迁移 DeepSeek-V3 MLA 和完整 decode 入口，使用 `HardwareContext`、`OperatorResolver` 及新 compute/transfer providers。
- 保留旧脚本的可观测 profiling、AF 拓扑和 decode-only 行为，并让缺失 operator-table 记录时明确失败。
- 增加模型加载、MLA shape、完整 V3 runtime plan 和端到端执行回归验证。

## Capabilities

### New Capabilities

- `deepseek-v3-operator-table-adapter`: 在新配置和 operator-table 后端上加载 DeepSeek-V3 MLA 模型并运行完整 AF decode 仿真。

### Modified Capabilities

- 无。现有 operator-data provider 和 AF execution 的行为契约不变，本变更只增加 DeepSeek-V3 的模型适配和入口迁移。

## Impact

- 影响 `TokenSim/config/model_config.py` 及模型 catalog/schema 校验。
- 新增或转换 `data/models/deepseek-v3-671b-mla.yaml`，并更新相关模型索引/文档。
- 重写 `scripts/run_deepseek_mla.py` 与 `scripts/run_deepseek_v3_full.py` 的模型加载、runtime plan 和 provider 初始化。
- 依赖新 `TokenSim.operator_table`、`TokenSim.hardware`、`TokenSim.af` API；不再依赖已删除的 `Calibration`、`ModelOperatorManifest`、`OperatorData*Provider` 接口。
- 影响测试、profiling 输出 schema 和 V3 仿真运行命令；不改变已有非 V3 模型和 AF-disabled 路径。
