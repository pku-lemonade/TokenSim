## Why

TokenSim 已能表达并校验 Prefill/Decode 的 Attention-FFN 分离拓扑，但合法的 `af.enabled=true` 配置仍会在模拟开始前被拒绝，无法验证逐层依赖、micro-batch 流水和共享通信资源竞争。现在需要把配置契约落实为可运行、可观测且可由确定性 oracle 验证的 AF 执行路径，为后续接入实测 kernel 数据奠定独立于性能数据精度的执行机制基线。

## What Changes

- 新增 AF iteration、block 和逐层 stage 执行语义，将所选阶段展开为 `A compute -> A-to-F transfer -> F compute -> F-to-A transfer`，并在返回 A 后进入下一层。
- 按 `af_block_size` 将 scheduler 选出的同阶段 requests 确定性分组；Prefill 与 Decode 均可进入 AF 路径，未选择的阶段继续使用原完整阶段路径。
- 为 A worker、F worker 和 AF link 资源建立共享 SimPy 竞争域，遵守配置中的 worker 数量、全/半双工资源键和容量，使多个 block 能流水重叠且不能在容量为 1 的同一资源上重叠服务。
- 引入最小可注入 compute/transfer latency provider 契约。Provider 只返回非负有限的 service time；资源申请产生的 queue time 由 executor 单独计算。正式 AF 路径不得回退到完整 iteration Roofline/LLMCompass 时延。
- 在 iteration barrier 后统一推进 requests，保证每个 request 每轮恰好推进一次，且 AF block 不改变 KV Cache block 的分配与所有权。
- 输出结构化 AF trace，记录 ready/start/end、queue/service、资源、方向、block、layer、request/token 数以及 iteration barrier，支持重建关键路径和利用率。
- 增加从真实集群配置进入模拟入口的端到端测试矩阵，覆盖单/多层、双 block 流水、A/F 瓶颈、全/半双工、通信竞争、多 F worker、Prefill/Decode 选择和 AF 关闭回归。
- 使用确定性 stub providers 验证执行机制：双 block 单层在 `A=2 ms`、`A-to-F=1 ms`、`F=3 ms`、`F-to-A=1 ms`、各资源容量为 1 时，单 block 为 `7 ms`，完全串行为 `14 ms`，正确流水 makespan 为 `10 ms`。
- 本 change 不实现 operator-data Parquet 加载、实测 kernel 数据、插值、公式校准或真实系统精度结论；这些由后续 `add-operator-data-latency-provider` change 完成。

## Capabilities

### New Capabilities

- `af-execution`: 定义 Prefill/Decode AF block 的逐层状态机、资源调度、流水并行、iteration barrier、provider 边界、trace 和端到端验收行为。

### Modified Capabilities

- `af-separation-config`: 将合法且已启用的 AF 配置从阶段性运行拒绝改为在 executor 及所需 provider 可用时进入 AF 执行路径，并保留缺失运行依赖时的显式失败语义。

## Impact

- `TokenSim/af/`: 新增 AF block/iteration、builder、资源、executor、latency provider 接口和 trace 模块。
- `TokenSim/llm/llm_engine.py`: Engine 级持有共享 AF executor，在所选 Prefill/Decode 阶段接入 AF iteration，并在 barrier 后保持现有 request 推进语义。
- `TokenSim/config/cluster_config.py` 与配置导出：向运行时提供规范化 phase/link/resource 映射，不改变既有配置字段语义。
- `benchmark.py`: 移除“AF executor 尚未实现”的固定拒绝，在运行依赖完整时启动 AF 模拟并支持 trace 输出。
- `tests/af/` 与现有回归测试：增加真实入口端到端事件时间线 oracle、资源竞争断言和非 AF 兼容验证。
- 依赖继续使用仓库已有的 SimPy；本 change 不新增 AIConfigurator、InferSim、Parquet 或硬件数据运行依赖。
