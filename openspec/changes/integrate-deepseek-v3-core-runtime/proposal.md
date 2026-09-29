## Why

DeepSeek-V3 当前虽然已有标准 `ModelSpec`、MLA operator-table 适配和可运行的 AF decode，但完整模型执行仍依赖 `scripts/deepseek_v3_runtime.py` 中的专用 runtime factory 与 `V3Executor`。这使 embedding、final norm、LM head、token selection 和模型边界 collective 游离于通用 TokenSim 执行路径之外，标准 `benchmark.py -> LLMEngine` 无法像运行其他模型一样自然运行 V3。

同时，AF 通信和执行性能存在已经确认的缺口：通信配置、文档和测试要求 host submit、transport、completion 三段时序，但当前 executor 只执行 transport；跨域 AF 传输没有按 endpoint topology 定价，P2P oversubscription 公式方向错误；trace 关闭后仍构造完整事件；每个 decode step、每层都重复创建和校验高度相同的 DAG。长序列 V3 QPS 扫描因此既可能产生不正确的通信时间，又需要不可接受的墙钟时间。

## What Changes

- 将完整 Transformer 模型边界执行迁入 TokenSim 核心 runtime，使 V3 通过标准模型、集群、workload、scheduler、KV cache 和结果导出路径运行，不再需要 V3 专用 executor。
- 让通用模型执行器依据 `ModelSpec` 自动处理 MLA、Dense/MoE 混合层、Shared/Routed Experts 和模型边界算子；V3 脚本缩减为可选的薄 CLI/实验配置。
- 恢复 AF 通信的 host submit、transport、completion/synchronize 三阶段语义，严格区分 host、fabric 和 dependency wait；分段模式拒绝口径不明或已包含 host 间隙的通信测量。
- 让跨域 endpoint leg 使用实际 physical endpoints 和 topology path 定价，并修正 P2P oversubscription；明确 fabric capacity 是独立 lane 数，不把并发槽隐式解释为无限共享带宽。
- 在 trace 关闭时于构造 `AFTraceEvent` 和深层 metadata 前短路，保证关闭 trace 不改变模拟时间和结果。
- 优化 trace 开启时的历史管理：按 iteration 建立事件位置索引，完整历史事件只保留一份；每轮通过索引完成校验和单遍资源统计，完整事件字典延迟到结果导出时生成，保持对外 JSON 字段、事件顺序和数值不变。
- 将 AF planning 拆成不可变 DAG template 编译与逐轮动态 binding；复用静态依赖、placement、tensor ownership、collective plan 和资源映射，同时保留每轮独立的 SimPy events、resource requests、context、routing 和排队状态。
- 缓存相同规范化 query 的 operator/transfer service estimate；仅融合相同物理资源上严格串行、无外部可见中间依赖的 compute components，减少 SimPy 事件而不跨越通信或分支边界。
- 在已有 DAG/service 缓存上继续减少命中路径开销：仅为当前使用的配置预计算静态指纹，采用静态 query 骨架与动态 key，在命中时避免完整 query 和临时 shape/dtype 容器构造；不枚举所有计算配置或预估所有 shape。
- 增加优化前后等价性、通信时序、拓扑、trace 开关和 V3 标准入口回归，并记录墙钟性能基线。
- 迁移三个遗留测试模块到现有 operator-table API，保留公式、KV、完整执行、路由与测量覆盖的行为断言；明确旧脚本测试工具和 shape-scoped calibration API 的退役及当前按表校准替代契约。

## Capabilities

### New Capabilities

- `deepseek-v3-core-runtime`: 将 DeepSeek-V3/MLA 作为 TokenSim 标准模型执行能力，而不是脚本级专项实验器。
- `af-simulation-performance`: 定义 trace 快路径、DAG 模板复用、估值缓存和安全事件融合的结果等价契约。

### Modified Capabilities

- `af-execution`: 增加模型边界执行、三阶段通信、endpoint-aware topology 定价和统一资源/trace 语义。
- `operator-data-latency-provider`: 要求规范化查询可安全缓存，且通信 timing scope/provenance 不因缓存丢失。

## Impact

- 影响 `TokenSim/af/executor.py`、`planning.py`、`runtime.py`、`resources.py`、`operator_provider.py`、`trace.py`、`TokenSim/comm/collectives.py`、`LLMEngine` 和标准 benchmark 配置入口。
- 新增通用 whole-model AF executor/planner 语义；迁移并最终删除 `scripts/deepseek_v3_runtime.py` 中的 `V3Executor` 和核心 runtime factory。
- 将 V3 A/F 拓扑与硬件假设迁入 `data/clusters`/实验配置，模型事实继续保留在 `data/models`。
- 保持非 MLA、AF-disabled 和现有模型兼容；优化不得改变模拟时间、请求完成顺序、队列区间、通信字节或 operator provenance。
- 后续缓存快路径涉及 `templates.py`、`estimate_cache.py`、executor/provider 查询边界及专项验证；以当前缓存已开启的实现为性能对照，明确配置生命周期和失效契约，不承诺未经测量的提速比例。
- Trace 管理增量涉及 `trace.py`、模型边界事件替换、`LLMEngine` 摘要记录和 `util/results.py` 等导出调用方；保留全部历史和现有校验范围，降低运行期重复扫描与序列化副本成本。最终整份 JSON 的导出内存峰值仍存在，本轮不引入流式导出或历史淘汰。
- 本变更不实现 checkpoint 推理、MTP，也不直接实现跨多个稳定 decode steps 的周期跳跃；多-token 批量推进需另行证明动态状态等价性。


### 核心路由修复（用户已确认）

Synthetic routing 按 seed、phase、layer、request 和 token identity 确定 expert，排除 AF block、iteration 和调用顺序。Uniform 使用循环均匀分配，hot/skew/burst 使用每 token 独立的确定性采样且 top-k 无重复；热点不足时从剩余 experts 补齐。旧统计型 histogram API 保持原口径，显式外部 assignments 保持不变。该修复会改变旧 synthetic 仿真结果，需重新验证优化开关下的结果等价性。
