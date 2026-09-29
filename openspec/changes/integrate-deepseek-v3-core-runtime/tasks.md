## 1. 基线与正确性修复

- [x] 1.1 建立 V3 小规模标准 workload、现有专用入口和 AF-disabled 的结果基线，记录请求完成时间、makespan、queue、通信字节、operator provenance 与墙钟 profiling。
- [x] 1.2 在 trace disabled 时于所有 `AFTraceEvent` 和深层 metadata 构造前短路；增加 enabled/disabled 模拟结果等价测试与墙钟基准。
- [x] 1.3 修正 P2P oversubscription 公式并增加同节点、跨节点、跨机架单调性测试。
- [x] 1.4 迁移 MLA、完整 V3 decode 和 operator batching 三个遗留测试模块；保留行为覆盖，记录旧 shape-scoped calibration 退役与按表校准替代，验证非法校准参数、缺数据失败和实测值不重复校准；修复核心 synthetic routing 的分块独立性。

## 2. AF 通信语义

- [x] 2.1 将 endpoint leg/collective 执行拆为 host submit、fabric transport、completion/synchronize，接入 per-worker/shared host resources 和固定成本校验。
- [x] 2.2 强制分段模式只接受 `transport_service` provider scope，拒绝 `legacy_unspecified`/`eager_interval`，并验证 fixed cost 不重复计入。
- [x] 2.3 让跨域传输按实际 physical endpoints 解析 topology path、link 与 oversubscription；验证 endpoint legs、full/half duplex、collective 与跨 block 竞争。
- [x] 2.4 明确并校验 fabric capacity 的独立 lane 语义；拒绝同一 resource key 上冲突的 network/duplex/capacity 定义。

## 3. V3 核心运行时

- [x] 3.1 在核心库实现通用 whole-model AF execution，覆盖 embedding、逐层 Dense/MoE、final norm、LM head、token selection 和必要 boundary collectives。
- [x] 3.2 将 MLA、latent KV ledger、Shared/Routed Expert 与混合层选择接入标准 `LLMEngine`、scheduler、KV cache、workload 和结果导出路径。
- [x] 3.3 将 V3 A/F 拓扑和硬件假设迁入数据配置；使 `benchmark.py` 或等价标准入口可选择 V3 模型和集群运行。
- [x] 3.4 将 `scripts/deepseek_v3_runtime.py` 缩减为薄 CLI/兼容包装，删除 `V3Executor` 与脚本级核心执行语义。

## 4. DAG 与估值复用

- [x] 4.1 将 planner 拆为不可变 DAG template compile 与逐轮 bind，定义包含 topology、dtype、layer type、shape class 和 routing structure 的安全缓存键。
- [x] 4.2 复用静态节点、依赖、placement、tensor ownership、collective plan 和资源 identity；每轮仅创建动态 binding 与必要 SimPy 状态。
- [x] 4.3 为规范化 compute/transfer query 增加 service estimate cache，保留全部 provenance、timing scope、payload 和 component details。
- [x] 4.4 增加缓存命中/未命中差分测试，覆盖 batch/context 变化、routing skew、replica 轮转、TP/EP、dtype 和 operator fallback。

## 5. 减少事件和验证性能

- [x] 5.1 识别同一不可抢占 compute service 内可融合的连续 components，合并 timeout/resource request但保留 component breakdown；禁止跨通信、分支和 worker 融合。
- [x] 5.2 为 trace-off fast path 使用紧凑节点索引和轻量运行状态，避免重复 node-id、metadata、tensor/routing 容器复制。
- [x] 5.3 对 61 层 V3、固定 batch 的多 decode-step workload 记录构图、lookup、SimPy、trace 和总墙钟时间，验证优化收益且模拟输出等价。
- [x] 5.4 运行完整回归和 OpenSpec 校验；记录仍未实现的稳定周期/多-token 跳跃为独立后续变更，不在本变更中用近似乘法替代离散事件模拟。

## 6. 已有缓存命中路径优化

- [x] 6.1 保存当前 DAG/service 缓存均开启的性能基线；在无完整 service-record observer 的普通路径做函数级 profiling，定位静态 key、动态 key、query 构造及 lookup 成本。
- [x] 6.2 明确 runtime 配置、lookup policy 和 dataset 的不可变快照或显式版本/失效契约；仅为实际使用的配置预计算静态指纹，不枚举 shape/context/routing 组合，并覆盖配置更新或拒绝更新的行为。
- [x] 6.3 在 DAG/service 缓存中复用静态指纹，保留完整动态结构与估值身份；验证 topology、dtype、context、routing、ownership、participants、endpoints、policy 和 dataset 变化的隔离，避免每次重新递归扫描整份配置。
- [x] 6.4 引入静态 query 骨架与动态 key，在内置 provider 命中时跳过完整 query 和仅供查询的临时容器构造；miss、缓存关闭及兼容 provider 保留完整路径。仅对可信不可变模板复用静态校验，保留动态、手工节点、timing-scope 和错误诊断校验。
- [x] 6.5 增加命中不构造 query、未命中按需构造、配置失效、非法输入与兼容 provider 测试；对照既有缓存路径及缓存关闭路径，验证 trace 开关下 service/provenance、queue、通信字节、完成顺序和模拟时间等价。
- [x] 6.6 对 61 层 V3 短/长 decode 重复测量本轮前后缓存开启路径，分别报告冷启动/热缓存、wall/CPU、波动、命中率和 query 构造次数；差分观测单独运行，更新验证文档并执行完整回归与 OpenSpec 校验，如实记录收益或回退。

## 7. Trace 索引与单份历史存储（已实施）

- [x] 7.1 保存当前 trace enabled/disabled 的完整事件、聚合统计及标准入口导出 JSON 基线；审计事件追加/替换、摘要消费和导出调用方，记录执行期与导出期成本及现有历史扫描访问次数。
- [x] 7.2 在 recorder 维护 iteration 到全局事件位置的索引，追加时同步更新；使检索和校验按索引读取本轮全部有序事件，原位置替换后读取最新事件，保留全部历史；若存在 reset/clear 入口，同步维护索引。
- [x] 7.3 将 resource utilization 改为按事件顺序单遍累计，保留多资源去重、capacity、零 makespan 和稳定输出顺序；每轮生成轻量聚合摘要，不深层序列化完整事件。
- [x] 7.4 调整 LLMEngine 摘要及所有相关消费方，在结果导出边界按索引生成完整事件字典；保持外部 JSON schema、顺序和数值，重复导出无副作用且不缓存完整副本回 engine。
- [x] 7.5 增加交错 iteration、模型边界事件替换、空记录、trace-off、多资源统计与重复导出的差分测试；验证新事件不会修改既有导出结果、轮次结束不调用完整事件序列化，以及固定每轮事件量的 8/16/32-step 检索访问量线性增长。
- [x] 7.6 对标准 V3 serving 验证模拟结果和导出等价，分别记录执行期/导出期 wall、CPU 和内存；明确完整历史及整份 JSON 导出的内存限制，更新验证文档，运行完整回归与 OpenSpec 严格校验。
