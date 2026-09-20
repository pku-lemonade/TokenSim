## Context

当前 `HardwareContext.load()` 已经统一加载设备、拓扑、模型目录和 operator-data package，但模型目录只读取 YAML；现有 DeepSeek-V3 入口仍构造旧的 metadata namespace，并调用已删除的 operator-data lookup/provider。现有 `ModelSpec` 和通用 operator-table resolver 主要按 MHA shape 计算，仓库中另有独立的 MLA geometry/decomposition 代码，二者尚未连接。详细行为要求见本变更的 `specs/deepseek-v3-operator-table-adapter/spec.md`。

## Goals / Non-Goals

**Goals:**

- 让 DeepSeek-V3 以标准模型 catalog 条目被解析，并在解析阶段验证 MLA 与 MoE 结构。
- 将 MLA attention decomposition 接入 operator-table resolver，保留 exact/interpolation/missing-data 的统一策略。
- 将 MLA smoke 和完整 V3 decode 入口统一到 `HardwareContext -> ModelSpec -> compile_af_runtime_plan -> providers -> AFExecutor` 路径。
- 保留现有 AF block、TP/DP/EP placement、通信 queue/service trace、profiling 和模型边界操作的可审计性。
- 用小型 fixture 和 V3 shape/runtime 回归覆盖模型加载、runtime plan、provider query 及完整 decode。

**Non-Goals:**

- 本变更不加载 DeepSeek-V3 checkpoint，也不在 GPU 上执行真实推理。
- 本变更不实现 V3 的 MTP、完整 serving engine、权重 FP8 数值误差或逐 rank prefill；入口只承诺已声明的 decode/MLA smoke 阶段。
- 本变更不重新设计 operator-data schema、通信竞争语义或已有非 MLA 模型的时延公式。
- 本变更不把理论/analytical 数据伪装成目标硬件实测数据。

## Decisions

### 1. 在 ModelSpec 中加入可选 MLA 配置

新增一个可选的 MLA 结构字段（包含 `attention_type` 和五个维度），普通 MHA 模型保持现有字段和派生属性。MLA 模型解析时要求字段齐全、正数、RoPE 维度为偶数，并检查 query heads、TP 和 cache layout 的约束。MLA 的 `q_dim`、`kv_dim`、cache bytes 和 local shape 由该配置派生；不会让调用方继续把 `num_key_value_heads * head_dim` 当作 MLA KV 宽度。

选择在 ModelSpec 里保存结构，而不是在脚本中挂临时属性，是因为 runtime plan、memory ledger、provider 和 profiling 都必须看到同一份模型契约；代价是需要更新序列化、校验和少量通用 dimension helper。

### 2. 使用 YAML 作为唯一 catalog 输入，保留 JSON 作为来源资料

把现有 V3 JSON 的已确认字段转换成 `data/models/deepseek-v3-671b-mla.yaml`，将 HF config URL、参数口径、BF16 architectural simulation 和 routing 假设放入 `sources`/说明字段。`ModelCatalog` 继续只消费 YAML，避免每个入口拥有不同的 JSON 解析语义；原 JSON 保留为参考/迁移输入，不作为 runtime fallback。

### 3. 在 operator-table AF provider 中增加 MLA composition 分支

复用已有 MLA decomposition 的算子边界，但每个 component 通过新 resolver 查询：Q down/norm/up、KV down/norm、RoPE、decode 的 Q absorb、MLA attention core、decode 的 value expand、output projection 以及必要 residual/norm。MLA core 使用包含 phase、execution mode、context lengths、latent/head dimensions 和 dtype roles 的专用 key；decode 采用 absorbed 模式，prefill 只保留已实现的受限 smoke 语义。所有结果继续返回 component breakdown、source id、dataset version 和 match type。

这样可以让 V3 共享 operator-table 的 exact/interpolation、backend 选择和缺失记录错误；不在脚本里复制一套 Roofline 或旧 lookup fallback。若目标 package 没有 MLA 专用表，只有在既有 provider policy 明确允许时使用同一版本的 analytical candidate，并在结果中标为 analytical。

### 4. 统一入口的依赖注入顺序

V3 入口按以下顺序构造运行时：

1. `HardwareContext.load(data_root)`；
2. `hardware.model("deepseek-v3-671b-mla")`；
3. 从 cluster 配置解析 worker/domain 和 topology placement；
4. `compile_af_runtime_plan(cluster_config, worker_configs, model, hardware, ...)`；
5. 用 `OperatorTableAFComputeProvider(hardware, model, ...)` 和 `TopologyAFTransferProvider(hardware, cluster, workers, placement, ...)` 创建 providers；
6. 把两者注入现有 `AFExecutor`，由 executor 负责 queue、resource contention、DAG overlap 和 trace。

脚本只负责实验参数、请求和输出目录，不再创建 `SimpleNamespace` model metadata，也不直接实例化旧 `OperatorData*` 类。完整 V3 与 MLA smoke 共用一个解析 helper，避免两套模型字段或 dtype defaults 漂移。

### 5. 模型边界操作沿用同一 resolver

embedding、final norm、LM head 和 token selection 作为 boundary queries 使用对应设备的 operator-table resolver；boundary service time 与 transformer compute 一样进入 A 资源和 profiling。TP collective 仍由 topology/collective provider 估计，logical payload 在 planner 边界保持未乘算法系数，避免把通信或排队混入 compute composition。

### 6. 以可审计失败替代隐式兼容 fallback

迁移完成后，旧 provider API 不再作为 V3 入口的兼容层。lookup miss、MLA shape 不完整、非法 TP/EP 或缺失通信 fabric 均在 event loop 前或对应 query 处报出完整上下文。这样能区分“模型结构未接入”“operator 数据缺失”和“拓扑配置错误”，也避免生成看似完整但不可追溯的 V3 结果。

## Risks / Trade-offs

- [Risk] 现有 operator package 可能没有 MLA 专用 shape，导致完整 V3 首次运行失败。→ 先提供 shape discovery/backlog 输出；只允许已声明的 analytical policy 填补，并在 profile 中标注证据等级。
- [Risk] 通用 cache/runtime plan 仍假设普通 head dimension。→ 将 MLA cache width、KV rank 和 per-rank layout 集中在 ModelSpec/MLA helper，并为 memory ledger 与 plan 增加专门回归。
- [Risk] 旧脚本外部用户依赖 JSON 参数或旧 CLI。→ 保留脚本路径和主要实验参数，入口内部迁移；对不再支持的旧参数给出明确弃用/错误信息。
- [Risk] 完整 61 层 trace 运行时间和输出体积较大。→ 默认支持小 batch、单 step 和可关闭详细 trace；测试使用单层/小 batch fixture，端到端验收再运行代表性完整配置。
- [Risk] V3 官方 checkpoint 使用 FP8，而当前目标是 BF16 architectural simulation。→ YAML 明确记录 compute/storage 口径和来源；不得把 BF16 模拟结果宣传为 FP8 checkpoint 实测。

## Migration Plan

1. 扩展 ModelSpec/MLA geometry 和 YAML loader，添加 V3 YAML，先通过 catalog/shape 单元测试。
2. 在 operator-table provider 中加入 MLA component resolver，并补齐 missing/exact/interpolation/provenance 测试。
3. 迁移共享 V3 runtime factory，再迁移 MLA smoke 和完整 decode 入口，删除对旧 provider 的导入。
4. 运行 AF-disabled 回归、MLA 单层回归、V3 小 batch 完整 decode 和 profiling 校验；记录 operator coverage/backlog。
5. 若新入口出现回归，可回退到迁移前提交；YAML 和新适配代码均为独立提交，便于逐步回退。

## Open Questions

无。未测量的 MLA shape 是否补充实测数据属于 operator-data 采集工作，不改变本适配的接口和模型契约。
