## Context

现有 AF executor 已通过 `AFComputeQuery`、`AFTransferQuery` 和 `AFServiceEstimate` 隔离时延来源，并由 SimPy 资源单独产生 queue time；参见 `proposal.md` 与 `operator-data-latency-provider`、`af-execution` delta specs。当前缺口是可用于 Ada/Groq 的正式数据包、查询与组合实现。

本设计以总任务书和 operator-data 工作书为准。Phase 1 中将 AF 限定为 Decode-only 的内容不适用：provider 必须覆盖 Prefill、Decode 或两者。AIConfigurator（研究锚点 `805a14e`）证明了按 GEMM、Attention、MoE、elementwise、communication 分表并进行来源解析、插值和组合的可行性；InferSim（研究锚点 `ae3f73d`）提供 FLOPs/内存/通信量与 kernel benchmark 校准的轻量拆解。两者只作为设计证据，不成为 TokenSim 运行依赖。

约束包括：目标硬件实测数据可能不完整；当前 requirements 未包含 Parquet reader；现有 Linux 原生扩展不能在本机 Windows CPython 3.13 直接加载；旧 backend 和 stub oracle 必须保持兼容。

## Goals / Non-Goals

**Goals:**

- 建立“离线生成 -> 版本化数据包 -> 运行时查询/组合 -> AF executor”的单向依赖。
- 让每个 service time 都能追溯到 source、exact/interpolation 和 composition breakdown。
- 以严格校验、受控一维插值和显式 missing error 避免看似可运行但不可解释的估计。
- 用真实生成数据包驱动端到端 oracle，并将物理时延误差与事件调度正确性分别验证。

**Non-Goals:**

- 不在 TokenSim 事件循环中运行 benchmark、拟合模型或即时 Roofline fallback。
- 不实现 AIConfigurator 的部署搜索/Rust-PyO3 栈，也不复制 InferSim 的 GPU 专属 MFU 常数。
- 不把网络随机拥塞、重传、乱序或尾延迟加入第一版通信模型。
- 不用 Codex、公开峰值或公式输出替代 Ada/Groq 的物理 ground truth。

## Decisions

### 1. 三层架构与职责边界

```text
offline generation
  measured samples + formulas + calibration
                 |
                 v
versioned operator-data package
                 |
                 v
runtime lookup + A/F composition ----> AFServiceEstimate (service only)
                                              |
                                              v
AF executor: SimPy resource queue + service -> trace/result
```

公式补齐只发生在离线层。运行时负责加载、查询、插值、组合和单位转换；executor 只消费秒数并调度资源。这样可以冻结实验数据版本、重放结果，并避免查询成本或模型分支污染离散事件时间。

备选方案是在 provider miss 时即时调用 Roofline/LLMCompass。该方案会混合不同粒度的时间模型且难以追溯，因此拒绝。

### 2. 数据包布局与 schema

数据包根目录包含：

```text
hardware.yaml
generation_meta.yaml
shape_manifests/<experiment-id>.yaml
gemm_perf.parquet                 gemm_analysis.parquet
context_attention_perf.parquet    context_attention_analysis.parquet
generation_attention_perf.parquet generation_attention_analysis.parquet
moe_perf.parquet                  moe_analysis.parquet
elementwise_perf.parquet          elementwise_analysis.parquet
communication_perf.parquet        communication_analysis.parquet
```

Runtime perf 表保持窄表：规范化查询键、`latency_us`、`source_id` 及必要 model/version 字段；预测中间量、FLOPs/bytes、校准残差和证据等级进入 analysis 表。`generation_meta.yaml` 记录 schema/dataset version、生成时间、工具版本、输入哈希、设备/backend、calibration/holdout manifests 和来源目录。

通信表作为工作书五张 compute 表之外的独立扩展，因为 executor 已有 transfer provider，且通信不能混入 A/F compute composition。AllReduce、MoE dispatch/combine 和 A-to-F/F-to-A 使用显式 operation/direction 键。

加载阶段先校验整个 bundle，再暴露 provider。Exact key 建立唯一索引；重复、未知版本、单位不一致、悬空 source、非有限值立即失败。备选方案是容忍脏数据并在查询时处理，拒绝原因是错误会延迟到仿真中途且破坏可复现性。

### 3. Parquet 读取采用显式轻量依赖

第一版运行时使用 `pyarrow` 直接读取 Parquet，并把版本加入 requirements/安装说明；装配正式 provider 时执行一次入口预检，缺少依赖则给出明确安装错误。核心 AF 模块继续延迟导入 reader，使 stub provider 和 AF-disabled 路径不要求 pyarrow。

选择 pyarrow 而非 pandas，是为了避免只为查询加载完整 dataframe 语义和额外依赖。备选方案 CSV 缺少稳定类型与高效列式读取；SQLite 虽便于索引，但偏离上级指定的 Parquet 交付格式。若实际部署环境已有统一数据访问层，可在不改变 spec 的前提下替换 reader。

### 4. Typed key normalization 与确定性 lookup

为每类算子定义 typed key 和规范化函数。Loader 将枚举、整数、dtype、device/backend、phase 规范化后建立不可变索引；provider query 先映射成一个或多个 operator keys。查询算法固定为：

1. 查询 exact unique key。
2. Exact miss 后锁定所有离散维度，只在该类别允许轴上取严格包围目标值的最近左右邻点。
3. 以线性插值计算 `latency_us`，记录邻点 `source_id` 和权重。
4. 无两个邻点、目标越界或离散键不同则抛出 typed missing error。

结果与表行顺序无关；比较与权重计算采用明确的整数 shape 和有限浮点约束。第一版不外推。未来若证据证明非线性插值更准确，应通过 schema/version 和独立 change 引入，而不是改变同一数据集的解释。

### 5. 离线生成与证据分级

Generator adapter 将 benchmark/public/formula 输入先转换成统一 observation。每个 source 严格采用上级 operator-data 文档的证据等级：A 为官方规格、论文中的算子级或层级实测，B 为公开端到端基准，C 为经过公开结果校准的公式估算，D 为项目初始假设。B 级数据只检查整体量级，不拆成单算子延迟；等级描述数据性质，不自动决定数值质量。

相同 exact key 的候选选择规则固定为：通过质量门槛的目标实测优先，其次按 manifest 指定的 source precedence；所有未选候选留在 analysis。公式由算子 FLOPs、读写 bytes、硬件参数和按类别拟合的 calibration 系数组成，不把设备峰值当持续性能。

Calibration 和 holdout 根据稳定 manifest/hash 在拟合前切分。报告按算子类别给出 coverage、相对误差统计、bias、bottleneck consistency 和 worst shapes；没有 A/B 级 ground truth 时报告只评价内部一致性。

### 6. A/F composition graph 防止重复计时

Provider 使用版本化 model/operator manifest 将 `AFComputeQuery` 展开为有序原子节点：

- A：RMSNorm -> QKV GEMM -> RoPE -> Attention core -> output projection GEMM -> Residual Add -> 可选 data reorder。
- F：RMSNorm -> Router/Gating GEMM -> grouped/fused expert -> 必要 elementwise -> Residual Add。

每个 fused kernel 在 manifest 中声明 `includes` 集合。Composition validator 拒绝同一逻辑 operation 被两个节点覆盖，亦拒绝必要节点缺失。Dispatch/combine、AllReduce 和 A/F activation transfer 进入 communication provider，不进入 compute sum。总 `latency_us` 由原子结果求和后一次性转秒，breakdown 保留原值、key、match 和 source。

备选方案是直接存 A/F 聚合时延。它查询简单，但无法复用形状数据、定位误差或检查重复计时，因此只允许未来将聚合实测作为独立、明确的 fused composition 类型，而不覆盖原子语义。

### 7. 通信 service 与资源竞争分离

Transfer provider 由 query 计算或校验 payload bytes，按 direction/device/network 查询 `fixed_latency` 与 `effective_bandwidth`，返回：

```text
service_seconds = fixed_latency_seconds + payload_bytes / bytes_per_second
```

若有同键端到端链路实测，可用版本化实测 service model 替代公式，但必须保留来源。Provider 不接触 SimPy resource。全/半双工、容量和跨 block 竞争继续由 executor 的规范化 resource key 决定，因此 trace 中 `total = queue + service`。

### 8. 接口演进与 trace

保留现有 query protocols，并按 operator key 所需的最小字段扩充 model metadata。`AFServiceEstimate.details` 采用稳定、可序列化结构承载：

```text
dataset_version
match_type
components[]: name, key, latency_us, match_type, source_id, interpolation?
```

公共字段可提升为强类型字段以便校验，details 仅保存类别特有内容。Executor 在验证 duration 后将 provider metadata 原样复制到 `AFTraceEvent`，但只读取 `service_duration` 和已有 `payload_bytes`；通过同输入 trace on/off 对照证明 metadata 不影响调度。

为兼容 stub，可为缺少 dataset/match 的测试 estimate 提供明确的 `test-stub` identity adapter；不得给正式 provider 静默补默认 provenance。

### 9. 两类端到端验证分别回答“机制”和“物理值”

机制 oracle 从小型 source samples 运行 generator，加载 Parquet，执行多 block AF，精确断言 lookup、composition、unit、queue、overlap、makespan 和 trace provenance。它同时覆盖非法 bundle、duplicate、missing、out-of-range 和各插值轴，并保留已有 7/10 ms stub baseline。

精度验证使用隔离 holdout 与实际采集的 kernel/link/系统 trace，分别比较 compute service、transfer service、queue、TTFT、TPOT 和吞吐。Queue 的确定性 oracle 来自事件依赖与资源容量，service 的物理 oracle 必须来自对应 benchmark。没有真实系统 trace 时不能以 E2E 自洽测试替代物理准确性声明。

## Risks / Trade-offs

- [Groq/Ada 数据覆盖不足导致正式查询频繁失败] -> 先由 workload shape manifest 驱动采集与离线补齐，报告 coverage；保持 missing 显式失败而非隐藏误差。
- [线性单维插值不能描述 kernel 边界跳变] -> 记录 worst shapes 和分段误差；边界处补测，未来以版本化策略扩展。
- [Fused kernel 语义不一致造成重复计时] -> 数据 schema 强制 `includes`，composition validator 检查逻辑 operation 唯一覆盖。
- [Parquet 依赖增加安装体积] -> 延迟导入并只在正式 provider 入口预检，AF-disabled 与 stub 路径不加载。
- [公开资料的证据等级被误读为实测] -> source catalog 与报告固定展示等级、设备、方法和限制，不允许生成器升级标签；B 级端到端数据不得直接进入单算子运行时表。
- [Provider metadata 过大导致 trace 膨胀] -> 保存规范化 key 和 source 引用，公共 dataset metadata 只存 identity；保持 trace 可关闭。

## Migration Plan

1. 增加 schema、typed keys、loader 和最小 fixture bundle，不接入默认运行路径。
2. 实现 generator、source catalog、calibration/holdout 与 validation report，生成可复现测试包。
3. 实现 compute lookup/composition 与 transfer provider，扩充 estimate/trace metadata，并以显式配置装配。
4. 运行生成包 E2E、既有 AF stub oracle、AF-disabled 和完整 Python 回归；只有数据覆盖与来源校验通过后才允许选定生产 dataset。
5. 回滚时停用正式 provider 配置并保留数据包；stub 与旧 backend 无需迁移。Schema 不兼容时通过新 dataset/schema version 并行发布，不原地重解释旧包。

## Open Questions

- 首批 Ada/Groq 数据包的具体硬件 SKU、backend/编译器版本和实验 shape manifest 由后续数据采集输入确定；这些值不改变查询契约或实现分解。
- `effective_bandwidth` 是由独立链路 benchmark 直接给出，还是按 message-size bucket 拟合，将根据首批通信数据选择；两者均遵守相同 transfer provider 契约。
