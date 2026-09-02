## Context

参见 `proposal.md` 的动机。当前 `LLMWorker.run()` 将 scheduler 选出的 batch 交给完整 iteration latency backend，再执行一次 `env.timeout(latency)` 和一次 request 推进；`ClusterConfig` 已提供 `phase_to_af_link`、worker group capabilities 与规范化通信 resource keys，但 AF 运行入口仍固定拒绝。现有 SimPy 环境、scheduler、Request/KV Cache 语义和 Prefill/Decode worker 迁移机制必须保留。

本设计以总任务书为优先：AF 执行能力覆盖 Prefill 和 Decode，`phases` 只决定一次实验启用哪些阶段。Phase 1 文档中的 Decode-only 主流程作为受支持配置保留，不作为实现上限。内部时间统一沿用 TokenSim 的秒；规格中的毫秒 oracle 在测试输入处换算为秒。

## Goals / Non-Goals

**Goals:**

- 把 scheduler 已选 batch 到 iteration barrier 之间的完整 iteration timeout 展开为逐 block、逐层 AF 事件。
- 以同一份物理 worker 与链路资源注册表表达跨 block、跨 link 和并发 iteration 的竞争。
- 将 provider 给出的 service time 与 SimPy 资源产生的 queue duration 严格分离。
- 让真实配置入口可注入确定性 providers，执行到 request result 和结构化 trace，以事件时间线作为机制正确性的最终 oracle。
- 保持 AF 关闭时的代码路径、时延 backend、Request.step、KV Cache 和 KV transfer 行为不变。
- 为后续 operator-data provider 保留稳定查询/结果边界，不在本 change 中实现性能数据模型。

**Non-Goals:**

- 不加载 `*_perf.parquet`、`*_analysis.parquet`、shape manifest 或 `generation_meta.yaml`。
- 不实现实测 kernel 匹配、插值、公式校准、AIConfigurator/InferSim 运行依赖或物理精度报告。
- 不实现同一阶段多 link 选路、work stealing、多 NIC、随机拥塞或芯片内部事件。
- 不拆分单个 Prefill request，不改变 speculative decoding、模型并行或 KV block 粒度。
- 不把 F 逻辑设备内部 TP/EP 细分为独立的 AF 外部链路事件；后续 compute provider 负责将其包含在 F service estimate 中。

## Decisions

### Decision 1: 在 worker 的 STEP 边界接入 AF，executor 不推进 Request

Engine 根据 `ClusterConfig.phase_to_af_link` 建立阶段路由。worker 从 scheduler 获得 homogeneous batch 后确定当前阶段：若 AF 关闭或该阶段未选中，完全沿用原 `dynamic_batch -> timeout -> step`；若阶段已选中，则创建 AF iteration，等待 executor 返回，随后仍由既有 worker 入口对 batch 调用一次 `step()` 和 scheduler update。

Executor 只负责从 iteration 开始到所有 blocks 完成的模拟事件和耗时，不调用 `Request.step()`，也不直接操作 BlockManager。这样 block/layer 事件不能意外生成多个 token，现有请求完成统计仍只有一个入口。

Prefill 与 Decode 使用同一个 executor 状态机，但 builder 分别计算 token 数。所选阶段的 A group 由对应 link 的 `attention_group` 决定；未选阶段继续使用完整阶段 capability。若 Prefill 和 Decode 使用不同 A groups，阶段边界继续走现有 worker 派发和 KV transfer 机制，A-to-F/F-to-A 仅表示单层 AF 外部传输，不能代替阶段间 KV 迁移。

备选方案是在 executor 内复制 scheduler、Request.step 和 KV 逻辑。该方案会产生第二份请求生命周期真相，难以证明每轮只推进一次，因此不采用。

### Decision 2: AFIteration 持有 immutable blocks，block process 串联各层

新增独立 `TokenSim/af/` 包，不复用 KV `TokenSim/block/` 或 `RequestStatus`：

```python
@dataclass(frozen=True)
class AFBlock:
    iteration_id: int
    block_id: int
    phase: Literal["prefill", "decode"]
    request_ids: tuple[int, ...]
    request_count: int
    token_count: int
    context_lengths: tuple[int, ...]
    source_a_worker_id: int
    link_name: str

@dataclass
class AFIteration:
    iteration_id: int
    phase: str
    blocks: tuple[AFBlock, ...]
    start_at: float | None = None
    end_at: float | None = None
    block_completed_at: dict[int, float] = field(default_factory=dict)
```

Builder 按 scheduler 顺序用 `af_block_size` 连续切片，并生成 iteration 内稳定的 block IDs。Request 对象仅在 builder/入口处用于提取 ID、token 和 context；并发 process 使用 immutable block snapshot，避免 stage 间修改 batch。

`run_iteration()` 为每个 block 启动一个 SimPy process，再 `yield AllOf(processes)` 形成 barrier。每个 block process 按模型层数循环四个 stage；同一 block 天然串行，不同 blocks 由 SimPy 并发推进。iteration 的 `end_at` 取最后 process 完成时间，而不是各 block duration 求和。

备选方案是为每种 stage 建全局生产者/消费者队列。它适合更复杂的动态路由，但第一版会引入额外队列终止、关联和背压状态；每 block process 已能准确表达依赖与资源等待，因此先采用后者。

### Decision 3: Engine 级资源注册表是唯一物理竞争真相

`AFResourceRegistry` 在 Engine 初始化时由规范化 cluster 配置构建一次，并被所有 worker/iterations 共享：

- 每个物理 worker ID 对应一个容量为 1 的 compute slot。
- 每个 attention group 保存可作为 A 所有者的 worker IDs；A stage 只申请 block 的 `source_a_worker_id`。
- 每个 FFN group 保存可用 worker IDs；F stage 从该 group 取得一个 worker token，结束后归还。
- 每个 `AFCommunicationResourceConfig.resource_key` 只创建一个 `simpy.Resource(capacity=...)`。

F pool 使用显式 worker tokens 而不是只有计数的匿名 semaphore，使 trace 能记录具体 F worker，并为后续每设备查询保留身份。若某物理 worker 同时承担完整阶段与 AF capability，其 compute slot MUST 被两条路径共享；完整阶段 acquire/release 的零竞争开销不改变 AF 关闭时的模拟时间，却能防止部分阶段 AF 配置对同一设备重复计时。

全双工两个方向键生成两个资源；半双工的两个方向解析到同一个 shared key。资源容量只表示并发服务槽位，不缩放 provider 返回的单次 service time。

备选方案是在每次 iteration 内创建资源。这会让并发 iterations 和共享 links 彼此看不到，无法模拟真实竞争，因此不采用。

### Decision 4: 一个通用 stage runner 统一产生 queue/service trace

每个计算或通信 stage 使用同一资源生命周期：

```text
ready_at = env.now
request resource/token
yield request
start_at = env.now
estimate = provider.estimate(query)
validate service_duration
yield env.timeout(service_duration)
end_at = env.now
release resource/token
record trace event
```

其中 `queue_duration = start_at - ready_at`，`service_duration = end_at - start_at`。Provider 查询可以在获得资源前构造，但只有在 provider/runtime 依赖预检通过后才启动 iteration；shape 级 unsupported 或非法返回值带 iteration/block/layer/stage 上下文失败。使用 `try/finally` 归还资源，避免异常后泄漏 worker token或链路容量。

同一 block 下一 stage 的 `ready_at` 是前一 stage 的 `end_at`。下一层 A stage 只在 F-to-A 结束后 ready。Executor 不人为增加 block 间依赖；所有额外等待都必须能归因于一个资源队列或 iteration barrier。

### Decision 5: provider 契约面向单 block、单层 stage，并为后续数据模型保留元数据

本 change 定义 executor 所需的最小稳定接口：

```python
@dataclass(frozen=True)
class AFComputeQuery:
    iteration_id: int
    block_id: int
    layer_id: int
    phase: Literal["prefill", "decode"]
    stage: Literal["a", "f"]
    token_count: int
    context_lengths: tuple[int, ...]
    model_id: str
    dtype: str
    device_type: str
    worker_id: int

@dataclass(frozen=True)
class AFTransferQuery:
    iteration_id: int
    block_id: int
    layer_id: int
    phase: Literal["prefill", "decode"]
    direction: Literal["a_to_f", "f_to_a"]
    token_count: int
    hidden_size: int
    dtype: str
    source_device: str
    target_device: str
    network: str

@dataclass(frozen=True)
class AFServiceEstimate:
    service_duration: float
    source_id: str
    payload_bytes: int | None = None
    details: Mapping[str, object] = field(default_factory=dict)
```

Compute 和 transfer providers 分别实现 `estimate(query) -> AFServiceEstimate`。Executor 只消费有限且非负的 `service_duration`；`source_id`、payload 和 details 原样进入 trace，不参与调度。测试 stub 根据 stage/direction 返回固定值，且 `source_id` 明确为测试来源，不能用于性能结论。

后续 operator-data change 可在 `details` 中提供 `match_type`、components、composition 和数据键，并实现精确匹配/受控插值，无需修改状态机。运行时依赖预检检查 provider 存在、模型层数/hidden size/dtype 可解析；缺失时在启动 block processes 前失败。单个 shape 的 unsupported 仍由 provider 查询报告。

备选方案是直接调用现有完整 iteration backend 后除以层数/block 数。该值既无法拆分 A/F，也会掩盖 shape 与资源竞争，且违反已定时延契约，因此明确禁止。

### Decision 6: trace 是 append-only 事实记录，不参与调度

`AFTraceRecorder` 接收 immutable event records。每条事件记录规格要求的时间、资源、block/layer、request/token 数；transfer 增加 direction 和 payload，compute 增加实际 worker ID与 provider 元数据。iteration summary 在 barrier 后由 records 和 block completion 生成：

- `makespan = iteration.end_at - iteration.start_at`
- `barrier_wait(block) = iteration.end_at - block.complete_at`
- resource utilization 的 numerator 只合并/累计 service intervals，不包含 queue 或 barrier wait

Trace 关闭时使用 no-op recorder，executor 的 resource requests、timeouts 和选择顺序完全相同。Trace 不回调 scheduler，也不修改 provider 查询结果。

### Decision 7: 错误必须结束 iteration，不能留下半活跃进程

Engine 构造 executor 前先完成配置校验和 provider/model metadata 预检。运行期任一 block process 抛出 unsupported 或非法 duration 时，`run_iteration()` 捕获错误、取消仍活跃的 sibling processes，并依赖 stage `finally` 释放资源，然后抛出带 iteration/block/layer 上下文的统一错误。入口不得在错误后调用 request step 或 scheduler update。

若 SimPy event queue 耗尽但仍有未完成 AF blocks，沿用项目现有未完成请求保护并附加未完成 block/resource 状态。这是死锁诊断，不是正常完成路径。

### Decision 8: 端到端事件 oracle 优先于 isolated smoke tests

测试从真实 JSON cluster config 构造 Engine/Executor，注入 deterministic providers，提交真实 Request batch，并运行到最终 result/trace。核心测试直接检查时间区间和依赖，不只检查函数被调用：

- 单 block 单层与多层链式依赖。
- 双 block `2/1/3/1 ms` oracle：7 ms 单 block、14 ms 串行上界、10 ms 正确流水。
- A 慢/F 快、A 快/F 慢，确认瓶颈资源 queue 增长。
- full-duplex 方向重叠、half-duplex 容量 1 串行、多个 links 共享 resource group。
- F pool 从 1 增至多个 workers 后并发不超过容量且 makespan 非增。
- Prefill-only、Decode-only、两阶段、trace on/off 和 AF disabled baseline。
- barrier 前不 step、barrier 后每 request 恰好一次、KV block accounting 不因 AF block 改变。
- provider unsupported/负数/NaN/Inf 失败且资源全部释放。

测试报告明确区分机制正确性与物理精度：stub oracle 只能证明调度、竞争和记账正确。实测 kernel、通信 benchmark、公开端到端基准与真实系统 trace 对齐留给后续 provider/calibration changes。

## Risks / Trade-offs

- [Risk] 为复合 capability worker 引入共享 compute slot 会触及既有完整阶段 timeout。 → Mitigation：AF 关闭时资源立即获得且不增加 `env.timeout`，用完整回归逐项比较 request 时间、吞吐、KV transfer、并行和 MoE baseline。
- [Risk] SimPy `FilterStore` 或资源请求顺序可能使同时间事件的选择不稳定。 → Mitigation：按 worker ID 初始化 token、按 scheduler/block ID 创建 process，并固定 FIFO/tie-break 规则；测试断言完整时间线。
- [Risk] sibling process 失败后资源泄漏导致后续测试挂起。 → Mitigation：所有 acquire 都用 `try/finally`，iteration 统一取消并等待清理，增加失败后再次提交的端到端测试。
- [Risk] 只有 stub provider 时用户误把 makespan 当成硬件性能。 → Mitigation：trace 强制携带 `source_id`，示例/测试使用 `test-stub`，缺少正式 provider 时 CLI 明确失败，不提供静默默认值。
- [Risk] Prefill 和 Decode 使用不同 A groups 时阶段迁移与现有 KV connector 交叉。 → Mitigation：AF executor 边界只覆盖单阶段 iteration；阶段间所有权迁移仍由现有 Engine/KV connector 完成，并增加两阶段端到端测试确认没有把 KV 迁移计入 AF link。
- [Trade-off] 每 block process 比集中队列更易验证，但对未来动态多 link 路由扩展性较弱。 → 第一版配置已规定每阶段恰好一条 link，当前选择与规格一致；多 link 另开 change。

## Migration Plan

1. 先加入 `TokenSim/af/` 数据对象、provider 协议、trace 与资源注册表，不接入 worker 主路径。
2. 用 deterministic providers 完成 executor 的独立事件 oracle，验证资源释放和失败路径。
3. 在 Engine 中按 capability 分离 request-owning A workers 与 F-only compute tokens，并共享物理 compute slots。
4. 将所选 Prefill/Decode 阶段接入 executor，保留未选阶段和 AF disabled 的原路径。
5. 移除 benchmark 的固定“executor 未实现”拒绝，改为 provider/runtime 依赖预检；没有正式 provider 时仍明确失败。
6. 运行 AF 端到端矩阵和完整现有回归。若发生回归，可关闭 AF 回到旧路径；新增模块与入口分支可独立回滚，不需要迁移用户数据。
