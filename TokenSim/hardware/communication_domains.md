# 节点通信域 FIFO 模型

AF 的 `physical` 模式使用 `node-domain-fifo-v1`。默认每节点 8 张卡，每节点一个 `local` 共享资源，默认有效聚合带宽 300 GB/s；每节点还有独立的 `inter_tx` 和 `inter_rx`，默认各 50 GB/s。这些数值是模拟器假设，尤其 300 GB/s 表示整个节点，不能当作 NVIDIA 每 GPU 的 NVLink 规格或实测值。

在集群 JSON 的 `af` 内声明：

```json
{
  "transfer_topology_mode": "physical",
  "communication_domains": {
    "node_size": 8,
    "node_count": 5,
    "placement_strategy": "dedicated",
    "local_bandwidth_bytes_per_s": 300000000000,
    "inter_tx_bandwidth_bytes_per_s": 50000000000,
    "inter_rx_bandwidth_bytes_per_s": 50000000000,
    "local_latency_us": 1,
    "inter_latency_us": 5
  }
}
```

带宽单位为单向 bytes/s，GB 使用十进制；400 Gb/s 等于 50 GB/s。节点内和节点间参数分别设置。`local` 的所有方向共用一个 FIFO；跨节点传输同时申请源节点 tx 和目标节点 rx，使用两者较小带宽，等待时不先占住其中一个。相同源或目标会竞争，互为反向和资源不相交的传输可并行。不存在一个覆盖整个集群的隐含 50 GB/s 队列，也不分配并行 lane。

`networks` 中原有 link 名继续用于配置兼容和 flat_link；physical 的有效带宽来自 `communication_domains`，跨节点 50 GB/s 不是从 NVLink link 名推导的硬件规格。

`node_count` 是声明库存。自动 `dedicated` 或 `mixed` 放置必须提供它；超出库存明确失败。`mixed_a_slots` 默认 2，表示 mixed 节点内前两个槽位给 A，其余给 F。`manual` 可使用 `device_indices`，键是物理 worker identity，例如 `"f-pool:0": 8`；索引等于 `node_id * node_size + gpu_local_id`。手工覆盖先占槽，自动填剩余槽位。与 worker group 的同名 device_indices 冲突、重复占卡或违反角色策略会失败。

旧 33 卡同域实验必须显式迁移放置。不要修改原实验来冒充同一物理假设；建立新配置并记录版本。逻辑 lane 仍可通过 `flat_link` 使用，但它同样采用修正后的 AF alpha-beta 公式。未迁移的非 AF `LinkClass.transfer_us` 消费者保留原效率曲线。

## 算法与数据量

collective 输入是每 rank 的逻辑输入 M，实际 participants 必须与 group_size 相符。AG 输出 N·M、每 rank ring 发送 (N−1)·M；RS 输出 M/N、发送 (N−1)·M/N；AR 输出 M、发送 2(N−1)·M/N。RS/AR 的分片须包含完整 dtype 元素。保留的 ring_traffic_factor 是完整 buffer 的兼容系数，不能直接用其 AG 系数乘每 rank 输入。

节点级时间按本节点各有向发送之和计算，不因接收再翻倍。四 rank AG 每 rank 输入 100 bytes，总发送 1200 bytes；忽略固定成本时为 1200/B_local。

| 操作 | 首版分层阶段 |
| --- | --- |
| AllReduce | local RS → inter shard AR → local AG |
| AllGather | 对应逻辑 local slot 的 inter AG → local AG |
| ReduceScatter | 按目标节点切输入，local RS → 对应 slot inter RS |
| All-to-All | 源节点内实际重排 → 远端交换；目标二次分发显式启用 |

规则 collective 要求各参与节点的参与数相等、分片可整除，不要求占满节点。P2P 和 singleton 不受满节点约束。现有 AF dispatch/return legs 保持精确 ownership，不重复产生 EP 搬运。

分层 AG 的 slot-major 输出在必要时增加本地 `data_reorder` 计算；非 node-major 输入的 RS 同样先做本地排列转换。排列已一致时跳过重排。这些操作占用已有 compute slot，网络字节为零。显存账本显式预留双缓冲工作区：`2 × af_block_size × max(hidden_size × top_k, vocab_size) × activation_dtype_bytes`，跨层复用，并从 KV 可用容量扣除；它不代表任意数量并发请求的工作区预算。

统计 All-to-All 的 `remote_fraction` 决定远端比例，`local_fraction` 包括同 rank 的无网络部分，`top_k` 通过显式 payload 的上游推导决定总量。`load_imbalance` 在总量和远端比例不变时，把每个源节点的远端流量偏向稳定顺序中的第一个远端节点；它是统计敏感性假设，不是实测路由。无 fraction 时使用当前参与者的均匀目的分布，输出 uniform_fallback。可通过 `comm.traffic.all_to_all_plan` 或 provider 的 all_to_all 查询取得计划。

统计字节表示期望值，可以有小数；逻辑输入仍要求整数 dtype 元素。热点使输出不均匀时，`output_bytes_per_rank` 为 null，`expected_output_bytes` 给出每个参与者的预期接收量，包括无需网络的同 rank 部分。不要把中转流量再算成应用接收量。

## 缓存和测量

既有有界服务缓存复用逐阶段服务，计划缓存额外区分 placement、域资源、参与者顺序、routing/locality 和依赖。每次执行创建新的事件与身份，独立计算 ready/start/end 和排队，不缓存资源可用时间或完成时刻。改变配置须创建新 runtime/provider。Trace 分列阶段字节、资源、依赖等待、排队和服务；多阶段结束事件只关联完成，不再次占用网络。

完整路由和 tensor slice 特征以 SHA-256 内容指纹进入计划键，逐记录生成，避免计划缓存保留大批已结束的 token 路由图。指纹仍区分影响流量、资源或依赖的特征，不借减少缓存体积排除这些输入。缓存条目数和原有静态对象指纹 memo 都有界。

基础解析为 alpha+wire_bytes/B，不使用旧 H 或未校准 collective_efficiency；超售只影响数据项。没有匹配阶段测量时，table_first 明确使用解析，table_only 失败，不拆分整个 collective 总时长。可选算子包 metadata `communication_phase_services` 存放阶段记录，精确匹配 model_version、algorithm、phase_kind、operation、dtype、participants、nodes、logical_input_bytes、wire_bytes、wire_schedule、domain_parameters，并要求 timing_scope=transport_service、有效 latency_us 和 source_id。`wire_schedule` 按轮记录 `(source_node_index, target_node_index, wire_bytes)`，节点索引由当前子组节点排序归一化，从而区分不同竞争关系。`domain_parameters` 为带宽及固定延迟字段的有序键值对，数据版本取包版本。重复加载相同记录会去重，有冲突的匹配记录会失败。旧 collective 表不自动视作阶段表，零网络操作不使用非零测量收费。

## 验证与限制

在仓库根目录运行 `python -B -m unittest TokenSim.validation.test_communication_domains`。基线在 `TokenSim/validation/communication_baseline_before.json`，它记录提交、源码指纹、配置和最小时间线，属于合成对照。

可运行配置位于 `TokenSim/validation/configs/dsv3-{compact,dedicated,mixed}.json`。compact 为 5 节点库存、33 卡活跃；另两份均为 9 节点库存、33 卡活跃，32F 的卡位完全一致，仅将 A 从第九节点移动到第一个 F 节点的空槽。这组受控 mixed 使用 manual 映射，不应与固定角色槽位的自动 mixed 策略混淆。配置仍需通过 `build_model_runtime` 加载目录中的完整 DSV3 ModelSpec。

专项验证命令为 `python -B -m TokenSim.validation.communication_experiments --suite micro`，其他 suite 为 decode、service、hot-service、supplement、profile。decode 的三个重复样本使用独立子进程和系统峰值工作集，profile 单独归因 CPU 开销。结果单独写到 `TokenSim/validation/results`，不覆盖用户实验。service 明确采用单请求准入 FIFO 服务器，完整执行 61 层 prefill→decode，再回放固定到达序列；请求准入等待与网络域排队分列。2048-token 服务样本显式关闭既有 DAG 模板缓存以避免保留一次性大形状，通信服务/计划缓存保持开启，模型和任务不缩减。这些实验的结论适用于声明的准入策略。

域级独占 FIFO 省略硬件实际多流共享；粗 phase 持有资源并集直到完成，可能延迟内部已空闲资源的释放。结果用于系统级趋势和瓶颈分析；需用固定路由和相同库存做敏感性对照，不能由 OpenSpec 校验或手算自洽宣称绝对硬件精度。

验证套件运行后会在 `validation/results` 生成实施验证汇总、FIFO 与粗阶段偏差、公式迁移差异及原始 JSON。本提交保留独立通信快照的 [回归日志](../validation/results/communication_commit_regression.log)；此前包含其他 AFD 工作区改动的实验输出保留在本地，不作为本次提交的独立验证证据。

实施后复查确认的未修复问题及验证范围见 [known_communication_issues.md](../validation/known_communication_issues.md)。这些问题包括 subgroup 全局输入等待、阶段测量的硬件适用范围，以及 flat All-to-All 的参数校验。
