## Context

现有实现已包含 DAG/service 缓存、host submission/transport/completion、physical/flat_link 与完整 trace。本次统一 payload、算法阶段和竞争资源，扩展既有缓存，不重复建设。

实施定位包括 comm/collectives.py、hardware/links.py、hardware/placement.py，以及 af 的配置、planning.py、operator_provider.py、resources.py、executor.py、model_executor.py。前置变更 integrate-deepseek-v3-core-runtime 已实现但尚未同步规范，同步顺序见 proposal.md。本文件描述待实施方案，不表示代码已经修改。

## Goals / Non-Goals

**Goals:**

- 用同一通信计划解释 logical bytes、wire bytes、服务和排队。
- 在固定库存、模型和路由语义下，比较 A/F 放置和 locality 对性能的影响。
- 保留阶段依赖，以静态计划复用和粗粒度任务控制事件规模。

**Non-Goals:**

- 本次不做 GPU endpoint tx/rx、GPU-to-NIC、PCIe、rail、交换机端口或逐物理链路建模。
- 不拆分并行 lane、不动态分配带宽、不做逐包或可调 chunk 流水；不新增生产带宽均分调度模式。
- 不精确模拟通信 kernel 的 SM/HBM 竞争，不做 NCCL 算法搜索或同卡 A/F 共驻。
- 不重写现有 MoE expert 选择、KV ownership 和无关计算估值，不以缺少证据的峰值公式宣称硬件实测精度。

## Decisions

### 1. 数据口径与基础公式

Collective 对外 payload 为每 rank 逻辑输入 M，另记输出、全局逻辑量、每阶段/轮次/通信域的实际发送 wire bytes。N 为当前算法子组参与数，不能把全局 group_size 套到每层。

| 操作 | 每 rank 输入 | 每 rank 输出 | 轮次 | 每 rank 总发送 |
| --- | --- | --- | --- | --- |
| AllGather | M | N·M | N−1 | (N−1)·M |
| ReduceScatter | M | M/N | N−1 | (N−1)·M/N |
| AllReduce | M | M | 2(N−1) | 2(N−1)·M/N |

保留 ring_traffic_factor 作为兼容辅助函数，明确其输入口径；AllGather 的 (N−1)/N 只适用于完整聚合 buffer，不能直接作用于每 rank 输入 M。flat 与 physical 共享算法和流量规则；只有资源容量、流量聚合和固定成本口径也相同时才要求纯服务相同。

节点内 300 GB/s 是整节点资源，必须累加各轮该节点全部实际有向发送字节，每次发送计一次，不因接收再翻倍。例如四 rank AG、M=100，每 rank 发送 300 bytes，节点共 1200 bytes；忽略启动成本时是 1200/B_local，不能以
300/B_local 估整个阶段。轮次依赖和真实发送固定成本按模板计算，显式展开后不再重复乘 rounds；只有实际打包才共用一次启动成本。

bytes 为非负整数，dtype 元素和 shard 边界可整除，group_size 与已验证实际 participants 一致。空、重复、缺失参与者失败；单 rank 不提升为两 rank，没有网络需求时无网络收费。wire bytes 包含算法数据及明确计入的元数据，不默认包含协议包头、重传或纠错。同一消息在源 tx 与目标 rx 各占资源，但仍是一份应用发送量；资源负载统计与应用总量分列。

基础服务为 alpha+M/B，不使用 H/half_bandwidth_bytes 或未校准 collective_efficiency。旧字段兼容解析并注明本模型忽略，其他未迁移消费者维持既有行为。oversubscription 只降低数据项有效带宽，不乘 alpha，也不重复惩罚已覆盖的同一共享瓶颈。

实测保留 source、版本、证据与 timing_scope。host 分离计费只接受 transport_service，拒绝 eager/full/unknown。阶段服务还须匹配算法、participants/拓扑、payload 口径和范围；只有整个 collective 总时长时不任意拆分，按显式 fallback 使用解析并标记来源，或 measured-only 失败。实际必要的本地 copy/reorder/reduction 独立保留，不借零网络成本删掉本地工作。

### 2. 每节点三个通信域资源

node_size 默认 8，可配置。physical 以 worker→GPU→node 放置确定通信所属域，不构造全 GPU pair 满速链路或端口图。direct/switched 只标识拓扑抽象，不推导 NVSwitch 物理连线，也不改变本版资源规则。

| 资源 | 容量与竞争边界 |
| --- | --- |
| node.local | 每节点一个共享资源，默认 300 GB/s，可独立配置；节点内所有有向传输 FIFO 独占 |
| node.inter_tx | 每节点一个发送资源，带宽可配置；所有发往其他节点的任务共享 |
| node.inter_rx | 每节点一个接收资源，带宽可配置；所有来自其他节点的任务共享 |

300 GB/s 是本模拟器节点级有效聚合假设，不能解释为 NVIDIA 每 GPU NVLink 规格或整机实测。节点间可在实验中设每方向 50 GB/s，与节点内参数独立；配置可分别声明 tx/rx，常见对称情况两者相同。GB/s 十进制，内部 bytes/s 和 seconds，400 Gb/s 等于 50 GB/s；双向汇总不能当作单向容量。

三个任务竞争同一个 local 时，A 先独占全部 300 GB/s，完成释放后 B，再 C。没有六条 50 GB/s lane，也没有每任务 100 GB/s 的运行时均分。多链路可以解释硬件容量来源，但本版仅接收有效聚合容量，不要求 link_count/per-link 字段，也不再乘链路数。

跨节点 node0→node1 原子申请 node0.inter_tx 与 node1.inter_rx，数据带宽取二者较小值。node0→node2 与前者竞争源 tx；node2→node1 竞争目标 rx。node1→node0 使用另一组资源，可与正向并行；node2→node3 与前者不相交，也可并行。不存在全体节点共用的隐含 50 GB/s 队列。跨节点任务不隐式占用 local；只有算法显式安排源节点重排或目标分发，才另有 local 阶段。

canonical resource identity 包含 node_id、domain、方向；别名归一化，冲突早失败，不同节点同名资源独立。带宽有限且正，延迟有限且非负，来源和计费范围明确。同次运行配置不可变，改参数建立新 runtime/provider、资源和缓存。

### 3. 放置、支持布局与库存

支持手工 worker→(node_id,gpu_id)、dedicated/mixed 自动规则及手工覆盖。先预留覆盖，再按稳定 worker/group 次序填槽；dedicated 分离 A/F 节点，mixed 使用指定角色槽位，尾节点可空。每 GPU 只承载一个 A 或 F worker；TP/EP rank reuse 不增加设备。占卡、节点槽位、策略冲突、通信域缺失、group/shard 与 weights+KV+buffers 显存预算在执行前校验。通信 buffer 采用明确预算，不以未知等于零绕过校验。

规则层级 collective 要求各参与节点参与 rank 数相同、阶段 shard 可整除。node_size=8 是物理容量，不要求每个 collective 参与全部八卡；两节点各四个参与 rank 合法，单 rank/P2P 不受满节点要求。角色 placement 合法也不保证每个 collective 布局合法，逐计划校验。

放置只使用声明库存，不隐式增加节点。输出物化映射、策略/种子/指纹、总和已用 GPU、空槽、节点数与各域容量。对比保持模型、请求、routing assignments、expert ownership 和硬件资源定义一致，只移动允许的 worker；更活跃的节点可能启用更多独立域容量，须明确报告。旧 33 卡虚拟同域配置迁移时显式重新放置，不静默重解释。

### 4. 分层算法与 locality

基础使用固定 ring；多节点规则 collective 使用确定层级模板，不做运行时算法搜索。local slot 是当前子组内稳定序号，通过 placement 映射 GPU，不要求物理 local_rank 相等。

- AllReduce：节点内 ReduceScatter → 对应 shard 跨节点 AllReduce → 节点内 AllGather。
- AllGather：对应 local slot 跨节点聚合输入 → 节点内聚合结果，必要 layout reorder 独立。K 节点、各 G rank、每 rank 输入 M 时，跨节点输出 K·M，最终输出 G·K·M；ring 跨节点总发送 K·G·(K−1)·M。两节点各四 rank、M=100 时总跨节点发送 800 bytes，最终每 rank 800 bytes，不能先复制整节点 buffer 再重复跨节点发送。
- ReduceScatter：按目标节点切输入，节点内归约分片到 local slot，再在对应 slot 跨节点归约散发，保留目标 ownership。
- 单节点退化为节点内算法，各节点单 rank 退化为跨节点算法；N=1 无网络任务。

分层 All-to-All 默认源节点内重排 → 远端交换。按目标逻辑 local slot 中转，跨节点已送达最终 rank 时不增加目标二次分发；只有显式选择节点级中间接收路由才添加目标 local 阶段。重排按实际搬运量计费，数据已在对应中转 rank 时不重复传，不能把纯布局操作直接当网络数据。计划分列本地目的流量和远端重排流量。

MoE 总 activation payload 可由 num_tokens·top_k·hidden_size·activation_bytes 推导。top_k 只放大总量，local_fraction/remote_fraction 指定本节点/远端比例，有限且在 [0,1]，同时指定则和为 1。本节点比例包含同 rank 无网络部分，不能直接等同节点内 wire bytes。无精确路由时用统计比例；无比例时按当前参与者均匀目的分布并标 uniform_fallback，不以物理 node_size 假定所有卡参与。两节点各四参与 rank 时，同 rank=1/8、本节点=1/2、远端=1/2。

可选 load_imbalance 表达固定总量下的热点分布，不能隐式改变 remote_fraction 或凭空放大 bytes；若想模拟 locality 变化，明确调整 fraction。热点影响域负载/内部排程，未提供精确分布时报告统计假设。输出 exact_routing、statistical_locality、uniform_fallback 及所用分布假设；局部偏多、远端偏多、热点作为显式敏感性场景，不声称无证据的精确 expert 位置。

participants 是总参与 rank 数，fan 是当前层级子参与单元数，wire_bytes 是本阶段实际网络量，三者分开。已有 AF P2P、A→F/F→A 按真实 endpoints 与 routing/ownership 执行，不强加同 local_rank 中转或重复 EP dispatch/combine；精确 AF 路由与通用统计 All-to-All 并存。

粗 subgroup phase 按轮次及稳定发送序建立内部无外部竞争排程，同域串行、不相交域并行，服务取内部完成时间。请求资源为去重并集，重复流量在内部计费，不自我申请同一独占资源。首版整个 phase 持有资源并集，省略内部提前释放，必须评估其保守影响；不为每个内部发送创建全局事件。只有相同可执行计划才可作聚合，不将非均匀流量平均到不存在的全局带宽池。

### 5. 阶段时间与调用边界

    ready = 实际输入依赖和必要提交全部完成的时刻
    start = 全部所需通信域资源原子获得的时刻
    queue = start - ready
    end = start + service

这些是时间值，排队对象是具体节点的 local 或 tx/rx 资源；不维护每张 GPU 的 tx/rx 队列。节点内 transmission 为 alpha_local+M/B_local；跨节点为 alpha_inter+M/min(B_source_tx,B_target_rx)，alpha 按当前阶段范围计一次，不额外展开物理跳数。粗 phase 用第 4 节的实际流量和内部排程估值。

串行算法阶段分别申请、释放和排队，不能把所有阶段统一取最小带宽；并行任务由依赖图重建 makespan，不串行相加所有 queue/service。示例：显式三阶段服务为 2、5、3，中间资源 t=6 空闲，得到 [0,2]、[6,11]、[11,14]，中间 queue=4；该示例不代表默认 All-to-All 必有第三阶段。

host submit/completion 保留现有逻辑调用边界，内部 phase 不重复 host 成本。transport 完成即释放网络，消费者仍等待必要 host completion。host_resource_scope 保留 per_worker/shared，node_id 不引入节点 host 锁；关闭 host timing 不改变网络 payload、阶段与资源。依赖等待、host 等待和资源等待分别记录，不重复叠加包含区间。

### 6. 原子多资源 FIFO 调度

每资源 capacity=1，一次服务独占其配置带宽。按 ready timestamp 与稳定逻辑 operation/phase identity 仲裁，稳定序不能依赖容器遍历、wall-clock、缓存冷热或 trace 开关。同一时刻先处理释放/取消与依赖满足，再统一仲裁。

多资源请求仅全部可用时原子获得，等待不部分持有。更早等待者与后到请求的资源集合相交时，后者不得越过，即使部分资源当前空闲；不相交请求可前进。有限前序且服务有限时后到任务不会导致前者饿死，代价是保守的队首阻塞。只维护活跃等待索引，按释放、就绪、取消检查受影响请求。

每 subgroup 等待自身全部输入，不用最早 rank 启动，也不增加无关 subgroup barrier。失败/取消移除全部等待及占用，后继不得误推进；缺失域、不可满足需求、非有限时间进入队列前失败。

flat_link 继续按 network/group/duplex/capacity 构造逻辑 lane 并校验冲突。physical 的旧 group/lane 仅兼容标签，不创建额外全局 lane；group 不同不能隔离共享域，group 相同也不能串行化独立节点。首版不新增隐含全局并发上限。

### 7. 合并与缓存

只合并真实打包为一次发送的数据：同 operation 实例、phase、source/target endpoint、通信域资源，以及兼容的 ready/consumer completion 边界。wire bytes 求和，只有真实一次发送才共用启动成本。不得跨 block/iteration/独立调用，不合并节点内外消息，不让晚 ready 消息拖延早就绪消费者。节点统计聚合不等于执行合并，统计模式也不能把所有域任务合为一个全局任务。

扩展既有 af/estimate_cache.py 和计划缓存，复用有界 LRU，按需构造。缓存不可变模板、逐阶段服务、依赖、logical/wire bytes 和静态资源需求；完整 collective 总服务仅用于汇总，不能代替逐阶段调度。P2P/MoE 保留既有 endpoint/batch 粒度，collective 使用 phase/subgroup；不能靠静默截断任务节省事件，稠密精确路由仍可能有 O(P²) 发送。

服务键包含操作/算法版本、阶段/子组规模、wire bytes、dtype、有效带宽、固定延迟及模型/数据版本。计划键或配置指纹额外覆盖 topology、placement、通信域映射、参与者及顺序、layout、routing/locality 与依赖。相同 payload 不代表相同资源计划；纯服务参数相同可复用估值，不代表计划可复用。

现有服务键排除的 routing_edges、tensor_slices、执行 node_id 不能直接沿用为计划键排除规则。仅排除请求/迭代等纯调用身份；影响流量、域资源或依赖的路由/布局规范化入键，物理 node_id 映射不得被误删。配置快照单次运行不可变，带宽、延迟、topology、locality、算法或数据版本变化创建新 runtime/provider、资源及缓存。

不得缓存 event、resource request、外部 queue、resource_available_time、绝对 ready/start/end 或已执行状态。内部时间表仅为输入已齐、无外部竞争、阶段起点为零的相对安排。每次命中重新绑定 operation/phase identity，创建独立事件，逐阶段申请/释放并计算 ready/start/end。

缓存验收：local service=3 微秒、inter service=8 微秒，任务 t=0 就绪，inter 资源 t=10 空闲。local [0,3]，inter queue=7、执行 [10,18]，完成 t=18。不能把缓存总服务 11 微秒作为一次占用，也不能缓存 t=18 供下次使用。

### 8. 可观测性与精度边界

保留完整 trace 历史、索引和延迟导出，记录 model version、operation/phase/subgroup/message identity、域资源、logical/wire bytes、轮次、估值及 locality provenance、ready/start/end；分列 dependency wait、queue、service 与 host。各域负载总量不能直接相加作为应用 payload；并行区间从事件依赖重建完成时间。

先固定旧调度验证纯算术，再迁移 physical 域调度，最后做放置和性能对照。缓存开关仿真结果一致；统计 task/event 数、CPU/wall、峰值内存、trace/export 成本。逐物理链路参考不再是本次必做模型；测试专用带宽共享参考和细粒度阶段参考用于评估 FIFO 与粗资源持有的敏感性，不新增生产模式。

默认目标为系统级趋势和瓶颈判断。FIFO 与动态共享可能改变平均完成时刻及后续计算重叠，不能仅因总 bytes/B 相同就断言结果相同。缺少真实测量与路由时报告假设、守恒和敏感性，不声称绝对硬件精度；后续如需精度升级，单独设计 endpoint/物理路径模型。

### 9. DeepSeek V3 验证矩阵

本节为待实现后的测试计划，不表示已执行或验证通过。模型使用仓库 deepseek-v3-671b-mla 定义：61 层（3 Dense、58 MoE）、hidden=7168、256 routed experts、top-8、1 shared expert，沿用 BF16 权重/activation/KV 与既有 synthetic routing。模型及 operator package 版本、seed、assignments 摘要必须写入 manifest。模型结构仿真与真实 checkpoint 执行区分；不改变模型尺寸来绕过显存检查。

#### 基线与负载

- 旧模型基线：现有 data/clusters/deepseek_v3/1a32f.json，仅用于旧数值及模拟开销比较，标记 33 卡虚拟 NVLink 域。
- 新物理紧凑基线：1A32F、A TP=1、F TP=EP=32、node_size=8、A100 SXM 80GB；A 单独节点，F 占四个完整节点。5 节点是库存，33 卡活跃，其余为空槽。显存必须先校验 weights+KV+buffers。
- 合法摆放配对：固定 9 个八卡节点库存，32F 在其中 8 节点各放 4F。dedicated 的 1A 放第九节点；mixed 将同一个 A 放入已有 F 节点空槽。两者 F 位置、GPU/节点库存及通信域定义不变，只移动 A；第九节点在 mixed 保持库存但空闲。这是隔离放置变量的受控实验，不代表最优部署。1A32F 的紧凑 5 节点 mixed 若造成 F 参与数不均匀，作为拒绝测试，不能静默更改 TP/EP 或路由以得到性能结果。
- 基础网络实验参数：每节点 local 聚合 300 GB/s、inter_tx/inter_rx 各 50 GB/s，两者均为可配置的域级假设，不作为每 GPU NVLink 或整机实测。分别扫描 local=150/300/600 GB/s、inter=25/50/100 GB/s；不引入 NIC 数量、接入映射或 lane 数。固定延迟按 manifest 假设值及 0/1/5 微秒单因素扫描。
- 负载分层：DSV3 真实形状的通信计划回放 → 单 Dense/MoE 层与边界子图 → 完整 61 层 → 多请求服务。子图与回放明确标记，不能冒充完整模型结果。
- Decode：预置 context=2048，batch=1/8/32，af_block_size=1/8；上下文敏感性加 4096，保持 token 数、路由和 dtype 可比。完整请求实验单独执行等长、未分块 prefill，输入长度 128/2048、输出 16 tokens；更长输出 64 tokens 只作为选定压力点。内存或 shape 不支持时明确记录，不自动缩模型或换 dtype。
- 先逐项改变变量，再选择高竞争组合，避免 node/bandwidth/batch/context/placement 的全笛卡尔积。控制实验使用固定 assignments 回放；均衡与热点 routing 是不同负载，每组内比较复用同一 assignments。全模型实际不触发的 collective 操作，从相同 DSV3 tensor 形状建立单独通信 fixture，不为覆盖率给真实 DAG 增加虚假操作。

#### 测试项目与验收

| 编号 | 待验证方案 | DSV3 测试设置 | 观察与验收 |
| --- | --- | --- | --- |
| V1 | payload、轮次与分层算法 | 提取 Dense/MoE 和 logits 的真实 query；另以相同 tensor 形状验证 AG/RS/AR，group=1/2/4/8/32 | logical/input/output/逐资源 bytes 守恒，精确分片；同算法流量一致，资源/聚合口径相同时纯服务一致；层级 AG 无整节点 buffer 重复发送 |
| V2 | 通信域排队 | 回放 DSV3 A→F/F→A，制造共享 local、源 tx、目标 rx、反向及不相交域案例 | 同域服务不重叠；源/目标同时原子占用；反向及不相交域可并行；不创建旧全局 lane |
| V3 | FIFO 调度敏感性 | 同一消息和到达序列，对比生产独占 FIFO 与测试专用总带宽共享参考，覆盖等长/异长及错峰就绪 | 报告平均/P50/P95 完成、queue 和后续计算重叠差异；不以总传输工作量相同推断平均完成相同，不新增生产共享模式或精度阈值 |
| V4 | 粗 phase 内部竞争和延迟释放 | 回放 F TP=32 collective，对比逐轮/逐消息域级参考，均匀及热点目的分布 | 按每节点实际有向 bytes 累计服务，去重预留不自锁；报告资源并集持有与提前释放差异及对 AF 消息影响 |
| V5 | 去 H、固定延迟及流水敏感性 | DSV3 Decode batch=1/8/32、prefill=128/2048；固定消息回放中扫描 alpha 与带宽 | 单域固定排程符合 alpha+M/B；改 H 不影响新模型；报告小消息 alpha 占比、大小消息误差；端到端结果不机械要求带宽翻倍时吞吐翻倍 |
| V6 | 节点内外带宽配置 | 固定库存/路由，分别扫描 local=150/300/600、inter=25/50/100 GB/s，加入同源及同目标热点 | 参数可独立修改，无每 rank/链路数重复乘容量；热点总 wire bytes 不变，资源负载与排队一致；无隐含集群总队列 |
| V7 | A/F 摆放 | 使用上述 9 节点同库存 dedicated/mixed 配对；先 1A32F，扩展 4A32F 时另立配对并固定 TP/DP | 不改变 experts/ownership；报告本地/远端 bytes、域 queue、step latency 和吞吐，不预设 mixed 更快 |
| V8 | 合并、locality 与热点 | 同 DSV3 block 命中本地/远端 F，设置不同 producer ready，均匀/偏本地/偏远端/固定 fraction 热点各一组 | 不跨独立调用合并或拖延早 ready；top-8 payload 守恒；fraction 与热点分开解释，默认无目标二次分发，现有 AF 路由不重复 EP 搬运 |
| V9 | host 与测量范围 | per_worker/shared、host timing 开关；注入整次 collective 的 transport_service 记录与可用阶段记录 | 内部 phase 不重复 host 开销；整段测量不能任意拆分；table_only 缺少匹配范围明确失败，允许解析时标记来源 |
| V10 | 缓存、trace 与稳定顺序 | 完全相同 DSV3 workload，缓存开/关与 trace 开/关；冷启动、热命中及路由变化分别测 | canonical 事件顺序、bytes 与仿真完成时间一致，浮点比较使用预先声明容差；每次缓存命中仍执行并排队，无全 shape 扫描 |
| V11 | 完整模型与服务负载 | 完整 61 层 prefill→decode；低负载、接近饱和和有限时长过载，均衡/热点选代表点 | 请求完成、ownership、取消恢复正确；分别报告 TTFT、TBT、吞吐、队列增长与最终 drain；decode-only 首 token 延迟不标为含 prefill 的 TTFT |
| V12 | 仿真规模与成本 | 1A32F 基线及合法 4A32F 配对；batch=1/8/32，路由复用高/低，trace 开/关 | 分开报告计划构建、provider/cache、事件执行、导出耗时，task/event 数及峰值内存；相同输入至少 3 次运行给中位数与范围，不截断任务 |
| V13 | 无效配置和恢复 | 派生非均匀 collective、重复占卡、缺失域、非法容量/别名、显存不足及等待/运行中取消 | 执行前或规定边界显式失败；全部已占资源释放，后续继续；部分节点均匀参与与合法 singleton/P2P 不误拒绝 |

服务负载 V11 先用一个固定参考配置测得可复现的吞吐基准 C，再对所有配对方案复用相同到达序列与绝对请求率（初始取 0.3C、0.7C、1.1C；C 的单位换算为同输出长度下 requests/s）。不为每个方案各用自身 C 隐藏负载差异。过载测试有固定到达窗口与 drain，报告未完成数量、测量窗口和队列趋势；若实际未跨过饱和点则补选点并记录，有限实验不宣称稳态。性能配对 seed 相同，至少三个独立 seed 检查路由/到达敏感性。

必须通过的验收是守恒、依赖、资源容量、确定性、失败清理和所有回归。V3/V4/V5 的近似准确性是评估结论：报告绝对/相对偏差、零基准处理、最坏案例及可能改变方案排序的情况；差异显著时回到设计评审，不能以 OpenSpec 校验通过代替模型精度验收。没有目标硬件实测时，最终结论限定为 DSV3 形状下的模型自洽性、相对敏感性和模拟成本。

## Risks / Trade-offs

- [节点级 FIFO 省略真实硬件并行] → 报告聚合假设，评估等长/异长任务及后续计算重叠的敏感性，不承诺绝对硬件精度。
- [粗 phase 持有资源并集导致延迟释放] → 对照细粒度域参考，报告最坏案例和放置排序变化。
- [固定延迟及去 H 对小消息的偏差] → 覆盖零字节、单 token、大小消息及版本化测量。
- [统计 locality 与真实 expert 分布不同] → 标记 provenance，分别扫描 fraction 与热点，保留已有精确 routing。
- [规则布局有限] → 明确报错，允许合法部分节点参与，不隐式补齐或重复发送完整 buffer。
- [事件增长与缓存键不完整] → 按需模板、活跃等待索引、有界缓存与键敏感性测试，不截断任务。
- [配置迁移、显存及共享函数影响旧实验] → 显式版本/placement，保留 AF-disabled 和未迁移调用者回归，不覆盖用户结果。
- [整段实测不适配阶段] → 严格匹配范围和显式 fallback，不任意拆总时长。

## Migration Plan

1. 保存 d24df24 与当前配置的公式、trace 和性能基线，使用明确输入/输出目录，不改用户实验。
2. 在旧调度下统一 payload、rounds、oversubscription 和公式，单独归因数值变化。
3. 引入节点 local/tx/rx 与不可变 placement/容量配置，补全规则 collective、库存、显存校验和迁移样例。
4. 接入逐阶段域级原子 FIFO、缓存与 trace；physical 去掉重复 lane，flat_link 保留逻辑资源。
5. 执行正确性、FIFO/粗 phase 敏感性、同库存放置、性能和完整回归，保留修正公式后的 flat_link 对照。
6. 记录模型版本和限制；需复现旧数值时用旧提交/配置。回退代码须匹配配置，不混用旧缓存。规范先同步前置变更再同步本变更，归档另行处理。
