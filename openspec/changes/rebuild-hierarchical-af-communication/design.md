## Context

动机见 proposal.md。现有实现已包含 DAG/service 缓存、host submission/transport/completion、physical/flat_link 模式以及完整 trace 索引与延迟导出。本设计建立在这些行为上，不重复建设缓存。

主要定位：comm/collectives.py 的 ring_traffic_factor、analytical、point_to_point、轮次选择；af/operator_provider.py 的 collective 与 flat 估值；hardware/links.py 的 H 效率曲线；af/resources.py 的 group/lane 资源；af/executor.py 的 host/transport 执行；af/planning.py 与 model_executor.py 的 payload、ownership 和 collective 边界。hardware/placement.py 的旧虚拟 placement 与真实节点映射需要区分。上述定位用于实施调查，具体行号随代码变化。

前置变更 integrate-deepseek-v3-core-runtime 已实现但未同步主规范。后续规范同步顺序见 proposal.md；本次只创建规划文件。

## Goals / Non-Goals

**Goals:**

- 让同一通信计划同时解释数据量、服务估计、资源占用与等待，能够手算验证。
- 固定模型与路由语义，对比 dedicated/mixed 放置造成的节点内外流量、队列与端到端性能差异。
- 通过静态计划复用和受约束的消息粒度，控制仿真事件规模。

**Non-Goals:**

- 首版不做逐包/可任意分块流水、逐物理段存储转发排队、动态多流带宽均分、自动 NIC striping 或 NCCL 算法调优搜索。
- 不精确模拟通信 kernel 的 SM/HBM 竞争，不新增同卡 A/F 共驻，不宣称理想峰值公式等同实测。
- 不重写 MoE routing/expert 选择、KV ownership 或无关计算估值。

## Decisions

### 1. 统一数据口径与解析公式

Collective 对外 payload 为每 rank 的逻辑输入字节 M，另记全局数据、输出、每阶段/每轮实际发送 wire bytes。N 为当前算法子组的参与 rank 数，不能把全局 group_size 套到所有层级。

均匀 ring 的手算基准：

| 操作 | 每 rank 输入 | 每 rank 输出 | 轮次 | 每 rank 总发送 |
| --- | --- | --- | --- | --- |
| AllGather | M | N·M | N−1 | (N−1)·M |
| ReduceScatter | M | M/N | N−1 | (N−1)·M/N |
| AllReduce | M | M | 2(N−1) | 2(N−1)·M/N |

同质单层估值为 rounds·alpha + wire_bytes/B。AllGather 的 (N−1)/N 系数仅在输入是完整聚合 buffer 大小时适用；测量数据需要记录并转换口径。模板已展开轮次时不再乘一遍 rounds。flat 与 physical 共享算法、字节和轮次规则，各自选择资源与路径参数。

输入 bytes 必须为非负整数，dtype 元素/分片边界精确可整除；rank 数来自去重后已校验的实际 participants，query 声明的 group_size 必须一致。空 group 为错误，单 rank 为明确 no-op，不能用 max(2,N) 修补。本文 wire bytes 指算法传输的数据字节及明确计入的元数据，不默认包含链路协议包头、重传或纠错冗余；按路径分段统计时与逻辑发送量分列，不能把同一消息经过三个段当成三份应用输入。

基础链路模型为 alpha + M/B。移除新解析通信路径对 half_bandwidth_bytes 和未校准 collective_efficiency 的使用；H 原本是效率曲线参数，H/B 并无已证实的启动开销含义。旧字段兼容解析并标注新模型忽略/弃用，其他消费者保持原行为直到明确迁移；不能全库删除。oversubscription 仅降低有效带宽，不乘固定 alpha。已建模共享上联容量的瓶颈不再叠加同源超售系数。

实测数据保留 source、版本、证据与 timing_scope，不能对实测总时长机械套解析系数。拆分 host 计费时只接受可证明为 transport_service 的记录；eager/full/unknown 明确拒绝或在离线转换后使用。零数据、单 rank 和无实际网络操作跳过网络资源与网络成本；实际需要的本地 copy/reorder/reduction 独立保留。

transport_service 还不足以证明一条实测记录可以拆成新算法的多阶段服务。记录必须匹配算法版本、participants/拓扑、payload 口径及阶段范围；只有整个 collective 的总时长时，不按字节比例任意分摊到阶段，不把同一总时长用在每个 phase。physical 多阶段查询缺少可用阶段数据时，按已有显式 fallback 策略选择解析模型并标记来源；measured-only 策略报错。不能把被拒绝的测量伪装为解析命中。

选择这一基线便于校验单位和流量守恒；继续使用未经校准 H/效率会掩盖公式错误。未来校准通过版本化数据与明确作用域引入。

### 2. 节点与接口资源

node_size 默认 8，可配置。节点内提供 GPU→本地交换结构→GPU 的全互通可达性；节点间提供 GPU→接入链路→NIC→网络交换结构→NIC→接入链路→GPU。交换结构默认无阻塞，可显式声明共享上联；GPU 到 NIC 不强制经过 NVSwitch。

| 参数 | 初始配置口径 |
| --- | --- |
| A100 SXM NVLink | 每 GPU 聚合 tx=300 GB/s、rx=300 GB/s |
| H100 SXM NVLink | 每 GPU 聚合 tx=450 GB/s、rx=450 GB/s |
| 200 Gb/s NIC | 每端口每方向 25 GB/s |
| 400 Gb/s NIC | 每端口每方向 50 GB/s |
| GPU-to-NIC | 独立配置有效接入带宽、固定延迟和共享总线组 |
| 固定路径延迟 | 各明确物理段的非重叠固定延迟，标记来源；不复用 host submit 值 |

GB/s 使用十进制，内部归一为 bytes/s 和 seconds；双向汇总规格不能再次当成单向。型号预设只适用于指定 SXM 拓扑，不能套用所有同名 GPU 系统。

资源包括 gpu:nvlink_tx/rx、GPU-to-NIC 接入及共享 PCIe group、nic:tx/rx、可选上联。全双工分方向，半双工共用同一资源。一个 GPU 的多条 NVLink 消息共享该 GPU 聚合方向容量；多个 GPU 共用 NIC 时共享其方向容量。多个 NIC 不自动相加，每条消息按固定映射选择 NIC。旧 capacity=N 仅表示 N 条真正独立 lane；共享总带宽不能复制成 N 份峰值。

选择端口/共享组模型以同时避免 P² 独立满速链路和全局单 lane 两个极端。接口之间只有显式共享硬件时才竞争，避免把 NVLink 与独立 NIC 操作无条件串行。

物理标识必须包含 node、device/port 与方向，别名解析到同一 canonical resource；不同节点的 nic0 不相互冲突。路径必须是实际可达的有向路径，反向单独解析；若声明多 rail/网络分区，固定 NIC 映射必须能连通目的端，不能自动穿越不存在的交换连接。默认同节点传输使用本地 GPU 交换路径，不占 NIC。正向带宽必须有限且大于零，延迟有限且非负；资源别名的带宽、方向和容量定义冲突时启动前失败。

### 3. A/F 放置是可复现输入

支持手工 worker→(node_id,gpu_id) 表，以及规则自动分配后手工覆盖。默认便利策略：dedicated 将 A、F 放在独立节点；mixed 指定每节点 A/F 槽位，如 2A+6F。比例为输入，尾节点可留空，不复制或丢弃 worker。

规则生成先预留手工槽位，再按稳定 worker/group 次序填充剩余槽位，最后完整校验。提供计数、TP/EP/DP group 约束、node_size、NIC 数量及 GPU→NIC 映射。一个物理 GPU 首版只承载一个 A 或 F worker；现有 TP/EP rank reuse 不产生额外 GPU。逻辑通信组独立于节点划分，不能由角色名称假定节点内外关系。

自动放置只使用声明的物理库存，不为满足角色比例隐式增加 GPU/NIC/节点；手工覆盖与 dedicated/mixed 槽位策略冲突时明确失败，用户可改用完整手工模式。放置对比同时列出总库存、实际占用、空槽、节点数和 NIC 总数；推荐固定同一库存比较，避免把额外 NIC 或额外节点带来的收益归因于角色摆放。

失败条件包括重复占卡、节点越界、缺失 NIC/路径、非法分片与层级布局、weights+KV+buffers 超出显存预算。复用现有内存核算，记录通信 buffer 的假设和保守预算，不把未知 buffer 默认为零以通过校验。输出物化 placement 表、策略、任何随机种子与配置指纹；不做隐式最优搜索。

比较放置时保持设备总数、模型、请求、routing assignments、expert ownership 语义一致。仅替换物理位置；旧 1A32F 同 NVLink 域是虚拟实验，迁移到 8 卡节点必须显式重新放置并验证显存。

### 4. 算法模板与 physical 路径分层

首版固定 ring 基础模板，physical 的均匀多节点 collective 使用确定的层级模板，不做运行时 ring/tree 搜索。逻辑 rank 顺序与节点内顺序稳定。每节点参与数相同、分片可整除为初始支持域；不支持的布局启动前给出错误和上下文。

这里的 local slot 是当前通信子组内的稳定序号，实际 GPU local_rank 可以不同；通过 placement 映射实际端点，不能按物理卡号相等推断参与关系。

- AllReduce：节点内 ReduceScatter → 对应 shard 的节点间 AllReduce → 节点内 AllGather。
- AllGather：相同 local slot 的跨节点子组先 AllGather 各自输入分片 → 每节点对这些结果做节点内 AllGather；输出必要时独立 layout reorder。若有 K 个节点、每节点 G 个参与 rank、每 rank 输入 M，跨节点阶段每 rank 输入 M、输出 K·M，随后节点内阶段每 rank 输入 K·M、输出 G·K·M。ring 跨节点总发送量为 K·G·(K−1)·M；不得先将节点完整 G·M 数据复制到每个 rank 再重复跨节点发送。
- ReduceScatter：按目标节点切分输入，节点内对每个目标节点分片做 ReduceScatter → 相同 local slot 之间对目标节点分片执行节点间 ReduceScatter，输出保持目标 rank ownership。
- 单节点退化为节点内基础算法；每节点单 rank 退化为节点间基础算法；N=1 无网络任务。

以上模板要用小规模张量 identity/字节 oracle 验证守恒；算法 phase 可由多个相互独立的 subgroup task 组成。每个 ring subgroup phase 首版可以粗粒度持有所需资源，按轮次累计服务，并披露提前释放被省略的保守性。估值必须基于各轮实际资源流量，不把非均匀路径带宽池化：10/90 的 NIC 负载仍按各 NIC 的实际占用计算；模板无法表达的布局明确拒绝。

粗 phase 内部的竞争也必须计入：按模板轮次和稳定消息序，对每条 transmission 使用整路径原子占用规则计算局部无外部竞争时间表，共用接口的消息串行、不相交路径并行；phase 服务为该局部时间表的完成时间，资源请求为所用 canonical resources 的去重并集。此局部时间表可缓存，不为每个内部发送创建全局 SimPy 事件；外部调度仍在整个 phase 期间持有并集资源。不能仅取所有发送的单消息 service 最大值，否则共享 NIC 的流量会少算。round 间依赖显式保留，未真实打包的发送分别计固定成本。

每个 subgroup phase 的 ready 必须包含其实际参与者的输入到齐条件，不能用最早到达 rank 启动；只关联自身前驱 subgroup，不增设跨无关节点/副本的全局 barrier。阶段计算和所有规范同样适用于当前支持的 AF prefill/decode、层内及 embedding/logits 边界，不能仅迁移 Decode 层内路径。

P2P、A→F 和 F→A 使用实际 endpoints 与已有路由，没有通用的“先送同 local_rank GPU”中转规则。只有算法需要 gather/relay/scatter 时才创建相应阶段。AF transfer 已承担 dispatch/return 职责，不能额外创建同数据的 EP 搬运。token/expert assignment、允许的同目标输入复用、shared 输入去重与 weighted combine 语义沿用现有契约；任何本地打包/归约工作若需要计时须明确独立节点。

### 5. 算法阶段独立排队，单次传输整路径预留

定义 ready 为本任务输入依赖和必要提交已完成的时间。start 为全部所需资源原子获得的时间，end=start+S，Q=start−ready。

单次端到端 transmission 的物理段构成一条路径，固定 wire bytes 为 M：

    S_path ≈ Σ alpha_segment + M / min(B_segment)

这里 min 仅用于这一条传输路径，表达整路径流水近似。源 GPU-to-NIC、源 NIC/网络、目标 NIC-to-GPU 是物理段，在本版本不各自排队；一旦任务开始，整路径已占用，途中不再等待。参数必须避免同一 NIC/交换固定成本在相邻段重复计入。

若算法确实需要 local gather → inter-node transfer → local scatter，则三阶段先后执行，各自申请/释放其资源，各自使用服务估计。串行链为：

    T = host_submit 的排队与服务 + Σ(Q_phase + S_phase)
        + host_completion 的排队与服务

不对三个算法阶段统一取瓶颈带宽；也不将每条并行消息的 Q 和 S 全部相加。一般 DAG 的完成时间由实际依赖事件决定。host submit/completion 沿现有逻辑调用边界计费，内部 phase 不自动重复加 host 费用；显式额外调用必须在计划中声明。

host_resource_scope 首版保留现有 per_worker/shared 语义，新增 node_id 不自动变成每节点共用 host 锁。host timing 关闭时不生成额外 host 成本；网络模型不得通过关闭 host timing 改变 payload、阶段或物理资源。输入依赖、host 等待与资源等待分别列出，操作级端到端时间不得再重复叠加子事件的包含区间。

示例：gather S=2、跨节点 S=5、scatter S=3，跨节点资源在 t=6 才可用，其他资源空闲且 host 成本为零。时间线为 gather[0,2]、跨节点 queue[2,6]、transfer[6,11]、scatter[11,14]，完成 t=14。传输[6,11] 内没有额外的中途排队。

选择整路径流水与阶段调度的组合，控制事件规模并保留算法依赖。逐段 store-and-forward 会改变并行与排队语义，本版本不实现；用户不能随意增加 chunk 数来获得更低时间。

### 6. 原子多资源调度

同一 transmission 所需资源全部可用才一次性预留；等待者不持有部分资源。首版资源独占或独立 lane，不做运行中多流带宽分享。physical 使用解析出的端口/总线/上联资源，不能再为同一瓶颈附加旧全局 fabric lane。flat_link 保留规范化 group/lane 队列；额外人为并发上限必须明确声明其含义。

默认确定性仲裁：ready timestamp 加稳定序号排序。存在更早且资源集合相交的等待请求时，后到请求不越过；资源集合不相交的工作可以先运行。有限前序请求且服务有限时不会被后续到达者饿死，代价是存在保守的队首阻塞。只在资源释放、依赖就绪、取消时重新检查受影响的活跃请求；不扫描已完成历史。

同一模拟时刻先处理该时刻的释放/取消和依赖满足，再进行稳定顺序仲裁；稳定序号来自逻辑 operation/phase identity，不能依赖字典遍历、缓存冷热或 wall-clock。只用于统计的合并/trace 导出不得改变调度次序。粗 phase 对重复出现的同一资源只申请一次，其重复流量由内部时间表计费，避免对 capacity=1 的 NIC 自我等待。

旧 AF link 的 network/group/duplex/capacity 字段继续完成兼容解析和冲突校验。flat_link 用它们创建 lane；physical 中它们仅保留逻辑标签，物理方向容量由拓扑配置决定，配置摘要必须明示这种区别。若需要额外全局并发上限，必须通过明确的独立限制声明，不能静默激活旧 lane。

失败/取消原子撤销等待或释放所有已占资源，取消的后继不得继续。容量不足、无路径、非有限时间等在进入等待前失败，避免永远不可满足的请求。每任务资源需求与服务公式使用同一计划，不能估值按一块 NIC 而调度按另一块 NIC。

### 7. 合并和缓存不改变可执行次数

只合并算法真实打包为一次发送的数据：同一 operation 实例、phase、source GPU、target GPU、physical path，且 ready 与 consumer completion 边界兼容。合并 wire bytes 相加；只有真实一次发送才共用一次启动成本。不得跨 block/iteration/独立调用；不得合并节点内与节点间消息；晚就绪的数据不得拖住原本可早启动消息。按节点聚合统计不等于合并执行任务。

P2P/MoE 使用 endpoint/batch 粒度；collective 使用 phase/subgroup 粒度。128 tokens、top8 的 1024 assignments 只有在算法确实每目标 rank 打包一次时，才可能成为至多 32 个目标发送。无法打包时保留原消息；稠密 all-to-all 仍可能 O(P²)，不通过任务数上限静默截断。

缓存 topology/path/algorithm 模板、按需实例化的静态计划和纯 service 估值。key/fingerprint 覆盖模型版本、topology、placement、NIC mapping、算法、payload、dtype、layout、路由 identity 与 provider 数据版本。拓扑等配置单次运行不可变；变更配置建立新 snapshot/resource/cache。禁止缓存 event、resource request、ready/start/end、排队或已执行状态；命中缓存仍重新创建任务并独立排队。无全 shape 空间预扫描。

### 8. 可观测性与验证

Trace 保留完整历史、现有索引和延迟导出，新增 model version、operation/phase/subgroup/message identity、endpoint/path、方向资源、logical/wire bytes、算法轮次、纯服务 provenance。dependency wait、resource queue、host 阶段分别记录；并行区间用事件时间线重建 makespan。

验证分层：先固定旧调度验证算术修正，再切 physical 资源执行，最后比较放置与性能。缓存开关结果一致，每个调用都独立产生活跃事件。指标包括每层/step 的 task 数、SimPy event 数、CPU/wall、峰值内存、trace 开关和导出成本。

用户已确认写入约束的事项：接口分离、支持布局、ready/completion 边界、流量与资源一致、缓存/执行状态隔离。实现后重点测两项：整路径提前占用接收端造成的保守误差；去除 H 与流水近似对小消息的偏差。用小规模显式阶段/分块参考模拟和可得实测对照，参考模拟仅用于验证，不成为可随意调节的生产 chunk 配置。缺少实测时只报告自洽性与敏感性，不声称真实硬件准确。

### 9. DeepSeek V3 验证矩阵

本节为待实现后的测试计划，不表示已执行或验证通过。模型使用仓库 deepseek-v3-671b-mla 定义：61 层（3 Dense、58 MoE）、hidden=7168、256 routed experts、top-8、1 shared expert，沿用 BF16 权重/activation/KV 与既有 synthetic routing。模型及 operator package 版本、seed、assignments 摘要必须写入 manifest。模型结构仿真与真实 checkpoint 执行区分；不改变模型尺寸来绕过显存检查。

#### 基线与负载

- 旧模型基线：现有 data/clusters/deepseek_v3/1a32f.json，仅用于旧数值及模拟开销比较，标记 33 卡虚拟 NVLink 域。
- 新物理紧凑基线：1A32F、A TP=1、F TP=EP=32、node_size=8、A100 SXM 80GB；A 单独节点，F 占四个完整节点。5 节点是库存，33 卡活跃，其余为空槽。显存必须先校验 weights+KV+buffers。
- 合法摆放配对：固定 9 个八卡节点库存，32F 在其中 8 节点各放 4F。dedicated 的 1A 放第九节点；mixed 将同一个 A 放入已有 F 节点空槽。两者 F 位置、GPU/NIC 库存及映射不变，只移动 A；第九节点在 mixed 保持库存但空闲。这是隔离放置变量的受控实验，不代表最优部署。1A32F 的紧凑 5 节点 mixed 若造成 F 参与数不均匀，作为拒绝测试，不能静默更改 TP/EP 或路由以得到性能结果。
- 基础网络实验参数：NVLink 每方向 300 GB/s；每节点 8 个 400 Gb/s NIC，GPU 槽位固定一对一映射；GPU-to-NIC 每方向 25 GB/s（本实验的接入假设，非设备实测）；无阻塞交换。再做接入带宽 50 GB/s 的受控对照，否则 NIC 从 200 升到 400 Gb/s 可能完全被接入链路掩盖。固定延迟先用明确列入 manifest 的假设值，按 0/1/5 微秒做单因素敏感性；不把实验值当成硬件标定。
- 负载分层：DSV3 真实形状的通信计划回放 → 单 Dense/MoE 层与边界子图 → 完整 61 层 → 多请求服务。子图与回放明确标记，不能冒充完整模型结果。
- Decode：预置 context=2048，batch=1/8/32，af_block_size=1/8；上下文敏感性加 4096，保持 token 数、路由和 dtype 可比。完整请求实验单独执行等长、未分块 prefill，输入长度 128/2048、输出 16 tokens；更长输出 64 tokens 只作为选定压力点。内存或 shape 不支持时明确记录，不自动缩模型或换 dtype。
- 先逐项改变变量，再选择高竞争组合，避免 node/NIC/batch/context/placement 的全笛卡尔积。控制实验使用固定 assignments 回放；均衡与热点 routing 是不同负载，每组内比较复用同一 assignments。全模型实际不触发的 collective 操作，从相同 DSV3 tensor 形状建立单独通信 fixture，不为覆盖率给真实 DAG 增加虚假操作。

#### 测试项目与验收

| 编号 | 待验证方案 | DSV3 测试设置 | 观察与验收 |
| --- | --- | --- | --- |
| V1 | payload、轮次与分层算法 | 提取 Dense/MoE 和 logits 的真实 query；另以相同 tensor 形状验证 AG/RS/AR，group=1/2/4/8/32 | logical/input/output/逐资源 bytes 守恒，精确分片；单层同算法 flat/physical 的纯服务一致；层级 AG 无整节点 buffer 重复发送 |
| V2 | 三段跨节点路径与排队 | 回放 DSV3 A→F/F→A 消息，分别制造共享源 NIC、目标 NIC、上联及独立路径 | 共享资源不超容量；独立路径重叠；算法阶段间可排队，单次整路径启动后无中途等待；无重复旧全局 lane |
| V3 | 整路径预留近似 | 同一批消息、同一到达顺序，对比生产整路径预留与测试专用分块/逐段参考；无竞争、共享接收端、双向拥塞各一组 | 报告完成时间、queue、接口利用率及 P50/P95 差异；参考 chunk 数递增检查收敛，未收敛则标明；不把参考模拟当实测，不预设误差方向或通过阈值 |
| V4 | coarse phase 内部竞争和提前释放 | 回放 F TP=32 collective；对比逐轮/逐消息参考；NIC 均匀与热点映射 | 内部共用 NIC 的时间被累计，去重预留不自锁；报告粗 phase 持有资源与细粒度提前释放的差异，以及对后续 AF 消息的影响 |
| V5 | 去 H、固定延迟及流水敏感性 | DSV3 Decode batch=1/8/32、prefill=128/2048；固定消息回放中扫描 alpha 与带宽 | 单链路固定排程符合 alpha+M/B；改 H 不影响新模型；报告小消息 alpha 占比、大小消息误差；端到端结果不机械要求带宽翻倍时吞吐翻倍 |
| V6 | 多 NIC 与接入瓶颈 | 每节点 NIC=1/2/4/8；速率=200/400 Gb/s；接入=25/50 GB/s，单因素扫描；加入 10/90 热点分配 | 空闲 NIC 不给固定消息加速；共享 NIC 正确排队；接入瓶颈下升级 NIC 收益受限；模型时间与资源占用一致 |
| V7 | A/F 摆放 | 使用上述 9 节点相同库存的 dedicated/mixed 配对；先 1A32F，扩展到 4A32F 时另立配对并固定 TP/DP | placement 不改变 experts/ownership；分别报告本地/跨节点流量、NIC queue、step latency 和吞吐；不预设 mixed 一定更快 |
| V8 | 合并边界与 MoE 热点 | 同一个 DSV3 block 同时命中本地/远端 F；为不同 producer 设置不同 ready，均衡/热点路由各一组 | 不合并不同路径/独立调用，不拖延早 ready 消息；真实打包字节相加；top-8 assignments、shared input 和返回 combine 保持守恒 |
| V9 | host 与测量范围 | per_worker/shared、host timing 开关；注入整次 collective 的 transport_service 记录与可用阶段记录 | 内部 phase 不重复 host 开销；整段测量不能任意拆分；table_only 缺少匹配范围明确失败，允许解析时标记来源 |
| V10 | 缓存、trace 与稳定顺序 | 完全相同 DSV3 workload，缓存开/关与 trace 开/关；冷启动、热命中及路由变化分别测 | canonical 事件顺序、bytes 与仿真完成时间一致，浮点比较使用预先声明容差；每次缓存命中仍执行并排队，无全 shape 扫描 |
| V11 | 完整模型与服务负载 | 完整 61 层 prefill→decode；低负载、接近饱和和有限时长过载，均衡/热点选代表点 | 请求完成、ownership、取消恢复正确；分别报告 TTFT、TBT、吞吐、队列增长与最终 drain；decode-only 首 token 延迟不标为含 prefill 的 TTFT |
| V12 | 仿真规模与成本 | 1A32F 基线及合法 4A32F 配对；batch=1/8/32，路由复用高/低，trace 开/关 | 分开报告计划构建、provider/cache、事件执行、导出耗时，task/event 数及峰值内存；相同输入至少 3 次运行给中位数与范围，不截断任务 |
| V13 | 无效配置和失败恢复 | 从 DSV3 配置派生非均匀布局、重复占卡、缺失/不连通 NIC、显存不足；传输等待/运行中取消 | 错误在规定边界显式出现；所有已占资源释放，后续请求仍可执行；不通过静默 fallback、扩库存或改模型修复非法输入 |

服务负载 V11 先用一个固定参考配置测得可复现的吞吐基准 C，再对所有配对方案复用相同到达序列与绝对请求率（初始取 0.3C、0.7C、1.1C；C 的单位换算为同输出长度下 requests/s）。不为每个方案各用自身 C 隐藏负载差异。过载测试有固定到达窗口与 drain，报告未完成数量、测量窗口和队列趋势；若实际未跨过饱和点则补选点并记录，有限实验不宣称稳态。性能配对 seed 相同，至少三个独立 seed 检查路由/到达敏感性。

必须通过的验收是守恒、依赖、资源容量、确定性、失败清理和所有回归。V3/V4/V5 的近似准确性是评估结论：报告绝对/相对偏差、零基准处理、最坏案例及可能改变方案排序的情况；差异显著时回到设计评审，不能以 OpenSpec 校验通过代替模型精度验收。没有目标硬件实测时，最终结论限定为 DSV3 形状下的模型自洽性、相对敏感性和模拟成本。

## Risks / Trade-offs

- [整路径提前锁住接收端、粗 phase 延迟释放] → 测竞争/并行/取消，报告相对细粒度参考的差异及负载条件。
- [峰值和流水基线对小消息乐观] → 覆盖零字节、单 token、小/大消息、多节点、多 NIC；以固定延迟和版本化测量逐步校准。
- [均匀布局支持有限] → 输出真实 wire bytes 和模板版本，明确拒绝不支持布局；不通过重复传输整节点 buffer 简化 AllGather。
- [排队事件增长] → 按需模板、活跃等待索引、限定合并；测稀疏/稠密路由，不以合并破坏依赖。
- [显存与配置迁移影响旧实验] → 物化 placement、显式兼容说明与模型版本；不要求通信旧时间逐位一致。
- [新增模型与旧共享函数影响其他调用者] → 先隔离解析模型版本与测试，再迁移 AF 调用点，保留 AF-disabled 回归。
- [整段实测与新阶段模型不兼容] → 校验测量范围，仅在显式 fallback 允许时使用解析服务，不任意拆分总时长。

## Migration Plan

1. 保存 d24df24 上可复现的公式、trace 与性能基线；只使用明确的输入/输出目录，不改用户已有实验结果。
2. 统一 payload、rounds、oversubscription 和基础公式，在旧调度下验证，单独解释时间变化。
3. 增加节点/NIC/放置配置与不可变计划，补全支持域校验和迁移样例。
4. 引入阶段/原子路径调度、缓存边界与 trace 扩展；physical 移除重复旧 lane 限制。
5. 运行正确性、专项近似验证、dedicated/mixed 控制实验、性能与完整回归。保留 flat_link 作为简化对照；它采用修正公式，并非恢复旧错误数值。
6. 文档记录新模型版本与旧配置迁移。需要复现实验旧结果时使用旧提交/配置快照；出现缺陷可回退代码与匹配配置，不能混用新拓扑与旧缓存。规范同步按前置变更先、本变更后执行，归档需后续单独操作。
