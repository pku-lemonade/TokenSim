## ADDED Requirements

### Requirement: Native message submission can precede peer data readiness

显式原生调用顺序模式 SHALL 将API返回与数据完成分开。接收提交 MUST 由接收方实际调用依赖放行，MUST NOT 额外等待对端计算；实际传输 MUST 等待双方提交、发送数据与fabric就绪。A控制payload API返回后可以提交本地计算，设备执行仍保留NCCL流完成依赖。F控制payload接收 MUST 等待长度CPU可用。

#### Scenario: Sender proceeds while receiver reads length
- **WHEN** F长度回读尚未完成，而A已提交长度消息
- **THEN** A可提交payload及其后CPU调用，payload传输等待F payload接收，A设备计算不提前越过控制流完成

#### Scenario: Return receive is preposted
- **WHEN** A完成本层data发送提交，F尚未完成本层MoE
- **THEN** A可完成返回接收提交；等待F数据期间不占用fabric，通信观察窗口保留该等待

#### Scenario: Native control fails before enqueue
- **WHEN** 长度或payload服务估计在提交前失败
- **THEN** 本轮明确失败并清理所有预贴进程和host资源，后续iteration可以继续

### Requirement: Native CPU groups preserve exclusive coverage

原生计算host profile SHALL 允许一个F MoE调用组覆盖router、grouped experts与combine，每层只收费一次；MUST 拒绝同时设置其内部router/expert提交费用。GPU工作量及设备服务 MUST 保持独立。runner bookkeeping MUST 排除嵌套token readback，与frontend输出整理分开计费并保留完成依赖。

#### Scenario: One F invocation includes internal routing
- **WHEN** 加载compute_f_moe_block_us并执行两层MoE decode
- **THEN** 每步收费两次，不按expert数或combine节点重复收费，与独立router/expert参数同时设置时拒绝执行

#### Scenario: Runner bookkeeping excludes readback
- **WHEN** 同时启用sampled_token_readback、runner bookkeeping和frontend输出整理
- **THEN** 三段按各自覆盖每步收费一次，输出完成依赖CPU token可用，bookkeeping常数不含嵌套回读

### Requirement: Sampled tokens become CPU visible before step completion

显式启用原生采样回读操作时，系统 MUST 在token_selection和必要stream前序完成后执行sampled_token_readback，CPU token可用及必要输出整理完成后才发布该decode step完成，并门控依赖该token的下一步。完整操作常数 SHALL 包括声明的自身copy/runtime/同步/转换；前序GPU/NCCL依赖等待 MUST 动态产生，不进入常数。该操作 MUST 不与通信完成、GPU采样或输出整理重复计费，不独占fabric；取消/失败 MUST 释放调用资源。旧配置未启用时 MUST 保持历史时间线并标记完成操作未建模。

#### Scenario: Delayed token readiness changes waiting only
- **WHEN** CPU在0µs请求结果，GPU token在100µs可用，调用资源空闲，回读完整服务11µs，其他完成成本为0
- **THEN** CPU结果111µs可用，step不得提前完成；token改为120µs可用后，结果131µs可用，回读常数仍为11µs

#### Scenario: Output processing is charged once
- **WHEN** readback已经覆盖token.tolist且另有每步输出整理条目
- **THEN** 整理条目不再次覆盖tolist；同覆盖配置执行前拒绝，每步合法整理仅计一次

### Requirement: Measured submissions map to native call groups

启用补测profile时，系统 MUST 将计算提交绑定至经审核的A/F原生调用组，将通信提交绑定至真实backend/方向，将引擎调度/输出整理绑定至每步边界。计费 MUST 保留CPU与GPU可重叠及真实数据依赖，不将完整前向wall或阻塞等待替代GPU纯计算服务。一个原生grouped/fused调用 MUST 只有一份同覆盖提交成本，旧通用DAG节点提交常数不得同时叠加。等待/资源竞争 MUST 不进入估值缓存。

#### Scenario: One native invocation spans multiple logical nodes
- **WHEN** 一个grouped专家调用对应多个专家以及独立的数学combine节点
- **THEN** 其固定提交费用只收一次，已融合的combine不另收提交费，数据及加权完成依赖保持不变

#### Scenario: Independent submission overlaps device work
- **WHEN** 第一条CPU提交3µs后设备执行10µs，第二条无数据依赖提交耗时2µs且CPU资源空闲
- **THEN** 第二条提交可在3到5µs运行，不必等待第一条设备在13µs结束；有真实输入依赖的设备消费仍等待输入就绪

### Requirement: Eager native control gates data reception

启用native_control_enabled时，系统 MUST 每个未分块eager iteration发送int64长度8 bytes与compact JSON内容两条控制消息，按DP=1/stage0真实格式确定内容字节。控制和activation MUST 分别标记torch_distributed_nccl与pynccl后端，并隔离估值缓存键。长度回读 MUST 先于内容接收；内容回读/解析 MUST 先于F activation接收。A计算 MAY 与F内容解析重叠。控制消息 MUST 与data竞争相同物理fabric，但各自服务/等待独立报告。首版 MUST 拒绝多block和多个in-flight事务。

#### Scenario: Control wire matches the installed serializer
- **WHEN** batch为1/4/8且所有生命周期标志false
- **THEN** 每步长度消息8 bytes、内容179 bytes，与原生序列化函数输出逐字节一致
- **AND** 两层activation仍是四条独立data消息，不把控制字节加进hidden

#### Scenario: F parses control while A computes
- **WHEN** 内容已交付而F仍在执行内容回读/JSON解析
- **THEN** A可开始计算，A-to-F activation传输等待F控制完成，等待不增加transport service

### Requirement: Timeline accounting preserves overlap and unknown gaps

系统 SHALL 按当前iteration索引导出service总量、busy区间并集和时间戳支持的一条关键链。关键链区间和 MUST 重建makespan；并行分支 MUST 不重复相加。未解释的前置间隙 MUST 标为unexplained，外部资源等待单列queue；摘要 MUST 不扫描完整历史trace。

#### Scenario: Parallel branches join
- **WHEN** 两个并行分支分别耗时10和20，后续共同节点耗时5
- **THEN** service总量35而makespan和关键链均25，不能把总量报告成墙钟

#### Scenario: Trace is absent
- **WHEN** 已知iteration起止时间但没有事件
- **THEN** 保留makespan并将解释缺口标为unexplained，不输出零耗时结论

### Requirement: Fixed host preparation preserves causal overlap

系统 SHALL 支持未校准的固定A/F每步准备、额外compute节点调度和接收提交服务。每步准备 MUST 每个参与worker/iteration恰好一次；双方独立准备 MAY 重叠，依赖方 MUST 等待对应准备和提交完成后才进入纯设备或传输服务。额外compute调度 MUST 不覆盖已有kernel启动费用。等待 MUST 由运行态依赖/资源产生，不进入纯service缓存。取消 MUST 清理准备事件及host资源，关闭trace MUST 不改变模拟时间。

#### Scenario: Preparation overlaps attention
- **WHEN** A计算100µs，F从step入口准备200µs，传输5µs，其他成本为零且链路空闲
- **THEN** 传输在200µs开始、205µs结束，peer等待100µs，fabric排队0，纯传输保持5µs

#### Scenario: Multiple blocks share step preparation
- **WHEN** 同一个iteration被拆成多个block并执行多个layer
- **THEN** 每个worker的准备服务只收一次，每次新iteration重新执行

#### Scenario: Cancellation releases preparation resources
- **WHEN** iteration在host准备或接收提交期间取消
- **THEN** 无残留host资源占用，后续iteration可正常完成

### Requirement: Native AFD executes F-side combine before hidden return

在 `vllm_afd` 协议下，仅decode的每层 MUST 执行 A Attention→A-to-F hidden→F Router/Top-K→F grouped expert 与 weighted combine→F-to-A hidden→A residual/layout；计算融合允许合并服务事件，但 trace MUST 保留数学 combine 的完成依赖和唯一计费归属。T 个 tokens 的输入和输出 activation 各 MUST 为 T*H*dtype_bytes，不能因 top_k 放大返回量；metadata 按实测协议独立统计。请求与 KV ownership MUST 保留在 A。每个 token 恰好 combine 一次、residual 一次，下一层 MUST 等待返回及 residual 完成。历史 A 路由、partial-return、A combine 的要求仅约束旧模式；原生模式由本要求定义。

#### Scenario: Top two changes computation but not hidden return size
- **WHEN** 单层8个 tokens、hidden=256、BF16、top_k=2
- **THEN** A-to-F 与 F-to-A 的 data 各为4096 bytes，路由包含16条 assignments；metadata 另列
- **AND** F combine 在返回之前完成，A 不再执行第二次 weighted combine

#### Scenario: Fused expert output already includes combine
- **WHEN** 原生算子测量已包含加权聚合
- **THEN** 模拟只计一次该 service，保留输出已合并的语义，不添加独立 combine 成本

#### Scenario: Protocol mode changes a cached plan
- **WHEN** 同一 shape 从 partial_return 切换为 vllm_afd
- **THEN** DAG、返回 bytes、combine 位置、调用粒度、trace 校验与估值键同步更新

### Requirement: Native timing derives waiting from invocation state

原生对照 MUST 记录 host submission、双端就绪、实际服务开始、transport service 与 completion 边界，并导出通信总等待。peer/fabric原因细分 MAY 作为诊断，但 MUST NOT 成为执行或校准前置条件。消息 fabric-ready MUST 不早于双方对应调用已提交且必要输入及双方stream前置依赖已就绪；peer/dependency wait MUST NOT 作为额外 service 再次推进时间。transport 完成即释放其资源，host completion 仅竞争所属 host 资源；同步等待 MUST 通过依赖及明确的调用阻塞语义表达，不以固定同步常数替代。缓存命中 MUST 创建独立调用状态并重新竞争资源。总 makespan MUST 由完成关键路径计算，不累加重叠端点或所有事件时长。

#### Scenario: Same service produces different queue times
- **WHEN** 两条消息共享容量为1的 fabric，ready 均为0，纯服务均为5微秒且第二条命中缓存
- **THEN** 两条服务区间分别为[0,5]和[5,10]，队列等待分别为0和5微秒
- **AND** 第二条不复用第一条的 start/end/queue

#### Scenario: Delayed receiver is a dependency
- **WHEN** 源端在0就绪、目标端在10微秒就绪，fabric 空闲且纯服务5微秒
- **THEN** transport 为[10,15]，peer wait 为10微秒，fabric queue 为0，service 为5微秒

#### Scenario: Completion work releases the fabric
- **WHEN** transport 在15微秒完成而 host completion 需要2微秒
- **THEN** 下游数据消费在17微秒后发生，fabric 从15微秒可服务另一条就绪消息

### Requirement: Communication reports unified waiting without duplicate intervals

系统 MUST 在声明的message identity、观察端及起止边界内报告通信时间＝总等待＋固定开销＋传输时间。重叠阻塞区间 MUST 只计一次；本观察链已计固定服务的区间 MUST NOT 同时计作等待。双方重叠服务与等待 MUST NOT 相加为整步墙钟，服务工作量统计 MUST 与区间分解区分。细分等待原因 SHALL 为可选诊断。运行态等待 MUST NOT 进入service缓存，也 MUST NOT 仅因等待而预留fabric。

#### Scenario: Peer and link blockers overlap
- **WHEN** F在20µs提交接收后进入等待，A在100µs就绪，链路120µs空闲，纯传输5µs且其余固定成本0
- **THEN** F侧通信等待100µs、传输5µs、总耗时105µs，数据125µs可用；重叠blocker不重复计时

#### Scenario: Link becomes free before peer readiness
- **WHEN** 同一消息中链路改为80µs空闲，其余条件不变
- **THEN** 总等待80µs、数据105µs可用，不累加链路与对端原始阻塞时长

#### Scenario: Charged submission is not charged as waiting
- **WHEN** 本观察端通信区间从0开始，先执行固定提交服务3µs，再等待至10µs，传输服务5µs
- **THEN** 通信总耗时15µs分解为固定3、等待7、传输5，不另加一次3µs；区间和计费归属可核对

#### Scenario: Optional attribution does not change simulation
- **WHEN** 仅切换等待原因细分诊断而保持服务、依赖和负载一致
- **THEN** 总等待、资源调度和makespan均不变

### Requirement: Complete control operations have unique causal service

启用operation_service_profile时，staging MUST 位于编码后和控制发送前；长度读回 MUST 等待长度消息可用并先于内容buffer准备/接收；内容读回 MUST 等待内容可用并先于JSON解析。每项 MUST 按实际调用次数唯一计费，替代覆盖相同操作的legacy设备聚合事件，不额外计内部拷贝/启动/完成同步。codec保持独立覆盖。完整操作 MUST 约束实际调用线程及必要stream依赖，不据此独占整个GPU或fabric；输入及资源等待 MUST 动态产生，不进入操作常数。trace MUST 单列operation_service，明确资源近似且不将其重复计入纯CPU/GPU忙时。取消/失败 MUST 释放调用资源，关闭trace或命中缓存 MUST 不改变时间线。

#### Scenario: Length readback uses measured complete operation
- **WHEN** 长度消息100µs可用，操作资源空闲，完整读回固定服务11µs
- **THEN** CPU长度111µs可用，后续内容准备不早于111µs；不再加另一段11µs设备读回或同步费用

#### Scenario: A calling thread blocks without locking the GPU
- **WHEN** 一个同步完整读回尚未结束，存在同线程后续调用和无依赖的其它GPU工作
- **THEN** 同线程后续调用等待完成，其它GPU工作不因该完整操作被无依据全局阻塞，fabric也不因该操作被预留

#### Scenario: Codec and staging have disjoint coverage
- **WHEN** 同时加载codec profile与完整staging及内容读回条目
- **THEN** encode/decode各一次，staging包装及内容字节转换各只由相应operation条目覆盖，不并入codec再次计费

### Requirement: Engine overhead follows actual invocation boundaries

显式启用 engine overhead profile 时，系统 MUST 在真实可解释的 step、计算提交、消息提交和完成边界计费，通过 host 资源和 DAG 表达顺序或重叠。一次 step 成本不得按层或 block 重复收取；通信内部算法阶段不得重复收取原生一次提交的 host 成本。compute service 与 host submit MUST 有互斥的覆盖范围，底层设备固有启动延迟仍可属于 device service。关闭 overhead profile MUST 不改变协议、数据量或计算依赖。

#### Scenario: Two layers share one step overhead
- **WHEN** 一个 step 执行两层且 profile 指定 step 调度成本10微秒
- **THEN** 调度服务只计10微秒，实际每次计算或消息调用按其独立已验证项计费

#### Scenario: Host can submit while device executes
- **WHEN** 两个无额外依赖的调用共用 host，第一条提交后已进入设备服务
- **THEN** 第二次提交可与第一次设备服务重叠，除非记录的同步调用语义要求阻塞

## MODIFIED Requirements

### Requirement: Each block follows the layer-local AF dependency chain

以下执行位置、传输内容及场景约束适用于 `partial_return` 兼容协议；`vllm_afd` 使用 Native AFD executes F-side combine before hidden return 的完整契约。两种协议均 MUST 保留原始 token/assignment identity、实际 workload、唯一资源计费、有效 dtype 转换和 A 请求/KV ownership，不得丢弃必要依赖。

系统 MUST 根据模型元数据为每个 Decode AF block 的每一层选择 Dense 或 MoE DAG，而不得对所有层复用同一 MoE DAG。Dense 层 MUST 依次完成 A 域 Attention、本层必要 A TP collective、一个 logical A-to-F stage、F 域 Dense FFN、本层必要 F TP collective、一个按 ownership 的 logical F-to-A stage、A 侧 residual/layout 恢复。MoE 层 MUST 依次完成 A 域 Attention、本层必要 A TP collective、按router_location选择的leader Router/Top-K（F Router在输入传输后执行）、一个 routed/shared-input logical A-to-F stage、F 域 Routed 与 Shared Expert branches、一个按 ownership 的 logical F-to-A partial-return stage、A 侧 weighted combine、residual 与必要 layout collective。每个 logical transfer stage MAY 展开为多个 endpoint legs；每个 leg 只等待其实际 producer，下游 rank只等待发给自己的 legs。下一层 A 计算 MUST 等待本层产生其完整输入的最后一个 A 侧节点完成。不同 blocks SHALL 仅由数据依赖和共享资源竞争约束，不得被强制全局串行化。

#### Scenario: One block follows the corrected AF order

- **WHEN** 使用 `partial_return` 协议且 一个单层 Decode AF block 使用 TP 和 EP，且所有事件 service time 非零
- **THEN** trace 展示与该层 Dense 或 MoE 类型匹配的 A compute、必要 collective、A-to-F、F compute、F-to-A 和 A 侧最终汇合顺序
- **AND** 任一事件的开始时间不早于其全部前驱事件的结束时间

#### Scenario: Dense layer omits MoE-only work

- **WHEN** 使用 `partial_return` 协议且 模型元数据将某层标记为 Dense
- **THEN** 该层执行 A Attention、A-to-F、Dense FFN、F-to-A 与 A residual/layout 恢复
- **AND** trace 不包含 Router、Top-K、routed Expert 或 weighted combine 事件

#### Scenario: MoE layer follows the complete dependency chain

- **WHEN** 使用 `partial_return` 协议且 模型元数据将某层标记为 MoE，且 `top_k > 1`、Shared Expert 数大于零
- **THEN** Routed Expert 与 Shared Expert branch各自等待发给自己的 A-to-F legs，每条 F-to-A leg等待其 producer，每个 A weighted combine 等待该 owner 的全部 return legs
- **AND** residual 对每个原始 token 恰好执行一次

#### Scenario: One block completes one layer in dependency order

- **WHEN** 使用 `partial_return` 协议且 一个 block 执行单层 Decode AF
- **THEN** 每个 rank-local、通信与汇合事件的开始时间不早于其全部前驱结束时间
- **AND** 该层只在 A 侧 combine/residual/layout 的最终节点完成后标记完成

#### Scenario: AF boundary communication occurs once per direction

- **WHEN** 使用 `partial_return` 协议且 一个 block 完成任一 Decode AF 层
- **THEN** 该层只产生一个逻辑 A-to-F stage 和一个逻辑 F-to-A stage
- **AND** 每个 stage MAY 记录多个 endpoint transfer legs，但不得为相同 payload 再生成第二套 EP dispatch/combine 搬运

#### Scenario: Next layer waits for the return transfer

- **WHEN** 使用 `partial_return` 协议且 一个 block 需要执行多层
- **THEN** 第 N+1 层 A rank compute 的 ready time 等于或晚于第 N 层最后一个 A residual/layout 节点结束时间
- **AND** Dense 层不需要伪造 weighted combine，MoE 层不能跳过 weighted combine

#### Scenario: Independent blocks may overlap

- **WHEN** 使用 `partial_return` 协议且 前一 block 已释放某资源且仍在后续资源上执行
- **THEN** 后一 block MAY 使用已释放资源开始相应事件
- **AND** 系统不等待前一 block 整体完成

### Requirement: Router and EP communication follow A-domain routing

以下执行位置、传输内容及场景约束适用于 `partial_return` 兼容协议；`vllm_afd` 使用 Native AFD executes F-side combine before hidden return 的完整契约。两种协议均 MUST 保留原始 token/assignment identity、实际 workload、唯一资源计费、有效 dtype 转换和 A 请求/KV ownership，不得丢弃必要依赖。

对于每个 MoE Decode AF layer，系统 MUST 在该层 A TP compute/collective 完成后，按router_location在A leader或收到输入后的F leader执行一次 Router/Top-K，并为当前层的每个原始 token 生成恰好 `top_k` 个逐 token assignments。每个 assignment MUST 保留 request/token identity、A DP owner、A tensor/layout owner、Expert id 和 routing-weight 语义；Expert histogram MUST 仅从这些 assignments 派生。router_location=a时，A-to-F MUST 直接把 assignment 对应 activation 与 Shared Expert 所需原始 token 输入发送到合法 F endpoints；router_location=f时，A-to-F MUST 先发送原始hidden，再执行F Router，不能依赖尚未完成的F路由来决定发送目的地。F-to-A MUST 将 Routed 与 Shared Expert partials 按其 A owner 返回对应 A ranks；F-to-A 只负责传输，不得冒充 weighted combine。各 A rank MUST 对自己拥有的 partials 执行一次本地 weighted combine 和 residual，并在下一层输入布局要求时执行明确的 A TP collective。

#### Scenario: Router runs once on the selected A leader

- **WHEN** 使用 `partial_return` 协议、router_location=a 且一个 MoE block 完成某层 A TP compute 与必要 collective
- **THEN** Router/Top-K 在该 A replica 的稳定 leader 上执行一次
- **AND** routing result 以当前层 assignments 表示，不复用其他层的 histogram

#### Scenario: Synthetic routing does not reset for every request

- **WHEN** 使用 `partial_return` 协议且 balanced synthetic routing 为同一层四个 Decode tokens、`top_k=2` 和八个 Experts 生成 assignments
- **THEN** 它在整个 layer block 上连续、确定性地生成八个 assignments
- **AND** 不得因逐 request 重置起点而把全部 tokens 固定路由到 Experts 0 和 1

#### Scenario: A to F carries routed activations

- **WHEN** 使用 `partial_return` 协议、router_location=a 且 当前层 `T` 个原始 tokens 产生 `T*K` 个 routed assignments，并确定 Shared Expert 输入 owners
- **THEN** A-to-F payload breakdown 由 assignment activation 与去重后的 Shared Expert 原始 token activation 共同构成
- **AND** payload 逐 endpoint、逐 tensor slice 计算，不得只由丢失 ownership 的 histogram 总数推导

#### Scenario: A to F is the EP dispatch

- **WHEN** 使用 `partial_return` 协议、router_location=a 且 当前层 assignments 与 Shared Expert owners 已确定
- **THEN** A-to-F logical stage 的 endpoint legs 和 payload breakdown 覆盖这些目标 F ranks
- **AND** 每个 F compute直接等待发给自己的 legs，不再等待搬运相同 activation 的独立 dispatch stage

#### Scenario: F to A returns partials to their owners

- **WHEN** 使用 `partial_return` 协议且 Routed 或 Shared Expert branch产生一个 output partial
- **THEN** F-to-A 将每份 output partial 直接发往拥有对应 token/tensor shard 的 A rank
- **AND** 该 leg不等待与其数据无关的 F branch，非 owner A rank也不因 iteration barrier 而被视为获得输入

#### Scenario: F to A is the EP combine

- **WHEN** 使用 `partial_return` 协议且 当前层需要从多个 F producers 返回 partial outputs
- **THEN** Planner 生成一个按 owner 分 leg 的逻辑 F-to-A return stage，承载旧称 EP-combine 的返回通信职责
- **AND** 数学 weighted combine 作为随后独立的 A 侧 compute node 执行，不得包含在 transfer service time 中

#### Scenario: A combine collapses Top-K assignments

- **WHEN** 使用 `partial_return` 协议且 `T` 个原始 tokens 产生 `T*K` 个 routed partials 及每 token 一份 Shared Expert partial
- **THEN** A 侧 weighted combine 使用 routing-weight 语义将结果恢复为每个 token 一份 activation
- **AND** residual 和后续层工作量均以 `T` 而不是 `T*K` 计算

### Requirement: Compute resources enforce A affinity and F placement

以下执行位置、传输内容及场景约束适用于 `partial_return` 兼容协议；`vllm_afd` 使用 Native AFD executes F-side combine before hidden return 的完整契约。两种协议均 MUST 保留原始 token/assignment identity、实际 workload、唯一资源计费、有效 dtype 转换和 A 请求/KV ownership，不得丢弃必要依赖。

系统 MUST 让一个 block 在整个 iteration 中保持其 A DP owner，并将 A rank compute、A侧combine、residual 与 layout collective 放到该 replica 的合法 ranks 和 physical resources 上；Router MUST 按router_location绑定A或F的leader。Dense 与 Shared FFN MUST 使用选定 F DP replica 的 TP group。`EP=1` 时 Routed Experts MUST 使用该 F TP group；rank-reuse `EP=TP*DP` 时 Routed Expert compute MUST 使用整个 F world 中实际驻留目标 Expert 的 rank。多个 blocks 若命中相同 physical F resource，MUST 按其容量自然竞争，而不得把派生 EP identity 当成额外设备。

#### Scenario: A work remains on its selected replica

- **WHEN** 使用 `partial_return` 协议且 一个 AF block 归属于某个 A DP replica
- **THEN** 该 block 各层的 A compute、combine、residual 与 KV ownership 均保持在该 replica，Router按router_location执行
- **AND** F ranks 不取得 request 或 KV Cache ownership

#### Scenario: Routed experts may span reused F ranks

- **WHEN** 使用 `partial_return` 协议且 一个 block 的 assignments 命中分布在多个 F TP/DP coordinates 上的 Experts
- **THEN** Expert branches 使用这些 coordinates 对应的已有 physical ranks
- **AND** 系统不为 EP 额外创建 `TP * EP * DP` workers

#### Scenario: EP one keeps work inside the selected F replica

- **WHEN** 使用 `partial_return` 协议且 F domain 使用 `EP=1` 且一个 block 选择 F DP replica 1
- **THEN** Routed、Dense 与 Shared FFN 都只使用 replica 1 的合法 F TP ranks
- **AND** 其他 F DP replica 可服务另一个 ready block

#### Scenario: Shared physical F rank serializes conflicting work

- **WHEN** 使用 `partial_return` 协议且 两个不同 A owners 的 blocks 同时需要同一个 F physical rank
- **THEN** 两项服务按该 resource capacity 排队
- **AND** trace 的 queue interval 解释由 rank reuse 造成的竞争

#### Scenario: Unselected DP replicas remain available

- **WHEN** 使用 `partial_return` 协议且 `EP=1` 的 block 只选择一个 F DP replica 且其他 replicas 空闲
- **THEN** 未选 replicas 不被该 block 锁定
- **AND** 另一个 ready block MAY 在这些 replicas 上并行执行

#### Scenario: Expert compute obeys placement

- **WHEN** 使用 `partial_return` 协议且 Top-K assignments 只命中放置在部分 F ranks 上的 Experts
- **THEN** routed Expert compute 只出现在这些合法 ranks 上
- **AND** 未命中且不承担 Shared Expert 或必要通信的 F ranks 不产生 Expert service time

### Requirement: EP executes routed per-rank workload and waits for required experts

以下执行位置、传输内容及场景约束适用于 `partial_return` 兼容协议；`vllm_afd` 使用 Native AFD executes F-side combine before hidden return 的完整契约。两种协议均 MUST 保留原始 token/assignment identity、实际 workload、唯一资源计费、有效 dtype 转换和 A 请求/KV ownership，不得丢弃必要依赖。

系统 MUST 使用当前 MoE layer 的逐 token assignments 和 F Expert placement 计算每个 Routed Expert execution owner 的真实 assignment workload。`EP=1` 时一个 owner 是所选 F DP replica 的 Expert TP group；rank-reuse `EP=TP*DP` 时一个 owner 是实际驻留该 Expert 的单个 F rank。每个 assignment MUST 恰好归属一个 owner；没有 assignment 的 Routed owner MUST 不产生 routed compute。每个原始 token还 MUST 恰好进入一个 Shared Expert TP group，Shared 工作量不得乘以 `top_k`，其分支 SHALL 与 Routed branches 并行。对应 F-to-A owner legs MUST 等待生成正确结果所需的全部 Routed 与 Shared predecessors。

#### Scenario: Uneven routing produces uneven rank workload

- **WHEN** 使用 `partial_return` 协议且 assignments 将 12 次 routed work 分别分配为 rank 0 上 10 次、rank 1 上 2 次
- **THEN**两个 routed Expert compute queries 分别使用 10 和 2 的 local workload
- **AND** 系统不按 rank 平均分配

#### Scenario: EP one executes each assignment across one TP group

- **WHEN** 使用 `partial_return` 协议且 F domain 使用 `EP=1` 且一个 Expert assignment 命中所选 F replica
- **THEN** 该 assignment 在此 replica 的 Expert TP ranks 上执行各自 local shard
- **AND** assignment 在逻辑总工作量中仍只计一次

#### Scenario: Rank reuse executes each assignment on one F rank

- **WHEN** 使用 `partial_return` 协议且 F domain 使用 `EP=TP*DP` 且一个 assignment 命中 Expert 7
- **THEN** 该 assignment 只在 Expert 7 placement 对应的已有 F rank 上执行
- **AND** Expert TP 为 1

#### Scenario: Shared Expert uses original tokens once

- **WHEN** 使用 `partial_return` 协议且 `T` 个 tokens 使用 `top_k=K` 且模型有 Shared Expert
- **THEN** routed 总工作量为 `T*K` assignments，Shared Expert 总工作量为 `T` tokens
- **AND** Shared Expert 不在每个 EP rank 上重复执行全部 `T` tokens

#### Scenario: Empty expert rank performs no expert compute

- **WHEN** 使用 `partial_return` 协议且 某个 F rank 没有命中 routed assignments，但被选为 Shared Expert execution group 的成员
- **THEN** 该 rank 不产生 routed Expert compute 或伪造的平均 routed workload
- **AND** 若它属于 Shared Expert execution group，仍 MAY 产生语义独立的 Shared Expert local work

#### Scenario: Return transfer waits for the slowest required expert branch

- **WHEN** 使用 `partial_return` 协议且 多个必要 Routed 与 Shared branches 的完成时间不同
- **THEN** 对应 owner partial return 不早于其最后一个必要 producer 完成
- **AND** 较早分支不会使 block 提前进入 A combine

### Requirement: Shared Expert executes once for every original token

以下执行位置、传输内容及场景约束适用于 `partial_return` 兼容协议；`vllm_afd` 使用 Native AFD executes F-side combine before hidden return 的完整契约。两种协议均 MUST 保留原始 token/assignment identity、实际 workload、唯一资源计费、有效 dtype 转换和 A 请求/KV ownership，不得丢弃必要依赖。

MoE Decode layer 中每个原始 token MUST 进入一个 Shared Expert logical branch；该 branch SHALL 与 Routed Expert branches 并行。Shared workload MUST 按原始 `T` 个 tokens 计算，不受 `top_k` 或 EP identities 数量放大；多个 Shared Experts MUST 通过扩大 shared intermediate 表达。Shared output MUST 在 A owner 侧与 routed weighted result 汇合。

#### Scenario: Routed and shared workloads use different counts

- **WHEN** 使用 `partial_return` 协议且 `T` 个 tokens 使用 `top_k=K` 且模型含 Shared Experts
- **THEN** Routed 总 workload 为 `T*K` assignments，Shared 总 workload 为 `T` tokens
- **AND** 两个 branches 可在资源允许时重叠

#### Scenario: Shared output participates in the A join

- **WHEN** 使用 `partial_return` 协议且 某 token 的全部 Routed partials 已返回但 Shared partial 尚未返回
- **THEN** 该 token owner 的 combine/residual 尚不可 ready
- **AND** Shared partial 到达后才完成最终求和

### Requirement: F results return to owning A ranks

以下执行位置、传输内容及场景约束适用于 `partial_return` 兼容协议；`vllm_afd` 使用 Native AFD executes F-side combine before hidden return 的完整契约。两种协议均 MUST 保留原始 token/assignment identity、实际 workload、唯一资源计费、有效 dtype 转换和 A 请求/KV ownership，不得丢弃必要依赖。

F-to-A MUST 是按 token/tensor ownership 的 partial return，而不是 mathematical combine。每个 F output MUST 按目标 A layout 切分为 owner-specific legs；每个 A TP rank只等待并本地 gather 属于自己的 Routed 与 Shared partial shards，再执行 weighted combine 与 residual。非 owner ranks MUST 通过显式 layout collective 获取其所需数据，iteration barrier 不得替代数据移动。

#### Scenario: Full Expert output is scattered by A tensor shard

- **WHEN** 使用 `partial_return` 协议且 rank-reuse Expert 产生一个 hidden width 为 `H` 的完整 output且 A 使用 `TP=2`
- **THEN** F-to-A 为两个 A tensor owners 各返回一个精确 `H/2` shard
- **AND** 两个 payload shard 聚合后恰好为一个 `H` output，不重复完整 tensor

#### Scenario: One A rank waits for all of its EP partials

- **WHEN** 使用 `partial_return` 协议且 同一 token shard 的 Top-K partials 来自两个 F producers
- **THEN** 对应 A owner 在两份 partial 都到达后执行 local combine
- **AND** 另一个 A rank 的到达状态不能替代该依赖

### Requirement: Decode payloads follow layer and ownership semantics

以下执行位置、传输内容及场景约束适用于 `partial_return` 兼容协议；`vllm_afd` 使用 Native AFD executes F-side combine before hidden return 的完整契约。两种协议均 MUST 保留原始 token/assignment identity、实际 workload、唯一资源计费、有效 dtype 转换和 A 请求/KV ownership，不得丢弃必要依赖。

Planner MUST 从具体 tensor items 与 endpoint legs 计算 payload，而不得只用一个无 owner 的总 token count。每个 source/target tensor shard MUST 以全局hidden-index区间和`sharded`、`replicated`或`partial` layout描述；跨不同TP sizes的legs MUST 由source/target区间交集产生。Replicated source MUST 由稳定canonical producer发送，数学上不同的partial sources MUST 全部保留。Dense A-to-F MUST 携带原始token activation；MoE在A Router时A-to-F MUST 携带每条routed assignment的activation和Shared branch所需原始token input；F Router时 MUST 先传原始hidden，由F在收到输入后路由。Dense F-to-A MUST 携带每token output；MoE F-to-A MUST 携带Routed partials与Shared output partials。相同source slice到相同endpoint的同一logical copy MUST 去重。每个方向 MUST 使用其effective transfer dtype；若与activation dtype不同，边界转换 MUST 作为独立compute nodes计时并按source/target worker、stable tensor item identity、slice interval和dtype pair去重。

#### Scenario: MoE return preserves Top K partials until combine

- **WHEN** 使用 `partial_return` 协议且 `T` 个 tokens、`top_k=K` 在 A 侧 combine
- **THEN** F-to-A 在 combine 前保留 `T*K` Routed partial identities 及每 token一份 Shared partial identity
- **AND** 返回后 activation 数才由 A weighted combine 恢复为 `T`

#### Scenario: Endpoint payload is the sum of unique tensor items

- **WHEN** 使用 `partial_return` 协议且 多个 logical consumers 引用同一 source tensor slice 与同一 destination endpoint
- **THEN** 该 endpoint leg 的 payload bytes 只计一次物理 copy
- **AND** 不同 destination endpoints 仍分别进入 payload breakdown

#### Scenario: Different A and F TP sizes map by global intervals

- **WHEN** 使用 `partial_return` 协议且 相同hidden tensor分别从`A_TP=2`映射到`F_TP=4`，或从`F_TP=4`返回`A_TP=2`
- **THEN** 每条leg只携带source与target全局区间的非空交集
- **AND** 所有unique intervals拼接后恰好覆盖global hidden width且无缺口或重复

#### Scenario: Replicated output selects one canonical producer

- **WHEN** 使用 `partial_return` 协议且 F TP all-reduce后多个ranks持有同一replicated output
- **THEN** 每个target tensor item只由该group最低domain rank的canonical producer发送
- **AND** 其他replicas不重复发送相同tensor item

#### Scenario: Partial outputs are not deduplicated as replicas

- **WHEN** 使用 `partial_return` 协议且 多个F producers持有同一hidden interval的不同mathematical partials
- **THEN** A owner接收所有必要partials并在combine/reduction语义中保留各自identity
- **AND** payload去重不得删除这些数值不同的贡献

#### Scenario: Conversion is deduplicated around fan out

- **WHEN** 使用 `partial_return` 协议且 同一source tensor item从BF16编码为FP8并发送到两个不同endpoints
- **THEN** source worker执行一次该item的encode，两个endpoint legs复用encoded item
- **AND** 每个target worker分别对自己收到的item执行一次decode

#### Scenario: Distinct tensor items do not share conversion

- **WHEN** 使用 `partial_return` 协议且 两个不同tokens或两个不同Top-K partials在同一stage、同一worker上具有相同hidden interval与dtype pair
- **THEN** stable tensor item identity使它们各自生成encode/decode work
- **AND** 去重不得把数值不同的items合并为一次转换

### Requirement: Latency providers return service time only

系统 SHALL 通过可注入 compute provider 分别查询单 block、单层、单 rank 的 A Attention、协议指定位置的 Router、F Dense FFN、F Routed Expert、F Shared Expert、协议指定位置的 weighted combine、A residual、必要 layout operation以及跨dtype边界的encode/decode service time，并通过 transfer provider查询每个endpoint leg或collective的service time。Provider 返回值 MUST 为有限且非负的秒数，并 MUST 携带可追溯 source、dataset、match type 与 composition/transfer details；这些元数据 MUST NOT 参与资源选择或调度。所有 composition MUST 互不重叠，每个逻辑 operation MUST 恰好计时一次。Queue duration MUST 仅由执行器申请共享资源产生，且 AF 正式路径 MUST NOT 将完整 iteration 时延拆分为静默 fallback。

#### Scenario: Stub providers drive a deterministic execution test

- **WHEN** 测试为 Dense 与 MoE DAG 的各 compute 和 communication operation 注入固定 service time
- **THEN** executor 按这些值与 DAG 依赖推进模拟时间
- **AND** 资源等待仍由事件竞争独立产生

#### Scenario: Operator-data provider drives service time

- **WHEN** 正式 provider 为任一 A、F、conversion 或 communication query 返回 service time 与 provenance
- **THEN** executor 只使用 service time 推进该节点的 service interval
- **AND** metadata 不改变 placement、resources、dependencies 或 start time

#### Scenario: F residual is not charged per assignment

- **WHEN** `top_k=2` 的 MoE layer 在两个 F ranks 上执行 routed work
- **THEN** 两个 Expert composition 均不包含 residual add
- **AND** A residual 只按原始 token 数计时一次

#### Scenario: Dtype conversion is charged once per boundary side

- **WHEN** transfer dtype 与 activation dtype 不同
- **THEN** 对应发送 encode 与接收 decode nodes 各查询一次 compute provider
- **AND** transfer provider 不再包含或重复这两项 compute service

#### Scenario: Invalid provider duration fails explicitly

- **WHEN** 任一 provider 返回负数、NaN 或无限值
- **THEN** 系统终止该 AF iteration并报告 block、layer、rank 与 operation

#### Scenario: Required estimate is unavailable

- **WHEN** provider 无法为任一必要 compute、transfer 或 collective query 提供受支持结果
- **THEN** 系统明确报告完整缺失查询上下文
- **AND** 系统不得跳过该节点或改用完整 iteration 时延


### Requirement: Ready input preparation has audited unique coverage

系统 SHALL 允许a_input_prepare完整操作：计时前排除前序设备工作，覆盖原生_prepare_inputs的CPU组织、block table/输入H2D、position/seq-len/slot mapping及自身完成；不包含codec、模型层计算或NCCL等待。接入 MUST 替代同覆盖a_step_prepare_us且拒绝旧host profile叠加。首版仅对经审计不含上述输入准备设备项的analytical_only计算路径启用；其他provider在覆盖未证明互斥时拒绝。每A worker每步唯一计费，约束调用线程并完成后才启动模型与控制操作，不独占GPU/fabric。同步补测扰动与正式无hook整步验收 MUST 分开；batch4不得用于拟合。

#### Scenario: Input preparation is charged once before use
- **WHEN** 输入准备固定服务P、已有同覆盖手工或host profile参数未指定
- **THEN** 每步恰好一次operation_service，控制编码与模型计算均等待准备完成；未知覆盖provider或显式重复成本执行前拒绝

A/F准备补测同时覆盖a_attention_metadata与f_dp_metadata。前者为原生attention metadata构建（包含必要自身设备完成，不含控制发送），位于输入准备后/控制编码前；后者为F的_make_ffn_dp_metadata纯本地整理，位于控制解析后/F data接收前，MUST NOT 提前到step入口。两项按三窗口实测固定操作唯一计费，拒绝与笼统A/F step准备叠加；计费资源为对应调用线程，F数据依赖 MUST 指向metadata完成事件。样本与无hook整步时延分别报告。

### Requirement: Current hierarchical calibration is explicitly refrozen

系统 SHALL 支持当前node-domain FIFO实现的同节点GPU2/3原生校准，显式配置位置、带宽及固定延迟；runtime指纹 MUST 包含通信域配置及模型版本。旧host/operation参数仅经覆盖审核后派生新profile，MUST NOT 绕过指纹或覆盖冲突校验；旧实验 MUST 保留。新参数与源码 MUST 在batch5实测前冻结，batch5不得用于拟合；至少三独立无探针窗口和独立路由采集，分项缺口仍单列。

#### Scenario: Domain configuration changes
- **WHEN** 本地带宽、节点摆放或通信域模型版本改变
- **THEN** 旧profile被拒绝，审核并重新冻结后才能进入新版本验收

#### Scenario: Supplemental batch was already observed
- **WHEN** batch3已用于旧版本补充验证
- **THEN** 新版本使用尚未观测的batch5作为留出，不将batch3重标为未见负载
