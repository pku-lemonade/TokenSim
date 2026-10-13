## Context

2026-10-12剩余任务续作的最新边界与冻结规则见`docs/plans/afd-completion-contract-20261012.md`。在新增API类型/拷贝方向审计及独立GPU服务校准后，v4允许已排除外部等待的完整CPU提交服务，覆盖并替换原framework-only条目；不得与旧设备启动假设叠加。该规则取代下文历史记录中“未经审核的API区间不得并入固定费”的未完成状态，不改变旧v1–v3语义。新GPU偏移仅拟合GPU服务，不使用整步差额；transport按backend/dtype/方向/范围严格匹配，新留出为batch7。

动机和范围见 proposal.md；证据见 docs/plans/afd-1a1f-alignment-20261006.md。当前 `executor.py` 已用 start-ready 与 end-start 分开记录 queue/service；启用 communication_timing 时还生成 host_submit、transport、synchronize，并拒绝 timing_scope 不明确的 transfer estimate。`AFCommunicationTimingConfig.enabled` 默认 false，本轮 `run_tiny_afd_alignment.py` 和其基础配置未显式开启。因此“已有排队分离”和“已完成真实引擎校准”不是同一状态。

另一个尚未实施的提案 rebuild-hierarchical-af-communication 负责物理阶段、整路径原子预留与拓扑。本变更针对当前单机 NV12 1A1F 对照，不以完成该拓扑重建为前提，也不把其未完成任务标为本次完成。

当前 planner 和 trace 多处要求每个 A owner 有 combine 节点；executor 的模型层结束判断也依赖现有终端节点。仅把返回 bytes 除以 top_k 会违反数据依赖和计费契约，必须按协议一起调整。

## Goals / Non-Goals

2026-10-07 本地实现补充：host profile v1 和 routing replay v1 的字段、拒绝条件、计费覆盖及复现命令见 `docs/plans/afd-host-profile-and-replay-20261007.md`。host profile 仅覆盖额外 CPU 服务；device copy/transport 仍独立待校准，原生实测算子必须声明 device_service，grouped expert 必须排除单独计费的 combine。标准 LLMEngine 的 native 协议统一使用完整模型执行器。真实 capture/实测参数/完整引擎验收未完成时继续标记 uncalibrated，不改变原定验收范围。

**Goals:** 用可审计的协议和计时边界解释数量级偏差；先修正 native 对照 DAG，再测量无外部等待的纯服务或明确覆盖的完整操作服务；在通用运行时而非脚本子类中表达必要 host 工作与依赖；保留历史 baseline 和每一步修正的归因。

**Non-Goals:** 首版不支持多 A/F、跨节点、shared expert、graph/DBO、多种 transfer dtype 的原生协议仿真，不验证 DSV3，不模拟每个网络 packet，不修改安装中的原生插件业务逻辑，不以总倍率追平整步，暂缓完整prefill到decode流程对照。

## Decisions

### 2026-10-10 原生提交与数据完成依赖

新增显式`native_call_order_enabled`，仅用于已有native control的1A1F eager支持域，默认关闭以复现旧冻结结果。开启后进入新的runtime指纹，旧profile必须经同硬件/形状/覆盖复核后生成派生产物，不能静默套用旧运行指纹。

F控制长度接收从iteration入口准备，payload接收等待长度CPU回读。A控制payload发送提交仅等待长度发送提交，不等待F长度回读；A后续CPU计算提交等待payload发送API返回，实际A设备计算仍等待payload传输完成所代表的流依赖。F首层data接收等待metadata完成，下一层接收等待上一层F发送提交；A返回data接收等待同层A发送提交。接收提交不依赖对端计算完成，也不提前占用fabric。实际transport仍等待发送端数据、双端提交及动态资源可用。

调用组内CPU与kernel逐条交错继续采用已声明的聚合近似；本项修正消息API顺序和必要数据依赖，不宣称重建每个kernel enqueue。控制读回仍为完整操作服务，外部等待不能进入常数。取消或任一control estimate失败必须结束所有预贴进程并释放host资源，可再次运行。

### 2026-10-09 原生调用组覆盖的实施细化

本次CPU计算常数采用调用线程的原生函数wall减去CUDA runtime/driver API嵌套区间并集；前序设备同步位于计时外。保留的范围是CUDA API之外的框架CPU组织，不能称为完整runtime launch已校准。已有provider设备启动下限继续作为分析假设，未测API成本单列缺口；不把API区间重新并入固定费。空计时器探针只衡量入口/出口调用本身，不代表全部CUPTI扰动，另用无CUPTI同边界诊断及无hook整步窗口检查。

F原生`Qwen3MoeSparseMoeBlock`一次调用包含内部router、grouped experts和combine，采用`compute_f_moe_block_us`在router前每层计一次；与独立router/expert参数互斥，后续expert/combine不再追加host提交。原有设备计算及工作量保持不变。CPU组整体放在设备组前是当前eager DAG聚合近似，不能解释为精确kernel提交/stream回放。

输出细分为token readback、`engine_runner_bookkeeping_us`、`engine_output_process_us`：bookkeeping测量扣除整个嵌套_to_list区间，仅覆盖同步runner其余CPU组织；output字段仅为Scheduler.update_from_output与frontend.process_outputs。运行时依次建立数据依赖，每步各一次。异步调度、额外logprobs/routed-expert D2H不在当前样本范围。

同路由三组消融中，“仅控制与外围”需要保留token-ready依赖，使用显式synthetic_test的零成本readback反事实；不能将该零值标为实测或导出为生产profile。正式冻结bundle保留实测56.342µs，不修改旧验收记录。

### 2026-10-08 三类剩余固定成本的最小补齐

用户授权自行修正，采用固定常数优先的补测与接入，不以等待细分、完整CPU微架构或重做GPU估值为前置。诊断来源为`docs/plans/afd-cpu-attribution-20261008.md`：按同一步非重叠区间定位调用范围，但区间包含异步执行和同步等待，不能直接拟合为固定费。现有六项operation和codec实测值继续保留；旧零值或5µs假设只有被合格实测条目替代后才标为已校准。

**采样完成：** 首版在显式启用新完成操作的原生decode配置下增加`sampled_token_readback`完整操作，位于GPU token_selection及相关stream前序工作完成之后。测量使用真实pinned buffer、copy、event和CPU结果转换，前序设备同步放在计时外，计时包含自身完成同步；保持真实batch和缓冲复用方式。CPU token可用之后才完成当前decode请求状态更新并允许依赖该token的下一步调度。等待前序GPU/NCCL不进入常数，也不复用通信host_completion_us作为采样费用；输出CPU整理若已包含tolist等步骤必须剔除同覆盖部分。旧配置不启用条目时保持历史边界，并明确标为未建模，不静默改变已冻结结果。取消/异常释放调用资源，不额外锁定GPU/fabric。

**计算提交：** 优先补测真实A/F调用组的CPU组织与提交服务。以来源函数、调用组、role和每步/每层调用次数绑定到已有执行节点；一次grouped expert调用不按expert或其拆出的数学combine节点重复收费。CPU API返回不等于GPU完成，后续无依赖提交可与已提交设备工作重叠，设备消费仍等待输入/stream依赖。已有通用compute_submit_us仅保留为旧近似；若实际调用组成本不同，扩展profile的按调用组覆盖映射，不用一个全局常数乘全部DAG节点。不混入GPU执行/外部同步；现有device launch常数若覆盖CPU提交，先拆分覆盖或拒绝叠加，不能只凭不同字段名视为互斥。

**控制提交与外围：** torch.distributed NCCL控制和PyNccl data分别审核发送/接收提交范围；API中的work.wait按实际返回/stream语义处理，CPU阻塞等待不得成为固定提交常数。按backend/方向选择经验证的常数，只有三窗口证据支持时才共享。引擎调度放在本步输入准备前，输出CPU整理放在token可用后，每step一次；避免再次收费已覆盖的_prepare_inputs、metadata、codec及readback。使用已有host资源和profile，只增加必要的调用种类/匹配字段及验证，不新建独立的开销调度系统。

**拟合与对照：** batch1/8各三独立窗口，先验证ready边界、实际执行路径、调用计数和输出，再对窗口中位数等权拟合常数，报告batch残差与分位数。不根据2ms整步差额、cProfile热点或整个sample_tokens wall填数。常数不能满足既有数量级目标且残差随batch稳定增长时，才使用现有非负规模项；缺证据仍保持未测。已知GPU2被占用时可以先完成本地实现与oracle，远端测量遵循原有空闲保护。

保存旧冻结bundle不改写；新profile生成新指纹及覆盖版本。batch4旧验收保留，作为新版本回归点不冒充新的盲测；本轮新增batch2作为未参与选参/拟合的留出点，在参数冻结后运行。batch1/8、batch4回归和batch2留出各三独立无profiler正式计时窗口，真实路由诊断另采。分别只启用采样完成、计算提交、控制/外围及全部启用，使用同配置路由报告边际变化和最终关键路径；旧配置、cache/trace等价、资源释放同时验收。4.4/4.5和5.2仍由完整证据验收，新增任务不自动关闭它们。

### 完整操作固定服务（本次用户确认）

目标是当前tiny模型1A1F数量级对齐。新增独立 `af.operation_service_profile`，条目scope为`operation_service`，不放宽旧host profile v1或纯device/transport的scope限制。这里的service指“排除外部等待后，完成声明操作所需的区间”，允许包含CPU/runtime、自身设备工作及等待自身操作完成的同步；不等于CPU busy或GPU busy。

首批白名单为control_staging、control_length_readback、control_payload_readback。每项记录数值/单位、操作与覆盖、输入ready条件、计时起止点、调用返回与数据可用的关系、调用线程/必要stream依赖、硬件软件与消息大小适用域、样本来源、三窗口统计和拟合规则。保持batch1/8训练、batch4冻结后留出。优先复用afd-control-staging-20261007的原样本；先核对操作与原生代码一致、同步在计时外、返回是否保证所需完成，再按各batch/window中位数等权汇总。数据源原有“不适用于纯host/device”标记保留，新建有审计来源的operation_service产物；不能只将原数据改标签。约38/11/20µs不是跨硬件默认常数。

放置及覆盖：staging在编码后、控制发送前，包含bytearray/frombuffer、H2D和长度tensor构建，排除编码与NCCL；长度读回在长度消息可用后、内容buffer分配/接收前，首批.item()测量不覆盖后续分配；内容读回在内容消息可用后、解析前，覆盖.cpu().numpy().tobytes()，排除JSON解析。现有6.163/14.971µs codec成本继续独立。尚未测得的接收buffer分配等项必须列缺口，不自动并入读回常数。

输入依赖满足后才申请对应调用线程，等待输入和线程排队都属于动态等待，不进入固定项。首批以完成调用的操作级事件建模：同一调用线程在事件结束前不能提交受阻后续工作；使用显式调用线程资源或经验证等价的现有host资源，不无依据独占整台CPU、整个GPU或fabric。输入未到时不预留服务资源。原操作涉及的stream依赖保留；对独立copy engine占用尚未细化的近似必须标注，首版限单in-flight原生控制路径，不能推广为并发copy竞争已验证。若API返回早于必要数据可用，单一常数不能发布数据ready，必须保留单独完成事件或补测到完成的明确边界。

同一覆盖只能选择完整操作或分解子项之一。启用operation条目时，替代对应legacy control_h2d_us/control_length_readback_us/control_payload_readback_us服务；默认零值不另收费，显式手工值、其它profile或拆分设备事件覆盖相同操作时执行前拒绝。不能额外加内部copy、runtime launch或完成同步。旧配置不启用新profile时保持旧设备聚合近似，不静默改变历史实验。

新scope、profile内容指纹、覆盖及调用边界进入估值/计划隔离；只缓存固定服务参数，不缓存等待或事件。报告将operation_service单列并纳入同观察窗口的固定成本，内部同步已包含，不再次归为外部等待；不得把其duration重复计入纯host/device忙时。对端观察到的等待可与该服务重叠，但整步仍按事件完成计算。

剩余大项按真实调用边界补测：A输入准备、A/F计算提交、两类消息提交、输出处理和完成调用。比较输入已就绪与人为延迟条件，稳定操作服务作为候选，延迟进入依赖等待；同时核对嵌套覆盖和采集扰动。仅“延迟不变”不足以证明无外部等待。原native metadata整段含NCCL和外部同步，仍不允许当作固定常数。正式复测先batch1/8同配置/路由，冻结后batch4；不通过时保留差异，不以整步差额补齐。

### 本地工作收口（2026-10-07）

native control开启的单in-flight模式按最大af_block_size预留一组逻辑协议buffer，跨层复用：A residual/send/return hidden各T*H*dtype_bytes，F input/output hidden各T*H*dtype_bytes，另计F router logits、top-k ids/weights及双方控制staging。KV容量在扣除weights及该预留后计算；此项不是整个CUDA allocator峰值，kernel workspace/碎片仍须实测。旧partial-return和控制关闭基线保持原容量口径。对照驱动固定af_block_size=32，与原生max_num_seqs一致，batch1/8/4共享运行指纹。

实测expert表声明融合weighted_combine时，provider在纯service估值中保留覆盖集合；executor仅为本次iteration保存完成覆盖，combine为零设备服务/零host提交的语义节点，结束/失败清除，不将执行状态放入估值缓存。未命中而使用analytical服务时不继承融合覆盖。

原生采集工具仅在新隔离副本插入诊断hook，采集A真实请求行/token位置、F实际top-k及buffer shape。tensor回读造成的扰动明确标记，不能进入service拟合。训练/留出计时和诊断采集分开，batch4运行前冻结profile；每个启动前检查GPU进程、显存、利用率和端口。准备文件不代表占用/预留GPU。详见docs/plans/afd-local-completion-20261007.md。

### 0. 固定host近似首版（2026-10-07用户确认）

在现有`af.communication_timing`中增加`a_step_prepare_us`、`f_step_prepare_us`、`compute_submit_us`、`host_receive_submit_us`，默认0，全部要求有限非负且拒绝bool。非零项首版仅用于`vllm_afd`、分离计时开启的decode支持域。配置是手工未校准假设；不给出伪造的硬件默认校准常数。

A/F准备在iteration入口各自启动，每个参与worker只执行一次，分block/分layer不会重复。A计算等待A准备；F接收提交等待F准备，允许F准备与A计算重叠。接收提交从该消息的数据依赖完成时建立，属于保守的按需接收近似，不声称已复现原生提前预贴recv。双方提交均完成后申请fabric，已有外部readiness事件仍有效。传输结束释放fabric，原有completion完成后才放行下游；等待不占host服务资源，也不加入纯传输估值。

`compute_submit_us`按实际模拟compute DAG节点收一次额外host调度成本，grouped专家节点不乘专家数；它只覆盖既有算子估值之外的CPU调度/包装工作，不能再次包含已有kernel_launch_us或把一个节点当成一个native kernel。细粒度调用覆盖与正式profile仍由2.3/3.3/4.5验收。每步准备可近似独立的输入准备/控制编码CPU工作；依赖设备数据的回读、metadata网络传输不允许提前折入它们。

host准备/提交共用已有per_worker/shared资源，等待与纯服务分开记录。固定成本不进compute/transport缓存；每次iteration重建准备事件，取消/失败释放资源。ModelAFExecutor与通用AF入口复用实现，trace关闭不改变时间。完整native metadata协议、host调用回放和校准不因固定成本接口完成而标完成。

验收oracle：A计算100µs、F准备200µs、传输5µs、空闲链路时transport=[200,205]µs，peer等待100µs，fabric排队0；不能得出305µs。追加完整模型、多个iteration/blocks、shared host竞争、trace/cache等价及取消后复用测试。

### 1. 显式协议，首版绑定已经实测的支持域

#### 控制协议首版实现约束

源码审计确认`connectors/metadata.py:376/403`走torch.distributed send/recv（CUDA/NCCL进程组），并非activation使用的PyNccl custom op。`native_control_enabled`默认false供旧基线复现；新的native对照驱动默认开启，关闭时报告disabled_baseline。首版仅一个未分块eager stage和一个in-flight iteration，拒绝多block或并行事务，避免把模拟分块误当原生ubatch。

每步控制内容按原生compact/sorted JSON序列化，stage0的DP token列表为[batch]；graph/warmup/replay/profile标志均false。首消息为int64长度8 bytes，第二条为uint8 JSON字节，batch1/4/8均179 bytes（由远端原函数CPU-only审计验证），batch16为181。payload大小按序列化内容计算，不能硬编码179或并入hidden。

固定`control_encode_us`覆盖原生encode_control_payload函数（不含后续bytearray/frombuffer等staging包装），`control_h2d_us`覆盖长度/内容的设备staging，`control_length_readback_us`覆盖长度回读服务，`control_payload_readback_us`覆盖内容回读服务，`control_decode_us`覆盖JSON解析/控制状态准备。上述默认0、有限非负，非零要求native_control_enabled；它们与原step准备的覆盖不重叠。未启用operation_service_profile时copy沿用所属worker设备资源的旧保守聚合近似；启用后按完整操作固定服务一节替代对应事件，约束调用线程和必要依赖，不把完整操作一律记成GPU独占。外部GPU/NCCL等待由前驱决定，不拟合进回读常数；操作自身完成同步可包含在声明覆盖中。

F长度回读完成后才能进入内容接收。聚合模型在内容交付时释放A执行model forward；F还需内容回读和解析，随后才允许activation接收。这保留A计算与F解析的重叠，发送端局部API返回/stream边界仍待原生实测，不能声称已精确回放。现有消息submit成本用于每条控制/数据消息，control不再收data completion常数，读回/解析独立收费。fabric由控制/数据实际共用，backend写入query与service缓存键，不复用另一后端的校准语义。

控制路径未替代真实路由回放或正式profile，2.4/3.1/3.2等整项仍按原验收要求保持未完成。详细实现和源码覆盖见docs/plans/afd-native-control-20261007.md。

已核查远端Qwen3插件：A的mlp是RemoteFFNProxy，不加载完整FFN，因此不能直接执行本地prefill。用户已确认暂缓完整流程对照，不扩展插件；原生prefill仍按插件现状运行但不纳入本次指标。模拟decode从给定context开始，不能宣称验证了prefill阶段衔接。

新增 `af.execution_protocol`，默认 `partial_return`，原生对照显式 `vllm_afd`。后者限定无 shared expert 的 Qwen3 MoE、1A1F/TP=DP=EP=1、BF16、F router，仅 decode（af.phases=[decode]）。拒绝不支持组合。将 gate 位置与 combine/return 协议分开；现有 router_location=f 不隐式等于 vLLM 协议。原生协议自动确定Router在F，不作为本轮独立调参项；旧协议保留A/F Router。

原生协议 DAG：A Attention（包括所需 Q/K norm）→发送完整 hidden→F gate/top-k→grouped expert+combine→返回完整 hidden→A residual→后续层。F 返回已加权结果，A 保有 residual/KV/请求状态；结束判断改为协议定义的终端集合，不再硬编码 A combine。纯逻辑 token/assignment 数仍分别为 T 和 T*K。

每层每方向 data bytes=T*H*2；两层 batch=1/8 的每方向总量应为1024/8192 bytes。metadata 按实际消息记录独立建模，不把它混入 activation 或任意定为零。物理拆包与消息数以插件源码和消息采集确认，不由“一个逻辑 stage”推断只有一次 NCCL 调用。

保留旧协议的原因是现有多 rank ownership 行为和历史测试仍有用途；直接全局移动 combine 会在尚未定义的多 F 场景引入隐藏归约。规范完整复制受影响的旧 requirement 并限定旧协议，新协议有独立完整契约；apply 时还需核对前置 delta，避免原有 F-router 扩展被旧主规范覆盖。

### 2. 数学操作、设备服务与 host 调用分别表达

保留路由 identity 和实际 per-expert histogram；native 严格对照回放每 step/layer assignments，不以合成均匀路由代替。grouped expert 区分上层调用与内部多个kernel提交；允许整体测量host服务但必须声明覆盖范围，内部 workload 仍保留完整专家分布。若测量服务包括 weighted combine，则该区间以融合覆盖集合计费，combine 完成可用零额外服务的语义完成点表达。若解析估值拆算专家与 combine，保留各纯服务及实际 grouped 提交边界，不按专家数重复加 host launch。

Q/K norm、embedding/logits、采样、metadata 与 dtype 转换核对覆盖表，每一项有唯一归属；GPU 采样属于 device compute，CPU 后处理属于 host service。新增 profile 接入时检查旧算子模型 launch 的含义：可剥离的 host launch 只计一次，设备固有启动延迟仍保留，测量范围无法拆分则拒绝混合计费，而非一律删除所有 launch。

### 3. 通信总等待与实测固定成本的统一口径

本轮优先补齐合格实测固定成本并代入通用运行时，再验证整步数量级。已有profile接口与默认零成本不代表已校准；可用项不必等待可选的等待分类完成才接入。无合格样本的项仍标未校准并补测，不从整步差额倒推。

通信时间＝等待时间＋固定开销＋传输时间，是同一观察区间内的统计分解，不是把两端所有服务与等待求和后额外推进模拟时钟。记录message identity、观察端及起止边界；例如F从提交接收后开始等待至数据可用的区间，只分解区间内的成本，区间前的提交成本另列，不混用A/F观察起点。若双方CPU提交有重叠，各项服务工作量可以另列，但不能把它们的总和冒充通信墙钟分解。

等待总量从实际事件推导：对端尚未就绪和链路繁忙同时存在的区间只计一次，不要求决定重叠部分归谁。无中间服务时，总等待为实际服务开始减等待起点；存在提交、回读等服务插入时，按事件分段取等待区间并集，排除本观察链上已计固定服务的区间。已有peer/fabric细项可保留为可选诊断，不增加独立等待估计器，也不把细分类作为固定成本接入的前置条件。

固定成本按实际准备、提交、设备启动或完成位置执行，每项唯一归属；准备成本已经推进就绪时刻后，不再在传输末尾补加。另一端的准备可以表现为本端通信等待，这是不同观察口径，不在整步重复累加。缓存只保存无等待的纯服务和静态参数，等待每次运行重新生成。等待本身不预留fabric，不因等待而无依据锁住整个GPU。

源端发送调用返回与目标端数据可用分开核对，消费操作只依赖真正可用事件。仅有证据的同步调用阻止相应CPU线程继续提交；优先核对控制长度.item()及内容.cpu()读回，不能把所有send/recv假定为CPU同步，也不能将stream依赖扩大为全GPU阻塞。依赖已正确的路径只补验证，不为诊断分类重建调度器。

验收例：F在20µs进入等待，A在100µs就绪，链路120µs可用，纯传输5µs、其余固定项为0，则F通信总等待100µs、传输[120,125]、通信总耗时105µs。100µs以前重叠的对端/链路阻塞不得相加为180µs。若链路80µs已空闲，则总等待80µs、数据105µs可用。两端通信统计不可相加为整步。

以下ready/start/end边界继续用于执行依赖与资源调度，其中peer/fabric归因仅为可选诊断：

对一条消息定义 t_send_submit_end、t_recv_submit_end、必要输入 ready、双方 stream 前置依赖完成；fabric_ready 为这些边界的最大值。fabric_start 由资源竞争决定，fabric_end=fabric_start+S_transport。fabric_queue=fabric_start-fabric_ready。peer wait 是本端提交/输入已就绪后至双方就绪的差，不能同时作为 fabric queue 计费。

host submit 和 completion 各有自身 ready/start/end、queue/service。completion 在 transport 后发生，必要纯完成服务结束后才发布调用完成；transport 结束即释放 fabric。同步 API 若确实阻止同线程后续提交，用调用序列依赖表达；不能让整台 CPU 在等待期间无依据地完全独占，也不能无限制允许被阻塞线程继续提交。接收调用建立、F 预贴 recv/循环顺序从 native 代码确认；输入和必要 metadata 未到齐时不可使 F 计算 ready。

例：源0、目标10就绪、空闲链路 service=5，则 transport=[10,15]、peer_wait=10、fabric_queue=0；若链路占用到12，transport=[12,17]、queue=2，service仍为5。若 completion=2，下游19就绪。以因果顺序划分区间；依赖等待可报告重叠的 blocker，但不能把它们相加为 wall time。

当前单 lane 调度器复用既有 queue/service；未来 hierarchical 模式的内部 phase 也遵守相同边界。原子路径资源的实现仍归另一个提案。两个提案合并时，一次原生消息的 host submit 不能按内部 phase 重复收费。

### 4. 校准固定项和规模项，不拟合未知等待

实施顺序：先盘点既有每项样本的scope与覆盖；已有合格实测固定值直接形成带来源的profile，缺失项针对性补测。首版优先常数，只有重复样本明确显示规模趋势且影响目标时才增加规模项。产物必须列出参数名、数值/单位、覆盖操作、每步或每调用计数、三窗口统计和原始来源；纯device/host服务仍分别接入；无法或无需进一步拆分但外部等待已排除的完整操作使用独立operation_service_profile，不塞入host profile。

随后通过标准engine/ModelAFExecutor实际加载，核对物化值和trace逐项计费次数；同一配置、真实路由分别运行未校准和代入实测参数两组，解释新增服务及关键路径变化。参数冻结后执行batch4留出。等待细分不作为交付门槛；无法隔离外部等待的测量继续诊断标记，不冒充固定开销。已接入的合格项与尚未测出的项分别报告，部分校准不得标为完整通过。

先采集同一窗口 CPU/GPU 时间线并核对跨进程时钟和消息 identity，不能按两端 ProfilerStep 数字强行配对。无 profiler 的 engine.step 墙钟用于端到端验收；profiler 窗口用于归因并单列扰动。A trace 的32个未匹配 kernel 必须解释或标为缺口。

profile 的费用类别为 step_schedule、compute_submit、message_submit、completion，必要的 CPU 后处理归入明确 step 阶段；服务时间可以是经过验证的常数、按形状查表或非负低维模型。固定项先用重复样本估计；确有规模趋势且样本足够时才增加 batch/token/调用数变量。仅两个 batch 均值不足以识别多个系数，应补充训练窗口/工作量；batch=4 留作冻结后的验证，不参与选择参数。本轮只建立decode档案；原生prefill仅建立KV/预热，不用于校准。校准结果不能直接用于 graph、其他模型或拓扑。

一次 step 调度成本只计一次，按真实迭代入口与输出完成边界接入 ModelAFExecutor/标准 engine；ModelAFExecutor 对照入口也需完整覆盖相同边界，不能只在报告总时长上加常数。host 服务使用 per_worker/shared 中与进程/提交线程实际对应的资源；host、GPU、fabric 通过依赖允许合理重叠。等待由事件生成，不进入 profile。

拒绝“实测 step - 模拟 step = 固定开销”的方案，因为其中混有协议差异、路由差异、服务误差和重叠。保留 unaccounted 区间，若无法归因则该环节仍待验证。内部CPU/device组合本身不是拒绝完整操作服务的理由，关键是外部等待排除和覆盖唯一。

### 5. 同款通信测量和数据契约

activation继续使用PyNcclCommunicator，与原生connector的stream、GPU2/3/NV12匹配，涵盖512/1024/4096/8192 bytes及两方向；metadata按源码实际使用torch.distributed NCCL进程组，单独测量8-byte长度和实际JSON。两种后端不得共用含软件开销的拟合常数。人为延迟接收方以验证测量能隔离等待，再以共享资源竞争验证queue。CUDA event范围可能含提交偏差；仅“双方程序启动”不足以证明对等ready，必须记录提交/事件边界。无法可靠隔离时保留诊断数据而不导入transport_service。

保留物理服务 alpha+bytes/B 的语义，alpha 是设备/传输固定服务，不是整个引擎的启动费用。实测可提供相同 scope 的服务表或可辨识参数；小尺寸样本无法可靠识别峰值带宽时不同时拟合 alpha 和 B。此前通用 torch.distributed 的73–80微秒不能直接导入 PyNccl profile。

profile 在拟合前声明 constant 或有界非负线性预测/插值规则及batch支持区间[1,8]，以支持冻结后预测batch4；离散精确表没有batch4项时不得伪造回填。

profile 保存单位、scope、版本、协议/backend、device/topology/source hash、shape/batch/context范围、固定项/规模项、覆盖集合、样本量与统计、训练/留出清单。显式启用 profile 时字段冲突、版本不匹配、越界均失败；手工 host 参数与 profile 同项互斥，配置默认值不视为手工覆盖，不叠加。未校准 baseline 可显式关闭 profile，但分离计时仍开并输出假设项。

### 6. 缓存与报告

计划键加入协议/调用粒度版本；纯服务键加入 profile 内容指纹、timing_scope、操作覆盖、shape、dtype、routing workload、路径及已有配置指纹。缓存不得包含 SimPy event、ready/start/end、resource request 或 queue。同一静态服务反复使用时，每次实例化状态并重新排队。

报告并列 kernel busy、模拟纯 device service、host service、operation_service、等待和关键路径墙钟，说明各自是否包含设备启动；同一统计口径再求倍率。等待和并行 busy 总和不是 step。trace 新增消息关联及等待原因，保留关闭 trace/缓存后模拟时间线相同的保证，按需生成缓存而不扫描全部配置。

详细时间线摘要按需读取events_for(iteration_id)，只索引当前iteration。分别导出service sum、同类别busy区间并集及一条由时间戳支持的关键链；依赖和容量1资源blocking连接后，仅关键链非重叠区间重建makespan。跨iteration资源blocking不扫描历史，而保留resource_queue；缺失依赖/空trace/不明间隙明确unexplained。关键链可能不唯一，不能把跨类别busy并集再相加。关闭trace不影响执行，但不能伪造完整关键链解释。

### 7. 验证矩阵

| 测试 | 主要判据 |
| --- | --- |
| 协议 payload / combine | top_k=1/2 返回相同 T*H*2；F combine 恰好一次；metadata 独立；A residual/KV 不变 |
| grouped/fused 调用 | 多专家只计一次对应 host 调用；融合覆盖不重复收 combine；实际 histogram 一致 |
| 完整 DAG | 两层 decode、embedding/logits/采样边界、下一层等待返回和 residual |
| 双端 ready | 接收延迟0/10微秒只改变 peer wait，service=5不变 |
| fabric 竞争 | 同 lane 两次 service=5，queue=0/5；缓存开关一致 |
| host/完成 | step10只计一次；submit与设备可重叠；completion2只延迟消费不占 fabric |
| profile 校验 | scope不明、含外部等待、负/NaN、错误backend/hash、越界、重复计费均拒绝；operation_service允许已声明的自身完成同步 |
| 完整操作 | 长度100µs到达+读回11µs→111µs可用；输入推迟20µs只增等待；调用线程阻塞、其它GPU工作不被全局锁定；旧/新覆盖冲突拒绝 |
| 原生测量 | 同窗口关联、未匹配kernel、data/metadata、同款PyNccl、人工等待与扰动报告 |
| 精度 | batch1/8训练、冻结后batch4；每点至少三组无profiler窗口；分环节及整步max/min<10 |
| 回归/性能 | 旧协议、AF-disabled、标准受支持prefill/decode；trace/cache等价；事件数、wall和内存增量 |

零竞争的 queue 应为零；人为延迟用因果 oracle 验证。无法观测/分离的指标标为未验证，不以宽松倍率遮盖。保留未校准原始结果、协议修正未拟合结果、校准结果三组；基线不被覆盖。tiny 通过不代表 DSV3 或多节点通过。

## Risks / Trade-offs

- [tiny 模型固定开销占比高，容易过拟合] → profile 限定支持域，参数冻结后留出验证，后续大模型单独测量。
- [本地插件有补丁且没有Git历史] → 保留源码hash与命令，任何源码变化生成新profile，不能宣称验证原版上游。
- [profile与旧launch/通信alpha重复] → 操作覆盖表和同项互斥校验；每项唯一归属，拒绝不可分解测量。
- [双端预贴接收改变真实硬件占用] → 首版分离逻辑等待与纯服务，记录NCCL waiting kernel可能占用SM的限制；若影响数量级必须补测并复审资源占用模型，不把等待改称传输。
- [新native模式与旧trace不变量冲突] → 按协议校验终端、combine位置及ownership，覆盖缓存切换与取消清理测试。
- [GPU不可用] → 空闲保护，不干扰其他进程；保留待测项，不伪造profile或改卡拼接。

## Migration Plan

1. 保存当前源码、配置、基线和原生源码哈希；读取前置delta，记录与hierarchical提案的交接约束。
2. 增加协议选择与支持域校验；同时改planner/provider/trace/完成判断与缓存键，交付确定性协议测试。
3. 开启对照分离计时，补双端ready和host边界，验证等待因果、资源释放及唯一计费。
4. 隔离采集并核对真实调用、路由和scope；测量、生成profile、冻结参数，再做batch4验证。
5. 保存三组对照及不足项，执行相关回归；若失败则保留证据，不用全局倍率掩盖。
6. 旧配置维持partial_return；回滚可禁用profile并选择旧协议，结果同时输出版本。规范同步先完成integrate-deepseek-v3-core-runtime，再同步本变更；hierarchical以后实施时语义合并，不整段覆盖本次新增契约。

## Open Questions

### 2026-10-07 已确认问题修复与后续测量约束

用户授权修复已知问题；允许在隔离插件副本修复DBO guard的重复NPU导入，原安装副本不改。缓存的是可选实现函数或不可用结果，不缓存dbo_enabled状态；每次调用仍检查当前线程/ubatch状态，NPU启用优先，未启用时保留native CUDA fallback。进程运行期间安装/卸载后端需重启，不支持动态重探测。除ImportError外的异常保持抛出。

本轮以可重复补丁、CUDA/NPU guard回归、原源码哈希守恒和无profiler引擎A/B验证验收这一修复。其它既定profile/metadata任务不因局部修复而标完成。

计算后续核对保留为已有范围内的证据缺口，本轮不新增计算优化任务；仅实际覆盖冲突影响固定项唯一计费时修正。既有关注项包括：Attention core与KV写入的独立kernel开销（KV写入字节已计入，不能再重复加流量）、batch相关布局拷贝、融合norm/residual、专家路由布局与combine覆盖、层外GPU辅助工作。仅tiny模型的copy观测不直接推广成所有模型恒定3次拷贝。

通信后续必须表达每步metadata长度/内容两条控制消息、GPU/CPU回读和模型输入就绪依赖；send局部完成与接收数据可用不同，回执RTT仅作诊断。host提交、阻塞等待和fabric服务各自归属，不把F端等待与A端准备相加。当前profiler时长不得直接作为纯服务profile；实际路由回放、接收就绪和host覆盖未完成时仍明确未校准。

- 具体 host 项能否用常数拟合、metadata 消息数量及等待内核占用对计算的影响，由同窗口测量决定；在获得证据之前不发布具体参数或准确性结论。这些是采集结果，不阻碍当前协议与计时边界的设计。


### A输入准备续作（用户确认继续完整校准）

系统 SHALL 允许a_input_prepare完整操作：计时前排除前序设备工作，覆盖原生_prepare_inputs的CPU组织、block table/输入H2D、position/seq-len/slot mapping及自身完成；不包含codec、模型层计算或NCCL等待。接入 MUST 替代同覆盖a_step_prepare_us且拒绝旧host profile叠加。首版仅对经审计不含上述输入准备设备项的analytical_only计算路径启用；其他provider在覆盖未证明互斥时拒绝。每A worker每步唯一计费，约束调用线程并完成后才启动模型与控制操作，不独占GPU/fabric。同步补测扰动与正式无hook整步验收 MUST 分开；batch4不得用于拟合。


A/F准备补测同时覆盖a_attention_metadata与f_dp_metadata。前者为原生attention metadata构建（包含必要自身设备完成，不含控制发送），位于输入准备后/控制编码前；后者为F的_make_ffn_dp_metadata纯本地整理，位于控制解析后/F data接收前，MUST NOT 提前到step入口。两项按三窗口实测固定操作唯一计费，拒绝与笼统A/F step准备叠加；计费资源为对应调用线程，F数据依赖 MUST 指向metadata完成事件。样本与无hook整步时延分别报告。

## 2026-10-12 Current hierarchical runtime integration

用户授权覆盖历史“层级实现尚未实施/不适配”的范围限制。tiny驱动不得继承DSV3 dedicated跨节点布局：显式node_size=8/node_count=1（逻辑节点库存，仅使用GPU2/3，不表示实机存在八张卡），A位于GPU2，F位于GPU3；本地带宽300GB/s及alpha=1µs来自原NVLink3目录假设，不能标为新实测。只用local域FIFO，通信等待仍动态产生。host/operation条目的边界、覆盖、原生软件和数值不变方可派生，保留父profile指纹。runtime指纹包含全部通信域配置与MODEL_VERSION，拒绝旧指纹直接加载。新bundle冻结源码、数据目录、配置、profile和新batch5；batch1/8复验，4/2/3回归，5至少三独立无探针窗口并单独采路由。完整分项仍按4.4/4.5/5.2证据审核，整步通过不能关闭未测项。

实施细化：host/operation profile新增schema_version=3，仅改变留出代际（预留batch5），不改变v1/v2的batch4/batch2规则或参数计费语义。源码及data快照保存在独立ZIP；测试依赖util和benchmark另存补充哈希清单，二者联合恢复可运行冻结副本，原bundle不被覆盖。
