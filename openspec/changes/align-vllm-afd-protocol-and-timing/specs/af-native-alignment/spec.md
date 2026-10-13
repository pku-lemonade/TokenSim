## Purpose

定义 TokenSim 与真实 vLLM+AFD 的可复现对照及校准契约，使计算、通信纯服务、引擎开销和等待在一致边界下可比较，并保存测量来源、适用范围和留出验证证据，避免用端到端差值污染网络或算子估值。

## ADDED Requirements

### Requirement: Complete submission has independently scoped device service

系统 SHALL 允许v4 host_submission_service，仅在前序设备工作位于计时外、原函数哈希及API计数匹配、API白名单和D→D拷贝审计通过时使用完整CPU调用区间。未知/同步API和主机方向拷贝 MUST 拒绝。此服务 MUST 替换framework-only费用，且绑定独立GPU-only服务校准来源；GPU偏移 MUST 仅拟合分类GPU服务，不使用CPU/NCCL等待或整步残差。新v4使用batch7留出，旧profile语义保持不变。

#### Scenario: Old launch assumptions would overlap CPU APIs
- **WHEN** v4完整提交profile配合旧未分界的设备启动来源
- **THEN** 运行前拒绝，不叠加收费

#### Scenario: New unseen workload validates the completed costs
- **WHEN** 完整参数和源码冻结后运行batch7
- **THEN** 使用独立GPU分项及无探针整步窗口验收，保留聚合和测量扰动限制，不利用该点重新拟合

### Requirement: Remaining overhead calibration starts with measured constants

三类剩余开销（采样回读/完成、计算提交、控制提交与引擎外围）MUST 优先采用真实每步/每调用的实测固定常数。测量 MUST 使用现有vLLM+AFD执行路径并声明CPU调用/设备完成边界、输入ready、外部等待排除、覆盖及计数；batch1/8各三独立窗口，保存原样本及窗口中位数等权拟合规则、p50/p95和batch残差。自身设备工作不可分而已排除外部等待时 MAY 使用完整operation_service；不将其标为纯CPU busy。

cProfile SHALL 仅用于热点定位。轻量前向、sample_tokens、torch.send等inclusive wall区间在未排除外部等待和覆盖重叠前 MUST NOT 导入固定profile。恒定拟合不足且残差随batch稳定增长时 MAY 使用已有非负规模项，不依据整步差额调参。三组影响大小 MUST 由同配置路由的单组/全部启用对照及关键路径变化确定，不预设为小，也不把三个边际差额机械相加。

#### Scenario: An inclusive sampling interval contains prior device work
- **WHEN** sample_tokens wall包含等待尚未完成的前序GPU或NCCL工作
- **THEN** 该区间只保留为诊断；先排除前序工作再补测回读自身服务，原生依赖等待在模拟运行中产生

#### Scenario: New calibration preserves the earlier held-out result
- **WHEN** 10月8日补齐三组成本并产生新profile
- **THEN** 旧冻结bundle及batch4验收不改写；batch1/8用于拟合，batch4作为回归，新增batch2在新参数冻结后作为未参与本轮选参/拟合的留出点，各正式对照点至少三独立无profiler窗口

#### Scenario: Separate improvements overlap on the critical path
- **WHEN** 独立启用两组成本分别改变整步耗时，全部启用后两组存在重叠
- **THEN** 报告分别保留三组单独和全部启用的实际模拟结果，以事件关键路径验收，不要求单组增量之和等于全部增量

### Requirement: Optional DBO backends are resolved once in isolated reference fixes

参考插件修复 MUST 在隔离副本实施并保存补丁及前后哈希，原安装副本 MUST 保持不变。可选NPU DBO实现的成功或ImportError不可用结果 MUST 在初始化后复用，不在每层反复导入。系统 MUST 每次重新检查动态dbo_enabled状态，保留NPU启用优先与native fallback语义；其它异常 MUST 不被吞掉。修复后输出token MUST 与相同负载原基线一致，时间采样与原始profile MUST 分版本保存。

#### Scenario: NPU dependency is unavailable on CUDA
- **WHEN** 首次解析可选NPU实现发生ImportError，之后多次调用guard
- **THEN** 后续调用复用不可用结果且仍检查native DBO状态，不重复导入

#### Scenario: DBO state changes between calls
- **WHEN** 同一已解析实现的enabled状态从false变为true
- **THEN** 后一次调用执行对应yield，不因缓存而冻结enabled状态

#### Scenario: Optional implementation raises a non-import error
- **WHEN** 可选实现解析发生RuntimeError
- **THEN** 错误向调用者传播，不伪装成后端不可用

### Requirement: Native comparison preserves workload and provenance

对照系统 MUST 记录模型配置与权重标识、真实路由或明确的合成路由标记、输入输出长度、batch、prefill/decode、dtype、GPU UUID/拓扑、软件版本及插件源码哈希、connector、eager/graph、调度设置、预热与计时窗口。每条通信 MUST 能关联到 step/layer/direction/message identity 和实际 shape/bytes；data 和 metadata MUST 单列。未匹配 kernel 与未解释区间 MUST 明确报告，不得冒充零成本。

#### Scenario: Installed plugin contains patches
- **WHEN** 被测插件没有 Git 元数据且含本地修改
- **THEN** 报告保存关键源码哈希和配置，不标为原版上游版本

#### Scenario: Replay routing matches the measured layer
- **WHEN** 执行严格路由对齐对照
- **THEN** 模拟使用对应 step/layer/token 的 assignments 或可证明等价的执行工作量，缺失时标为未对齐

### Requirement: Calibration uses attributable service intervals

系统 MUST 将 step 调度、计算提交、通信提交、完成处理、GPU compute 与 transport service 分项校准；固定项与 batch/token/调用数相关项 MUST 根据重复测量选择。系统 MUST NOT 将跨窗口的整步减 GPU kernel 和、含 peer wait 的 NCCL occupancy、或两端时间相加作为纯服务测量。无法识别的外部重叠/等待 MUST 保留 unknown，不拟合为启动开销。排除外部等待、覆盖明确的CPU/runtime/自身设备完整操作 SHALL 允许以operation_service校准，不强制拆成纯host和device；内部完成同步是否包含 MUST 声明。transport 测量 MUST 使用同款后端、相同路径及明确双方 ready 条件，并记录剩余提交偏差和计时边界。

#### Scenario: Receiver intentionally arrives late
- **WHEN** 数据大小和路径不变，只推迟接收方 readiness
- **THEN** 增量归入依赖等待，transport service 校准值不随延迟增长

#### Scenario: Communication trace includes peer wait
- **WHEN** 候选 NCCL 测量含未分离的对端等待
- **THEN** 系统拒绝其进入纯传输 profile，只保留诊断记录

### Requirement: Measured fixed overhead is applied to the common runtime

本轮校准交付 MUST 包括测试得到的固定开销实际代入标准engine/ModelAFExecutor，不以已有配置接口或默认零值作为完成证据。系统 MUST 优先复用scope合格样本并针对性补测缺项；每项 MUST 保存实测数值、单位、覆盖、计数规则和三独立窗口统计，按host/device/transport或operation_service归属接入对应服务，不能把完整操作标为纯host/device忙时。常数 SHALL 为首选，增加规模项 MUST 有重复测量支持。等待细分诊断 MUST NOT 阻塞合格参数接入；含未知外部等待的区间 MUST NOT 直接代入固定成本。

#### Scenario: Measured parameters are actually used
- **WHEN** 合格实测profile加载至通用运行时
- **THEN** 物化参数和trace证明相应数值按每步/每调用规则唯一计费，并保存相同配置及路由的代入前后结果，而不是仅在最终报告加常数

#### Scenario: Only configuration support exists
- **WHEN** 固定开销接口已实现但未生成或加载实测参数
- **THEN** 固定成本校准保持未完成，输出未校准标记，不宣称测量已代入

#### Scenario: Some costs remain unmeasured
- **WHEN** 只有部分固定项拥有合格测量
- **THEN** 可接入这些项并明确列出剩余缺口，但不能将缺失值当作实测零值或宣称完整校准通过

### Requirement: Complete operation measurements exclude external waiting

系统 SHALL 允许将输入已就绪条件下测得的完整操作区间作为operation_service固定成本，包含声明的CPU调用、runtime、自身设备工作及自身完成同步。系统 MUST 保存ready前提、起止及完成边界、覆盖集合、资源近似、来源、范围、三窗口统计，并验证数据可用与API返回关系。历史mixed-scope样本 MUST 经边界审核生成独立产物，不修改原证据标签冒充纯服务。系统 MUST NOT 将整步差额、含NCCL外部等待的metadata整段或嵌套区间相加作为固定项。

#### Scenario: Existing ready-input readback measurements are reused
- **WHEN** 既有长度.item()三窗口样本确认前序GPU工作在计时外完成，计时末CPU已取得长度
- **THEN** 可从原样本拟合operation_service并接入长度读回，无需先拆CPU/runtime/设备时间；原host/device不适用标记保留

#### Scenario: External input is delayed
- **WHEN** 相同读回操作的输入可用时间被额外推迟20µs
- **THEN** 模拟操作常数保持不变，额外延迟由依赖产生；测量对照需验证边界和扰动，不能把20µs写进常数

#### Scenario: API return precedes required completion
- **WHEN** 采样只测到API返回而需要的数据尚未可用
- **THEN** 不将该样本当作完整操作完成成本；保留独立完成事件或补齐对应完成边界

### Requirement: Calibration profiles are scoped and non-overlapping

profile MUST 保存 schema/model 版本、硬件与软件指纹、原始摘要来源、timing_scope、单位、样本数、分位数、适用范围、拟合方法和验证结果。host cost MUST 按实际 step、执行调用或消息数计费，不能按每个逻辑专家重复收取 fused/grouped 调用开销。每项 kernel launch、通信 alpha、metadata、host submit/completion MUST 有唯一计费归属；显式 host submit 已计费时，算子或通信 service 不得再计同一段 host 服务。profile MUST NOT 存储运行态排队值用于预测。

#### Scenario: One grouped expert call covers multiple experts
- **WHEN** 一次原生 grouped expert 调用服务多个被选专家
- **THEN** 提交费用按实际调用及其内部kernel提交覆盖范围计算，工作量保留完整 assignments，融合包含的 combine 不再单独重复计费

#### Scenario: Profile fingerprint changes
- **WHEN** profile 内容、协议版本或被计费调用粒度改变
- **THEN** 新运行不复用旧语义的估值缓存，报告新指纹

### Requirement: Alignment acceptance uses held-out measurements

首轮 SHALL 使用 PKUa100 GPU 2/3 的同一 NV12 路径、tiny Qwen3 MoE、输入128/输出32、BF16 eager 1A1F。batch=1/8 用于校准，batch=4 MUST 在参数冻结后作为留出验证。本轮MUST仅验收decode；原生prefill只用于建立KV/预热，不代表已实现A本地prefill。profile MUST在查看batch4结果之前声明预测/插值规则和覆盖范围。每点 MUST 有预热和至少三组独立无 profiler 测量窗口，报告样本数及中位数/尾部统计；profiled 时间线另用于归因并评估扰动。系统 MUST 并列未校准、协议对齐但未拟合、校准后结果，分别比较 decode 的A/F compute、双向 transport、host service、operation_service与整步；正值指标的 max/min<10 为本轮数量级目标。零值等待以零竞争和人为注入延迟的因果测试验收，不使用除零倍率；无法可靠分离的环节标为未验证，不能宣称全部通过。

#### Scenario: Held-out batch misses target
- **WHEN** batch=4 有必要指标相差十倍及以上
- **THEN** 结果为未通过，保留原 profile 与证据；若利用该点重新拟合，必须另设未参与拟合的留出负载

#### Scenario: Server GPU is occupied
- **WHEN** 补测前发现目标 GPU 被其他任务占用
- **THEN** 不启动干扰性测量，不替换拓扑拼接基线，记录该项待测及原因


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
