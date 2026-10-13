## ADDED Requirements

### Requirement: Measured direct transport profile is explicit

系统 SHALL 支持transport_service_profile，限定明确的同节点原生直接消息。模型版本、环境、通信域、GPU摆放、backend、dtype及大小范围 MUST 匹配；只允许已声明的训练范围内插值，不得外推或静默解析回退。CPU提交和外部等待 MUST 排除，profile内容 MUST 进入运行时指纹和估值缓存；消息仍由运行态FIFO调度。

#### Scenario: Message is outside the measured support
- **WHEN** backend、方向、dtype或消息大小无匹配记录
- **THEN** 明确拒绝，不使用零费用或另一通道常数冒充实测

### Requirement: Native call ordering is explicitly versioned

`native_call_order_enabled` SHALL 为显式布尔开关，要求native control启用，默认关闭保留旧结果。开启时runtime指纹 MUST 改变，避免旧调用顺序的profile和路由文件未经审核直接加载。

#### Scenario: Old frozen configuration is replayed
- **WHEN** 未配置native_call_order_enabled
- **THEN** 使用历史调用顺序及原runtime指纹，不静默改变冻结结果

### Requirement: Remaining fixed cost profiles preserve explicit coverage

系统 SHALL 在既有host/operation profile机制中扩展采样回读、原生计算调用组、控制backend/方向及引擎每步整理的覆盖映射，不改变旧profile v1语义。sampled_token_readback MUST 明确要求原生decode、分离计时与输入ready/CPU结果可用边界；计算/消息条目 MUST 声明role、调用组或backend/方向、每步/每调用计数规则及被替代的旧成本。参数 MUST 有限非负，未知调用组、缺匹配项、范围或指纹不符及同覆盖重复配置 MUST 执行前拒绝；不得悄悄使用另一backend常数或零值冒充测量。

每组 SHALL 可以独立启用用于消融；物化输出 MUST 区分实测条目、旧假设、未测缺项和被替代费用。新增字段、调用绑定及覆盖版本 MUST 进入profile指纹和相关缓存键。旧配置未启用新条目 MUST 保持原时间线；显式手工同覆盖成本与新profile MUST 互斥，默认值不叠加收费。

#### Scenario: Measured control costs replace an assumed default
- **WHEN** profile匹配torch.distributed控制send，而旧host_submit_us仅为省略配置后的5µs默认值
- **THEN** 对应调用只计实测费用，物化结果标明替代来源；PyNccl调用必须使用其独立匹配项或明确保留未测假设

#### Scenario: Explicit generic compute cost conflicts with call group coverage
- **WHEN** 同一调用配置原生调用组实测提交费用及显式compute_submit_us
- **THEN** 执行前拒绝重叠覆盖，不把两项相加，也不自动按逻辑节点重复收取调用组费用

### Requirement: Native control configuration declares uncalibrated service boundaries

系统 SHALL 提供`communication_timing.native_control_enabled`（默认false），以及默认0的`control_encode_us/control_h2d_us/control_length_readback_us/control_payload_readback_us/control_decode_us`。控制开销 MUST 有限非负且拒绝bool，非零开销 MUST 要求控制模式开启，控制模式 MUST 要求原生协议与分离计时。新的native对照驱动 SHALL 默认开启控制模式，显式关闭时 MUST 标为disabled_baseline；开启只表示协议被建模，不能冒充已校准。

#### Scenario: Control costs cannot silently disappear
- **WHEN** 控制模式关闭而任一控制服务成本非零
- **THEN** 配置失败，不静默忽略费用

#### Scenario: Native comparison records assumptions
- **WHEN** 对照驱动启用控制模式但未加载实测profile
- **THEN** 输出modeled_uncalibrated、固定服务假设及data/control各自payload

### Requirement: Manual fixed host costs remain explicitly uncalibrated

系统 SHALL 提供`communication_timing.a_step_prepare_us/f_step_prepare_us/compute_submit_us/host_receive_submit_us`，默认0；非零项 MUST 要求原生协议和分离计时开启。各项 MUST 有限非负且不接受bool。输出 MUST 保留配置值、计费范围和未校准标记，不能因参数可配置就声称原生时间线已对齐。

#### Scenario: Invalid fixed cost is rejected
- **WHEN** 任一成本为负数、NaN、无穷大或bool
- **THEN** 配置在模拟开始前失败

#### Scenario: Zero defaults preserve previous runs
- **WHEN** 旧配置省略新增固定成本
- **THEN** 不添加新成本，不改变原有时间线

### Requirement: Native protocol selection is explicit and validated

系统 SHALL 提供 `af.execution_protocol=partial_return|vllm_afd`；未指定时保留现有 partial-return 路径及已有 router_location 行为。`vllm_afd` 首版 MUST 仅接受 1A1F、两域 TP=DP=EP=1、Router按被测插件固定在F、无 shared expert 的受支持 Qwen3 MoE、BF16 activation/transfer 与decode（af.phases 仅为 decode）。原生协议 MUST 自动选择被测Qwen3插件的F Router，不以旧router_location默认值选择A；系统 MUST 在执行前拒绝不支持组合；协议和版本 MUST 出现在物化配置及报告中。

#### Scenario: Explicit native alignment configuration
- **WHEN** 用户选中受支持的 `vllm_afd` 配置
- **THEN** 计划使用 F combine 与完整 hidden 返回，且报告明确记录协议

#### Scenario: Unsupported layout is rejected
- **WHEN** `vllm_afd` 配置多 F rank、shared expert 或不支持 dtype
- **THEN** 系统在事件循环前报告不支持原因，不静默回退 partial-return

#### Scenario: Existing configuration remains reproducible
- **WHEN** 旧配置未指定 execution_protocol
- **THEN** 继续使用既有 partial-return 语义，不自动移动 combine

### Requirement: Runtime output distinguishes measured and assumed fixed costs

校准配置的物化输出 MUST 标明各固定项实际生效数值、单位、来源profile及覆盖，区分实测值、手工假设和未测默认值。profile成功加载 MUST 能在通用运行时计费事件中核对；仅存在配置字段不构成实测接入。

#### Scenario: A zero default has no measurement source
- **WHEN** 某固定项仍使用默认0且没有实测来源
- **THEN** 报告标为未测假设，不解释为原生系统已测得零成本

### Requirement: Complete operation profiles preserve old scope semantics

系统 SHALL 新增显式版本化af.operation_service_profile，支持原生单in-flight decode控制staging、长度读回、内容读回及经覆盖审计的A输入准备，要求native control与分离计时开启。条目 MUST 声明operation_service、操作/覆盖、非负有限值与单位、ready及完成边界、资源模型、指纹、适用消息/工作量范围及测量来源统计。未知操作或scope、bool/负数/NaN、范围或指纹不匹配 MUST 拒绝。旧host profile v1及手工参数语义 MUST 保留，未启用新profile的行为不变。

同一覆盖的operation条目、其它profile、显式手工control_h2d_us/control_length_readback_us/control_payload_readback_us MUST 互斥；省略的默认零值不作为冲突或额外费用。scope及内容指纹 MUST 进入缓存隔离。报告 MUST 显示实际生效值、来源、覆盖、近似和验证状态，不能以完整操作测量推断纯CPU/GPU利用率。

#### Scenario: Legacy and complete operation coverage conflict
- **WHEN** 完整长度读回条目与显式手工control_length_readback_us同时配置
- **THEN** 执行前报告覆盖冲突，不叠加或静默选择其中一个

#### Scenario: Separate codec profile remains compatible
- **WHEN** host profile只覆盖encode/decode，而operation profile覆盖三个无重叠控制操作
- **THEN** 两个profile可同时加载，分别物化来源与费用

#### Scenario: Older profile remains reproducible
- **WHEN** 用户未配置operation_service_profile
- **THEN** 既有host/device scope规则及计费行为保持不变，混合区间仍不能导入旧host profile

### Requirement: Calibration configuration declares scope and coverage

系统 SHALL 支持显式选择版本化 engine overhead profile，也 SHALL 支持显式禁用 profile 以运行未校准基线。原生对照 MUST 显式启用 `af.communication_timing.enabled=true`；profile 与手工 host 成本对同一计费项 MUST 互斥，避免默认 host_submit_us 再叠加。超出 profile 的模型、设备、拓扑、软件版本、执行模式或工作量范围 MUST 拒绝校准运行；显式未校准运行 MUST 输出清晰标记。

#### Scenario: Conflicting charging sources fail early
- **WHEN** 同一消息提交费用同时指定 profile 和手工覆盖值
- **THEN** 系统报告重复计费配置，且不开始执行

#### Scenario: Profile is unavailable or incompatible
- **WHEN** 校准对照请求的 profile 缺失或仅适用于另一后端、GPU 拓扑或 graph 模式
- **THEN** 系统明确失败，不使用零开销或其他 profile 冒充校准值

#### Scenario: Baseline uses separated timing without fitted costs
- **WHEN** 对照显式选择未校准 baseline
- **THEN** 分离计时保持启用，各项假设成本和缺失测量出现在报告中


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
