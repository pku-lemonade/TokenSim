## Why

PKUa100 的 1A1F tiny Qwen3 MoE 对照中，计算初筛相差约 1.5–1.9 倍，完整 decode step 相差约 15–19 倍。当前模拟采用 A 侧 combine 和专家 partial 返回，本轮驱动未开启已有通信分离计时，且完整引擎的 host 开销尚未校准，必须先统一执行与测量口径，再判断模型准确性。

## What Changes

- 2026-10-12继续完成4.4/4.5/5.2：独立batch6验证既有固定成本，接入经边界审核的配对设备transport服务；补齐已审核非阻塞CUDA API的CPU提交服务，并绑定独立GPU-only固定服务校准以拒绝重复启动费用。保留旧profile，新v4在参数冻结后使用batch7；具体边界与统计见`docs/plans/afd-completion-contract-20261012.md`。

- 2026-10-08用户授权自行补齐三类固定成本：采样结果CPU回读/输出完成、计算提交、控制提交与引擎外围处理。优先按真实每步/每调用边界使用实测常数，复用现有host/operation profile、host资源与依赖机制，不扩展GPU算子算法或重建通信系统。三类成本的影响大小由代入前后对照确定，不预设为小或用整步差额证明其大小。
- 在原生对照模式下补齐token_selection之后的token CPU可用边界及下一步依赖；等待前序计算/通信的时间由事件产生，仅把输入就绪后的回读自身成本拟合为完整操作。计算提交按原生调用组、控制提交按实际backend/方向计费，外围调度和输出整理按每步计费；排除已有准备、codec、设备启动及嵌套覆盖。保持旧配置和历史冻结结果可复现。
- 10月8日CPU诊断仅用于定位，cProfile及含等待的轻量前向/采样区间不直接成为固定参数。完成三窗口补测和通用执行器接入后，分别启用三组成本作消融对照，报告各组对关键路径的实际影响；不将各组独立差额机械相加。

- 用户确认增加完整操作服务 `operation_service`：输入已就绪、外部依赖/排队已排除时，允许用一次完整操作的CPU调用、runtime、自身拷贝及其完成同步合计耗时作为固定成本，不强制先拆为纯host/device。首批接入控制staging、长度读回、内容读回；原codec参数继续独立计费。通过独立版本化profile声明覆盖与调用边界，不能冒充纯CPU/GPU忙时。
- 优先复用三窗口staging/回读数据，在核对输入就绪、返回/完成边界及覆盖后形成参数并实际代入；历史约38/11/20µs仅为取值线索，正式常数从原样本按声明规则汇总。继续针对A/F输入准备、计算/通信提交及输出处理补测，不把已有约653µs输入准备、1.6ms metadata inclusive区间直接相加。

- 本轮后续实施的重点是将测试得到的固定开销代入通用执行器：复用合格样本、补测缺项、生成并实际加载版本化profile，再冻结参数执行对照。仅完成配置接口、保留默认零值或只出测量报告均不构成校准完成。含未知外部等待的inclusive诊断不冒充合格固定成本；排除外部等待且覆盖明确的完整操作可用operation_service接入。
- 通信统计统一为“等待时间＋固定开销＋传输时间”，三个部分在同一声明的观察区间内互不重复。等待统一报告总量，重叠blocker按区间并集计算；peer/fabric细分类仅可选诊断，不作为新的必做子系统。只修正经核对确实缺失的调用、数据就绪和资源依赖；计算建模沿用已有对照，本轮不扩展计算优化。

- 2026-10-07用户确认先用固定host成本近似：A/F每步准备独立执行，计算与接收提交在对应依赖处计费，等待由事件及资源竞争产生。新增参数默认零且标为未校准；不以整步差值填充，不替代metadata网络消息或真实路由回放。
- 延续实施：显式模拟每步8-byte长度及原生JSON内容两条控制消息、H2D/长度回读/内容回读/解析，区分torch.distributed控制通道和PyNccl数据通道。增加按iteration索引生成的时间线摘要，重叠服务不相加为整步，缺失边界保留unexplained。

- 新增显式 `vllm_afd` 协议模式，首版限定已验证的 1A1F、TP=DP=EP=1、gate-on-F、tiny Qwen3 MoE 路径；F 完成专家计算和 weighted combine，返回每 token 一份 hidden，A 保留 residual 与请求/KV ownership。
- 按原生调用核对 grouped expert 粒度、路由、Q/K norm、消息数和 metadata；协议切换同时更新 DAG、payload、缓存键与 trace 校验。旧 partial-return 模式保留供历史实验复现，不静默改变其他布局。
- 对照配置强制显式启用已有通信分离计时，核对双端 ready、必要调用依赖与纯服务边界，统一导出通信总等待；等待由本次运行产生，缓存只保存静态计划及纯服务。
- 新增基于 vLLM+AFD 实测的版本化引擎开销 profile，分别描述 step 调度、计算提交、消息提交和完成处理；固定项和工作量相关项按证据确定，禁止用整步差额拟合一个总倍率或把等待拟合为启动成本。
- 在 PKUa100 以 batch=1/8 校准、batch=4 留出验证；分别报告 A/F 计算、双向通信、host 服务、等待和完整 decode。保留未校准基线，使用同款 PyNccl 补测纯传输。
- 2026-10-07用户授权修复已确认问题：在隔离AFD插件副本缓存可选NPU DBO实现的导入结果（含不可用状态），保留动态yield判断；验证CUDA/NPU行为、原插件哈希不变及真实引擎输出一致。该修复先于host校准，不把重复失败导入成本固化进profile。

## Capabilities

### New Capabilities

- `af-native-alignment`: 原生对照测量契约、版本化开销校准与数量级验收。

### Modified Capabilities

- `af-execution`: 按协议选择 F combine 或旧 A combine，明确原生对照的时间边界、双端就绪与缓存隔离。
- `af-separation-config`: 增加显式协议/profile 配置、支持域校验与分离计时要求。

## Impact

涉及 AF config、planning、executor/model_executor、operator_provider、trace、estimate cache、路由输入，以及本轮对照脚本与报告。数据采集保留服务器 trace，只回传摘要；不得干扰服务器其他 GPU 作业。

复用已实现的 `integrate-deepseek-v3-core-runtime`，其 delta 同步必须先于本变更。历史提案时`rebuild-hierarchical-af-communication`尚未实施；2026-10-12用户已授权适配当前实现并重新冻结验收：本变更不重复实现其拓扑、collective 算法和原子路径调度；两者都保留 service/queue 分离，后续合并不得覆盖本变更的协议选择和计时边界。新增协议默认不替换旧配置，但新的 vLLM 对照样例必须显式使用它。本次按用户要求进入实施；只校准decode，完整流程对照暂缓。


### A输入准备续作（用户确认继续完整校准）

系统 SHALL 允许a_input_prepare完整操作：计时前排除前序设备工作，覆盖原生_prepare_inputs的CPU组织、block table/输入H2D、position/seq-len/slot mapping及自身完成；不包含codec、模型层计算或NCCL等待。接入 MUST 替代同覆盖a_step_prepare_us且拒绝旧host profile叠加。首版仅对经审计不含上述输入准备设备项的analytical_only计算路径启用；其他provider在覆盖未证明互斥时拒绝。每A worker每步唯一计费，约束调用线程并完成后才启动模型与控制操作，不独占GPU/fabric。同步补测扰动与正式无hook整步验收 MUST 分开；batch4不得用于拟合。


A/F准备补测同时覆盖a_attention_metadata与f_dp_metadata。前者为原生attention metadata构建（包含必要自身设备完成，不含控制发送），位于输入准备后/控制编码前；后者为F的_make_ffn_dp_metadata纯本地整理，位于控制解析后/F data接收前，MUST NOT 提前到step入口。两项按三窗口实测固定操作唯一计费，拒绝与笼统A/F step准备叠加；计费资源为对应调用线程，F数据依赖 MUST 指向metadata完成事件。样本与无hook整步时延分别报告。

## 2026-10-12 Authorized scope update

本轮适配当前node-domain FIFO通信实现，固定tiny Qwen3、BF16 eager、同节点GPU2/3、1A1F decode；不扩展跨节点实机验收。审核旧host/operation覆盖后派生新profile，数值不按验收差额调整。显式同节点布局，通信域参数及模型版本进入指纹。旧batch4/2/3结果保留，新batch5在新bundle冻结后作为未见留出。
