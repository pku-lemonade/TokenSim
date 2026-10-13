## 1. 证据与执行契约

- [x] 1.1 保存当前模拟/原生 baseline、源码哈希、实际配置及待测项，核对 integrate-deepseek-v3-core-runtime 与 hierarchical delta；以可复现 manifest 和规范交接清单验收，不覆盖用户已有实验。
- [x] 1.2 审计插件逐层 data/metadata 消息、F combine、grouped 调用、接收预贴和同步阻塞边界；以带源码定位的调用与计费覆盖表验收，明确 Q/K norm、embedding/logits 和采样归属。
- [x] 1.3 定义新协议与 profile 配置、适用范围及冲突校验；以合法1A1F加载、非法布局/dtype/shared expert拒绝、旧配置默认行为和重复成本拒绝测试验收。

## 2. 原生协议与算子边界

- [x] 2.1 实现按协议选择 F combine 和完整 hidden 返回，保留旧 partial-return；以 T=8/H=256/BF16 每层每方向4096 bytes、top_k=1/2不放大返回及数学加权结果 oracle 验收。
- [x] 2.2 同步终端节点、trace不变量、ownership/显存buffer和取消清理；以两层完整decode依赖、A residual恰好一次、KV不移交F、失败后资源可继续使用测试验收。
- [x] 2.3 对齐 grouped/fused expert 服务覆盖、Q/K norm 与模型边界计算；以一次实际提交不按专家重复计费、combine不重复、必要算子覆盖表和缺失scope明确失败验收。
- [x] 2.4 采集并回放 native step/layer/token 路由，按实际消息表达metadata；以 replay assignments/histogram一致、消息shape/bytes/次数与采集摘要一致验收，缺失项不得标为已对齐。

## 3. 分离时间线与缓存

- [x] 3.1 核对并按证据修正双端提交、数据可用和接收调用顺序；统一导出有观察端/起止边界的通信总等待，等待细分仅可选诊断。以20/100/120/5得到等待100、总通信105，以及链路80时等待80、缓存/诊断开关等价验收；不无依据串行化异步send/recv。
- [x] 3.2 核对 host step/compute submit/message submit/completion 的资源与调用依赖，优先验证控制长度/内容CPU回读；以step成本一次、合法重叠、证据支持的同步阻塞、等待不预留fabric及固定3/等待7/传输5不重复计费验收。
- [x] 3.3 增加scope与操作覆盖校验，剥离重复host计费并保留设备固有启动成本；以含等待测量、混合scope、默认值与profile叠加、负数/NaN拒绝测试验收。
- [x] 3.4 扩展静态计划/纯service缓存键与profile内容指纹；以协议切换失效、相同service5的两条竞争消息queue0/5、缓存/trace开关时间线一致测试验收。
- [x] 3.5 导出消息关联、host/peer/fabric边界和关键路径摘要；以并行事件不重复求和、makespan可重建、未解释区间明确标识验收。

## 4. 原生采集与校准

- [x] 4.1 隔离部署轻量采集，保存原生源码哈希和运行命令，增加GPU空闲保护；以不修改安装中插件业务逻辑、不干扰外部进程及只回传小摘要验收。
- [x] 4.2 补齐A侧未匹配kernel与同窗口CPU/GPU关联，核对跨进程时钟和消息identity；以覆盖统计、时间线归因与profiler扰动报告验收，保留无法解释部分。
- [x] 4.3 执行同款PyNccl双向data与torch.distributed NCCL metadata微基准及延迟接收对照；以明确ready/计时边界、样本分布和不随peer wait增长的纯service证据验收，含等待数据不导入。
- [x] 4.4 【主线】盘点并复用合格实测固定开销，针对性补测decode host及对应设备服务缺项；batch1/8各三独立窗口，优先固定常数。交付参数数值/单位、覆盖、每步/每调用计数、来源及统计/残差；拒绝含未知外部等待的inclusive区间；允许经ready/完成边界审计的operation_service，不使用batch4拟合，不以等待细分为前置条件。
- [x] 4.5 【主线】生成并实际加载实测固定成本profile至通用engine/ModelAFExecutor；host/device/transport/operation_service按scope分别接入，优先将既有staging/两种读回原样本形成完整操作参数。核对物化数值与trace计费次数，保存同配置真实路由下代入前后对照；接口存在、默认零值或仅报告加常数均不算完成。保留指纹/范围/冲突拒绝和部分校准缺口标记。

## 5. 对照验收与交付

- [x] 5.1 冻结参数后运行batch1/8及留出batch4，每点至少三组独立无profiler计时窗口；以预热/样本统计、数据路径/路由一致和prefill/decode分开报告验收。
- [x] 5.2 并列原始baseline、协议对齐未拟合与校准后结果；以A/F计算、双向transport、host服务及完整step的同scope倍率和max/min<10目标验收，等待使用零竞争/注入延迟oracle，未验证项明确保留。
- [x] 5.3 运行旧协议、AF-disabled、标准受支持prefill/decode和模型边界回归，评估事件数/wall/内存增量；以相关测试通过及性能报告验收，不为提速丢弃依赖或排队事件。
- [x] 5.4 更新使用/迁移说明和PKUa100报告，核对每个spec场景证据并运行 `openspec validate align-vllm-afd-protocol-and-timing --strict`；以支持域、未解决误差、复现命令、任务状态与证据对应验收，不宣称DSV3或层级通信重建已验证。

## 7. 已确认DBO问题修复（2026-10-07用户授权）

- [x] 7.1 提供隔离插件补丁，缓存可选NPU导入成功/不可用状态，保留动态enabled和fallback；验证缺依赖、NPU开/关、CUDA开/关、非ImportError传播及补丁输入漂移拒绝。
- [x] 7.2 在隔离服务器副本执行CPU guard和GPU2/3 batch1/8/4验证，保存前后哈希、输出一致性及无profiler耗时；原插件不变，不将性能改善归因为单一因素或宣称完整校准通过。

## 8. 固定host成本与因果等待首版

- [x] 8.1 更新固定成本约束并在通用executor/ModelAFExecutor实现独立A/F每步准备、额外compute节点提交和双端消息提交，复用host资源，保留默认兼容及未校准标记。
- [x] 8.2 验证100/200/5µs重叠oracle、每步唯一计费、shared资源竞争、完整模型、trace/cache等价、取消清理、非法配置和相关回归，补充使用说明。完整metadata/原生校准任务仍保留未完成。

## 9. 完整操作固定成本接入（本次用户确认）

- [x] 9.1 实现独立operation_service_profile及首批三操作白名单、边界/覆盖/范围/来源校验；新旧profile与显式手工冲突拒绝，codec可共存，旧配置与缓存隔离回归。
- [x] 9.2 以完整操作事件替代对应设备聚合事件，约束调用线程及必要stream依赖；验证100+11→111、外部延迟不改常数、不重复子项、其它GPU/fabric不被无依据锁定、trace/cache等价及取消释放。
- [x] 9.3 审核并复用已有三窗口staging/读回样本，按声明统计规则拟合并实际加载；逐步核对编码/解析独立计费、operation事件次数和物化来源，报告batch1/8同配置真实路由前后变化及近似限制。与4.4/4.5及5.1/5.2衔接，保留未测大项，不将局部接入标为整体验收。

## 本次修订的实施优先级

9.1–9.3及10.1/10.2已完成。当前优先推进11.1–11.3，补齐采样结果CPU可用边界，按真实调用补测并代入计算/控制提交与外围固定成本；随后11.4执行同路由消融和新冻结验收。3.1/3.2只补充实际影响完成时刻的依赖，等待细分不作为前置。旧5.1及旧冻结bundle保留为历史通过记录，新参数由11.4另验收；4.4/4.5/5.2不能因局部接口、诊断或整步通过自动勾选。不扩展GPU算子优化、层级通信或prefill范围。

## 11. 三类剩余固定成本修正（2026-10-08用户授权）

- [x] 11.1 在通用ModelAFExecutor/标准engine原生对照路径补齐sampled_token_readback及CPU token可用/输出整理完成依赖；使用ready输入补测完整操作常数并实际接入profile。以GPU100/回读11→CPU111、输入推迟20常数不变、下一步不提前、每步唯一计费、旧配置兼容及取消清理验收；不得用sample_tokens整段或通信host_completion替代。
- [x] 11.2 审核并补测A/F原生计算调用组CPU提交，batch1/8各三窗口常数优先，记录调用数、覆盖及设备启动互斥；接入按调用组profile映射而非统一乘所有DAG节点。以grouped多专家只计一次、与旧显式compute_submit冲突拒绝、合法CPU/GPU重叠和trace物化实测数值验收。
- [x] 11.3 分backend/方向补测并接入控制/data提交以及每步引擎调度、输出CPU整理，替代对应默认假设，复用已有host资源/profile；排除work.wait外部等待和已有准备/codec/readback覆盖。以三窗口统计、同覆盖拒绝、每步计数、缺匹配不伪装已测及cache/trace等价验收。
- [x] 11.4 新增三组分别启用/全部启用的同路由消融，保存新profile和冻结bundle；batch1/8、batch4回归及新增batch2留出各三独立无profiler窗口，诊断与计时分开。报告各组关键路径影响、整步及同scope分量倍率和仍未测项；验证范围/指纹/取消及相关回归，并strict validate。不改写旧验收或用新整步差额填数。

本次仅补充规划，三组影响大小待测，未新增固定参数或代码完成勾选。当前24/34任务完成；11.x细化并承接3.x/4.x/5.2的剩余证据，不替代原任务完整验收。诊断来源：docs/plans/afd-cpu-attribution-20261008.md。

## 实施记录

2026-10-09 调用组续作：11.3完成，26/34。新增runner bookkeeping六窗口1200样本，扣除完整嵌套_to_list后41.703833µs，接在CPU token ready之后、frontend输出之前；与已测六类消息提交/调度/输出一起实际加载，逐步核对计数、数值与覆盖。计算CPU另完成十二个role-window、14400函数样本，扣除CUDA API并集得到候选profile，F整块只收费一次；探针完整扰动仍在验证，11.2暂不提前勾选。三组同路由消融已保存，冻结后原生batch1/8/4/2三窗口验收进行中。315项AF回归通过，新增cache/trace和设备服务不变专项通过。详见docs/plans/afd-fixed-cost-candidate-20261009.md；没有将完整校准标为通过。

本次用户确认新增完整操作服务：排除外部等待后允许CPU/runtime/自身设备与内部完成同步合并计费，优先复用staging/读回样本，不再强制纯host/device拆分。新增9.1–9.3实施与验收任务；旧历史测量结论保留，不改原样本标签。本次仅修订六份规划文档，没有修改代码或加载新参数，当前16/28完成。

2026-10-07 空闲自动触发：GPU2/3空闲后执行三进程窗口的控制staging/长度及内容读回微基准，batch1/8每项每窗200样本，保留wall/thread CPU边界。前序GPU工作排除后wall约37.5–39.0/10.7–11.3/19.0–20.0µs，但仍为CPU/runtime/自身设备工作的混合区间，不导入纯host/device profile。监控已暂停，未使用batch4、未覆盖实验。详见docs/plans/afd-control-staging-20261007.md。仍16/25。

2026-10-07 统一等待续作：新增单观察窗口区间并集分解及按iteration消息汇总，sender/receiver观察不可相加为墙钟，缺接收事件不冒充零等待，未拆分的设备启动仍明确保留在provider服务。20/100/120/5及固定3/等待7/传输5等oracle通过；真实batch1/8各372窗口重建且makespan逐值不变，AF相关277项通过。GPU2约21GB/36%被占用，未启动原生双卡作业；固定成本本轮没有新增参数。3.1/3.2及完整profile仍待原生边界证据，保持16/25。见docs/plans/afd-unified-wait-20261007.md。

2026-10-07 实测参数局部接入：在PKUa100六个独立CPU进程窗口采集1488个原生codec样本，无CUDA初始化；编码6.163µs、解析14.9714167µs已形成版本化profile并实际加载通用ModelAFExecutor。batch1/8真实路由31步分别核对两项各31次唯一计费，模拟229.596/231.263→235.759/237.426µs。修复profile事件实测来源标记并输出逐项生效值，默认零值保持未测。AF相关274项通过、strict validate通过。此为CPU函数微基准的局部参数接入，其余host/device/transport成本、总等待新口径及原生边界仍未完成，4.4/4.5不提前勾选，保持16/25。见docs/plans/afd-applied-fixed-costs-20261007.md。

2026-10-07 用户确认修订：重点为代入测试得到的固定开销。通信统一为等待＋固定开销＋传输，重叠等待只计一次，等待细分类可选；真实数据就绪与必要同步依赖按证据修正。当前仍无已发布并加载的合格固定成本profile，本次只更新规划文档，不改变实现完成勾选。

2026-10-07 实机继续：GPU2/3空闲后完成batch1/8各三个独立无profiler窗口，输出重复一致100%；完成两组31步真实路由采集/回放，逐一核对各62个expert histogram及hidden buffer大小；完成PyNccl BF16与torch.distributed长度/JSON双向0/5/20ms延迟接收诊断。真实step3.36–3.88ms，回放模拟约0.23ms，倍率14.64–16.79，仍未通过整步数量级目标。局部CUDA事件不等于纯服务，未导入profile；metadata实际调用/消息边界及可归因host样本仍缺失，保留16/25，不用部分证据勾选整个任务。batch4仍留出未执行，全部实验进程已退出。见docs/plans/afd-live-results-20261007.md。

2026-10-07 本地工作收口：2.2/2.3/4.1/5.4完成，总进度16/25。完成逻辑协议buffer预留和KV容量扣减、fused expert包含combine时的唯一计费、真实A行身份/F top-k诊断hook、严格转换、非负host拟合、留出冻结执行器、双backend正确dtype/bytes延迟接收微基准及有限正值验收报告。远端新隔离目录已准备12个窗口/诊断case，仅dry-run及编译，没有GPU作业；原始plugin哈希不变。GPU2仍由PID3337486占用约37.6GB。全量487项通过，最终AF相关273项复验通过，79个spec场景证据映射已保存。剩余9项依赖原生采集/实测，具体缺口见docs/plans/afd-local-completion-20261007.md，不宣称纯通信、精确调用时间线或整步校准完成。

2026-10-07 回归收尾：5.3完成，总进度12/25。沙箱临时目录权限导致的36项错误经同一全量测试在沙箱外复核消除，479项通过；随后新增native边界失败恢复/KV归属用例及最后scope/trace改动的专项复验通过。31步batch8本地CPU基准，控制开启后trace关闭wall中位数201→251ms，trace开启285→331ms；开启trace的事件868→1395、Python峰值6.79→8.34MB，关闭trace不改变模拟时间线。见afd-local-overhead-20261007.json及host-profile/replay说明。本轮未启动远端GPU作业。

2026-10-07 本地续作：1.3/3.3/3.4完成，总进度11/25。host profile（固定/非负规模项、范围/来源/分位数/留出及冲突校验）接入标准engine和ModelAFExecutor；补齐Qwen3标准入口完整模型边界；加入带来源指纹的严格路由回放与CLI加载；实测compute scope及grouped覆盖检查、profile缓存隔离完成。2.4仍缺真实路由采集，4.5仍缺实测profile，其他完整原生验收不提前勾选。说明和证据见docs/plans/afd-host-profile-and-replay-20261007.md。

2026-10-07 控制协议续作：源码审计与原函数CPU-only wire oracle完成，控制消息/回读/解析依赖和backend缓存隔离接入，新增按iteration索引的关键链摘要。1.2/3.5完成，总进度8/25。2.4仅metadata部分完成，真实路由回放仍缺；3.1/3.2的精确原生调用边界与校准保持待验收。GPU2被外部进程占用（约70.9GB），未启动双卡作业。见docs/plans/afd-native-control-20261007.md及afd-native-source-audit-20261007.json。

2026-10-07 固定host首版：8.1/8.2完成，新增9项因果测试，AF相关236项测试通过；本地完整模型驱动31步通过，文档见docs/plans/afd-fixed-host-costs-20261007.md。总进度6/25；固定参数仍为未校准假设，原生metadata/精确调用回放/profile任务不标完成。

2026-10-07 DBO修复：隔离副本仅dbo.py改变，原插件哈希不变；12项专项回归通过，GPU2/3 batch1/4/8输出全部一致。CPU guard中位数164.73→0.17µs，真实decode步分别3.680/3.210/3.253ms。不同时间窗口的整步差异不全部归因于补丁；模拟约0.19ms，整步数量级仍未通过。新增7.1/7.2完成，总进度4/23；完整证据与剩余建模问题见 docs/plans/afd-dbo-fix-20261007.md。

2026-10-06：用户确认只校准decode，完整流程对照暂缓。协议配置、F combine/hidden返回、Q/K norm与专家纯服务组合、readiness事件接口及7项专项测试已实现；相关任务中的profile、原生回放或完整验收未完成时仍保持未勾选。详见 docs/plans/afd-native-protocol-implementation-20261006.md。GPU2/3被外部任务占用，尚未执行本轮原生校准，不能发布实测profile或声明数量级通过。

2026-10-06 后续：获用户授权，已在GPU0/3完成小Qwen3模型的decode计算专项对照，以及两张卡各3窗口的无AFD通信单卡控制。A/F聚合计算的实测/模拟倍率1.43–2.03，通过本次数量级检查；补齐slot mapping kernel后未归属kernel为0，原生跨模式输出一致。未拟合参数，未验证通信、host、路由回放或整步；4.2/4.4/5.1/5.2等完整验收仍未完成，保持未勾选。证据见 docs/plans/afd-compute-alignment-20261006.md 及同名JSON。

2026-10-07：GPU2/3空闲后完成PyNccl双向数据/32-byte回执微基准及5/20ms延迟接收对照；发送端局部完成不能证明接收端数据可用，回执RTT约22µs，未作为单向service导入。用户确认后另完成GPU2/3真实AFD batch1/8/4复测：无profiler完整decode步3.71–3.86ms，模拟约0.19ms，差约20倍，整步数量级尚未通过。原生跨GPU组合输出一致、31个decode步kernel全部归属；每点只有一个独立无profiler窗口。metadata/host/ready回放、单向服务profile和完整验收仍缺失，不新增完成勾选。证据见 docs/plans/nccl-gpu23-20261007.md 与 docs/plans/afd-gpu23-rerun-20261007.md。

2026-10-07 完整操作接入：9.1–9.3完成，19/28。staging38.1689167/长度读回10.946/内容读回19.3659167µs实际加载；batch1/8真实路由各31步逐项唯一计费，整步284.873446/286.540515µs。实机60次完成边界检查通过，operation专项10项通过；285项AF回归通过后补充两项测试。旧混合scope原样本保留，完整校准仍未通过，batch4未运行。见docs/plans/afd-operation-applied-20261007.md。
2026-10-07 本轮收尾：51模块逐模块进程回归共504项全部通过，退出码全部0；operation专项10项通过，三窗口原样本重新拟合与profile完全一致。单进程全量未产生完成摘要，不计为通过。仍19/28，实机边界检查已退出。

## 10. A/F准备操作补测

2026-10-09 继续实施：11.1完成，25/34。补齐回读六窗口1200样本，56.342µs实际加载并审核31步/点真实路由；另完成CPU调度/部分输出六窗口和双后端CPU提交六窗口，加入实测profile并替代发送5µs/接收0µs假设。当前batch1/8模拟整步1961.782/1963.058µs，未做新正式验收。310项AF回归通过。计算提交启动覆盖、runner bookkeeping、精确preposted依赖及新冻结/留出仍缺，11.2–11.4不提前勾选。详见docs/plans/afd-overhead-applied-20261009.md。

2026-10-08 固定成本续作：已实现sampled_token_readback、原生计算调用组/分backend方向host profile v2、每步调度及输出依赖；305项AF回归及5项新增专项通过，strict validate通过。完成回读batch1-w0/batch8-w0/batch1-w1各200样本，后续窗口因GPU2被其他用户占用、A worker初始化KV cache显存不足退出。部分数据未拟合为正式参数；计算/消息提交及引擎外围仍待测，新消融/留出未执行，11.1–11.4不勾选，保持24/34。详见docs/plans/afd-fixed-overhead-progress-20261008.md。

- [x] 10.1 完成A输入准备ready到设备完成的三窗口补测、覆盖审计、独立operation接入与冲突/依赖/取消测试；保留同步扰动和其他未测项，不把局部通过等同整步通过。
- [x] 10.2 补测并接入A attention metadata与F DP metadata，核对真实调用顺序、重复计费拒绝和输出一致性。

2026-10-07 A/F准备续作：10.1/10.2完成，21/30。A输入准备459.122833、A metadata157.488833、F DP metadata34.242083µs各1200样本并实际加载；真实路由31步每项唯一计费，模拟934.753625/936.029087µs。新无hook窗口和冻结后batch4验收正在进行，整体校准不提前标通过。见docs/plans/afd-preparation-calibration-20261007.md。

2026-10-07 冻结验收与通信诊断：2.4/4.3/5.1完成，24/30。batch1/8/4各三独立无hook/无profiler窗口共41633步，整步实测/模拟3.117/3.473/3.461倍，batch4未拟合；真实逐行路由、62个histogram/点及wire合同复核通过。双backend×三窗口×双向×两个payload×0/5/20ms延迟的配对设备测量与无profiler交换对照各3600样本，device service约8–9µs且不随peer延迟增长，模型匹配case差1.624–1.830倍。CUPTI设备区间仅作诊断，含CPU提交的外层区间不导入transport profile。292项AF回归及新增通信解析3项通过。六项完整操作和两项codec已加载；3.1/3.2/4.2/4.4/4.5/5.2仍缺原生精确API/ready同窗口关联和其余host成本，不能以整步通过将其勾选。详见docs/plans/afd-preparation-calibration-20261007.md及三个最新审计JSON。GPU任务均退出，监控保持暂停。

2026-10-10 结果收口（实验产物沿用20261009编号）：11.2/11.4完成，总进度28/34。计算原生组CPU扣CUDA API并集候选实际加载，F整块每层唯一计费；设备估值不变，新增cache/trace等价检查通过。十二个无探针窗口共63829步，batch1/8/4/2实测均值2633.024/2768.333/2728.265/2737.622µs，模拟4259.360/4261.027/4260.313/4259.825µs，所有窗口整步数量级通过。batch2在冻结后运行，62层路由直方图及消息大小核对通过。无CUPTI同边界诊断已完成，报告保留探针/窗口波动及调用前同步扰动，没有利用留出调参。315项AF回归及新增专项通过，strict validate通过。3.1/3.2/4.2/4.4/4.5/5.2仍缺精确原生调用顺序、API成本及同窗口分量独立验收，完整校准不标通过。见docs/plans/afd-fixed-cost-candidate-20261009.md及同目录验证JSON。

2026-10-10 原生调用顺序续作：3.1/3.2/4.2完成，31/34。新增显式native_call_order_enabled及新runtime指纹，接收预贴由本端调用依赖驱动、CPU提交与数据完成分开，旧模式保留；失败/取消释放全部预贴资源。331项AF回归通过。已有DBO修复后CPU/GPU trace按correlation、external id与Triton符号完成归属，A/F共同时间基准和prefill锚点配对，保留16个边界未配对调用及profiler扰动。新模式四点真实路由回放约4.218–4.220ms，固定参数不变。补充batch3冻结后启动被GPU2外部占用拦截，未启动GPU子进程。4.4/4.5/5.2不勾选。继续入口与证据见docs/plans/afd-native-order-progress-20261010.md。

## 12. 当前层级通信适配（2026-10-12用户授权）

- [x] 12.1 审核host/operation覆盖，修正tiny同节点GPU2/3摆放及通信域指纹；验证布局/带宽变更拒绝旧profile、实际计费及FIFO依赖。
- [x] 12.2 保留旧证据，派生新profile及完整源码快照冻结bundle，预留未见batch5，扩展collector并执行必要回归。
- [x] 12.3 冻结后batch1/8、历史4/2/3及新5各三无探针窗口，独立采集batch5路由并回放；报告同scope分项及整步、哈希和参数计数，strict validate，缺口不冒充通过。

2026-10-12 当前层级通信重新冻结验收完成：12.1–12.3完成，34/37。六个batch各三独立无探针窗口与新batch5路由回放通过；整步模拟/实测均值1.34–1.45倍，留出batch5为1.444倍。655项主源码和4项支持文件哈希不变，25项参数物化/计数及本地通信域核对通过；411项相关回归通过。4.4/4.5/5.2仍保留分项独立校准与transport profile缺口，不宣称完整校准完成。详见docs/plans/afd-hierarchical-refreeze-20261012.md及同目录验收JSON。

### 2026-10-12 剩余三项完成证据

- 4.4：`docs/plans/afd-complete-parameter-inventory-20261012.json`列出18项host与7项operation的数值、单位、覆盖、计数、统计与来源；设备固定服务仅由batch1/8 GPU区间拟合，transport仅由双端就绪后的设备区间拟合。API边界、原样本来源和独立batch6验证分别保存在complete-submission-fit、api-boundary-audit-final、complete-components-independent及transport-independent报告。
- 4.5：同配置真实路由的unfitted-b1/8/4与frozen-replay/b1/8/4位于`outputs/afd-complete-calibration-20261012`；逐步检查18项host、7项operation和6条消息实际计费，通用执行器覆盖/来源互斥回归通过。新源码及四类参数冻结于`afd-complete-frozen-bundle-20261012.json`。
- 5.2：`docs/plans/afd-complete-acceptance-20261012.json`与`afd-complete-results-20261012.md`汇总12个无探针窗口、冻结后batch7三窗口GPU分项、25项host/operation及36组transport独立验证；各同scope达到既定数量级目标，残余误差及聚合近似明确保留。原始baseline、协议对齐未拟合和校准后并列，源码/参数哈希与计费次数已审核。
