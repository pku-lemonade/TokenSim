# A/F准备成本续作

用户要求继续做到结束后，本轮补测并实际接入三个准备操作，不使用整步残差拟合。原控制三项与codec保持独立。

| 新增操作 | µs/次 | 每步次数 | 位置 |
| --- | ---: | ---: | --- |
| a_input_prepare | 459.122833 | 1 | A输入就绪后、控制/模型之前 |
| a_attention_metadata | 157.488833 | 1 | 输入准备之后、控制编码之前 |
| f_dp_metadata | 34.242083 | 1 | F控制解析之后、activation接收之前 |

每项batch1/8各三独立进程窗口，每窗200样本，共1200样本；六组中位数等权均值。输入准备覆盖CPU整理、block/input拷贝、position/seq-len、slot mapping和logits indices，不含模型层或NCCL。A两项计时前synchronize排除前序GPU工作，计时末端synchronize包括自身完成；F项仅整理已解码的CPU metadata，不执行网络I/O或GPU工作。A显式完成同步会改变运行重叠，故这些窗口只用于操作服务，不作为无扰动整步验收。

输入准备六窗口中位数426–539µs（batch8最低421µs），完成尾段约21–26µs；固定常数是当前范围近似，不随等待变化。输入准备输出与既有原生无hook数据逐token完全一致。metadata窗口内输出重复一致100%。

## 证据与实现

新profile为 `afd-preparation-profile-20261007.json`，保留先前控制profile及中间输入准备profile。原始数据分别在 `outputs/afd-input-prepare-20261007-v2`、`outputs/afd-metadata-prepare-20261007-v1`；服务器profiling下同名目录。脚本 `collect_afd_prepare.py` 创建隔离插件副本，`afd_prepare_hooks.py` 使用真实runner输入测量；`fit_afd_input_prepare.py`、`fit_afd_metadata_prepare.py` 拒绝缺窗口、batch4、错误GPU/来源漂移、非法区间或含外部等待标记。

首个输入准备v1窗口已测完，但驱动仍要求已关闭的profiler文件而报错；修复验收断言后在全新v2目录完成六窗口，v1原样保留且未用于拟合。被测安装中插件没有改动。每个case启动前核对GPU2/3进程、显存/利用率和端口；只清理本case启动的子进程。

通用AFExecutor/ModelAFExecutor把A两项作为每worker每步顺序准备事件；F项从控制解析完成到data接收之间执行，并更新真实trace前驱。操作使用调用线程资源，不把完整wall记为GPU独占。显式A/F step准备或相同覆盖host profile叠加会拒绝；A完整准备仅启用在已审计未包含这些设备项的analytical_only路径，未知table覆盖拒绝，避免重复计费。未启用profile保持旧行为。

## 实际代入

真实路由31步、同codec和配置，batch1/8分别为934.753625/936.029087µs。对比仅控制完整操作时284.873446/286.540515µs。每步六项operation各一次，codec各一次；每组186个operation事件，duration与profile逐项一致，A准备及F解析后的依赖正确，无重复control_device。见 `afd-preparation-replay-audit-20261007.json`。

参数和默认假设整体冻结在 `afd-frozen-preparation-bundle-20261007.json`。冻结范围明确为partial_calibration_step_validation，未测的其他host dispatch/output及纯transport仍列缺口，不能将整步通过解释为所有分量通过。batch4不参与任何拟合。新无hook、无profiler窗口与路由诊断分开执行。

## 验证

operation专项15项通过，覆盖唯一计费、A/F正确先后、同覆盖冲突/未知provider拒绝、取消恢复、旧配置与缓存、拟合污染/留出拒绝。metadata接入后的AF回归292项通过；后续通信解析新增3项通过。OpenSpec strict validate通过。

## 冻结后整步验收

batch1/8/4各三个独立进程窗口，均无profiler、无测量hook，合计41633个decode步。prefill只建立KV与预热，不计入decode。参数在九窗口前冻结，batch4未用于拟合；相同batch跨窗口输出token哈希一致。正式时延与路由诊断分别采集，后者不用于拟合。

| Batch | 原生均值µs | 模拟均值µs | max/min |
| --- | ---: | ---: | ---: |
| 1 | 2913.757981 | 934.753625 | 3.117 |
| 8 | 3250.962233 | 936.029087 | 3.473 |
| 4，留出 | 3238.352699 | 935.539049 | 3.461 |

三个点及每个独立窗口的整步数量级检查均通过。证据为`afd-frozen-preparation-validation-20261007.json`；每窗口样本数、p50/p95和token哈希均保留。此次无hook原生时间比历史3.36–3.88ms低，实验时段及warmup profiler设置不同，不能归因于模拟器改动。

真实路由与wire审计为`afd-frozen-routing-audit-20261007.json`：各batch31步、62个expert histogram全部一致；每步四条hidden消息，每条batch×256×2字节，控制8+179字节。逐行assignments可由原始capture重新生成。metadata字节用原序列化函数交叉核对，调用次数来自源码循环与每侧32次codec实采（含prefill）；没有把这些证据写成真实API时间戳回放。

## 通信诊断收口

GPU2/3上，两种后端各三窗口，每个窗口两个payload、双方向及0/5/20ms接收延迟。无profiler与配对CUPTI两套各3600个保留样本，大型trace保留远端，只回传小JSON。

配对后以双方kernel较晚的开始时刻为设备ready，以双方较晚的完成时刻为服务结束，先启动kernel的peer等待不计入服务。PyNccl各窗口中位数8.153–9.293µs，torch NCCL为8.148–8.943µs；模拟对应14.981–14.995µs，匹配case倍率1.624–1.830。每种payload/方向跨0/5/20ms延迟的三窗口中位数最大变化约1.14%，不会随等待增长到毫秒量级。反向torch控制仅为微基准控制项，真实协议没有对应消息，不虚构模型倍率。

无profiler的发送API入口→预贴接收端设备完成，delay0中位区间端点范围为PyNccl56.398–158.110µs、torch78.969–187.760µs；它包含CPU提交，与8–9µs设备服务是不同口径。注入延迟主要出现在sidecar ready等待，仍可能影响唤醒/提交。时钟映射误差界保留在逐case摘要，禁止用外层区间拟合纯传输。

同case的profiled/unprofiled外层区间中点比约0.684–2.768，既有profiler扰动也有跨窗口调度变化，不能归因为精确profiler增量。因此配对设备测量用于设备服务诊断，不自动发布transport profile，不声称CPU提交固定成本已校准。证据及复现见`afd-preposted-service-20261007.json`、`scripts/report_afd_preposted_service.py`。

## 交付边界与未完成项

本次已实际接入六项完整操作与两项codec实测成本，完成冻结后的整步数量级验收。计算沿用既有A/F设备聚合对照（约1.43–2.03倍），这包含融合范围近似；历史A统计含slot mapping，当前slot mapping已归入输入准备，不能把历史A总量和新的准备成本相加构造“实测step”。

以下仍没有完整证据，保留OpenSpec未完成状态：原生逐消息CPU提交/接收预贴/数据就绪的同窗口关联；其他host dispatch、输出处理及完成成本；完整host/device/transport参数归属和全分量验收。约2ms整步差额仍未归因，绝不作为固定常数填入。配对微基准解决了设备服务含peer等待的问题，但不能替代原生引擎时间线。

本次结果仅覆盖tiny Qwen3、BF16 eager、1A1F、batch1–8、GPU2/3、decode；不验证DSV3、分离prefill、多节点或并发流。没有改动安装中的参考插件、没有利用batch4重新拟合。校准进程均已退出，GPU2/3末次检查均4MiB/0%、无计算进程；监控保持暂停。

复现审计：

```powershell
python -m scripts.audit_afd_frozen_replay --output outputs/new-routing-audit.json
python -m scripts.report_afd_frozen_preparation --output outputs/new-frozen-validation.json
python -m scripts.report_afd_preposted_service --output outputs/new-communication-report.json
```

以上脚本拒绝覆盖输出。远端测量分别为`collect_afd_prepare.py`、`collect_afd_frozen_validation.py`、`measure_afd_preposted_service.py`和`summarize_afd_preposted_kernels.py`；后两者的receiver预贴需要先完成双向warmup，避免lazy NCCL communicator初始化与CPU sidecar互等。
