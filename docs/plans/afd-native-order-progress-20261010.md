# 2026-10-10 原生调用顺序、关联审计与剩余验收

## 2026-10-12 v3采集完成及版本阻塞

GPU2/3两次空闲检查通过，端口无占用、无重复实验后启动全新v3。batch3三个无hook/无profiler窗口分别4371/5208/4340步，均值3300.468/2776.023/3327.492µs；共13919步，窗口等权均值3134.661µs，输出哈希一致。独立capture已完成，严格转换得到372条assignment（31步×2层×3token×top2）。远端实验已退出，端口已释放，监控保持PAUSED避免重复采集。

原生摘要中的冻结bundle哈希一致，但当前executor.py与cluster_config.py不匹配冻结源码哈希（排除LF/CRLF差异）。当前仓库含新提交4f42e71的层级通信实现；runtime指纹由97297e…变为3964c4…。使用旧冻结profile的本地回放被正确拒绝，未绕过校验或重写profile。独立输出目录中尝试逆向该提交的patch检查因与未提交校准改动重叠失败，未修改主工作区，也不将该副本作为冻结源码。

证据保存于`afd-native-order-supplemental-20261012.json`和`outputs/afd-native-order-validation-20261010-v3`。strict validate通过；本轮无实现改动，不重复全量回归。4.4/4.5/5.2仍未完成，31/34。下一步需要恢复经哈希验证的冻结源码，或经用户确认将当前层级通信实现的profile适配与重新冻结纳入本轮；不能简单替换runtime指纹宣称旧验收适用。新版本如需调整参数，不得将本次已观测batch3当作未见留出负载。

## 2026-10-12 补充验证重试

本轮两次启动前检查GPU2均为4MiB/0%、GPU3为92MiB/0%，均无计算进程；实验端口无监听，未发现本线程重复实验。使用新目录`profiling/afd-native-order-validation-20261010-v2`启动冻结batch3采集，成功启动后暂停自动监控。

首个窗口初始化期间GPU2出现外部PID2458147（TokenTown，58368MiB）。A进程报告`Available KV cache memory: -9.13 GiB`和`No available memory for the cache blocks`后退出；driver清理F进程。随后检查未发现本线程采集或worker进程。未进入正式计时，没有合格窗口、路由或新校准结果；不修改参数及31/34任务状态。

小日志保存于`outputs/afd-native-order-validation-20261010-v2/{attention,collection}.log`。保留远端v1/v2，不覆盖失败证据；恢复30分钟监控，下一次空闲后使用不存在的v3或更高目录。空闲检查无法预留资源，本次不通过修改显存参数绕过竞争。

本轮实现3.1/3.2，完成4.2关联审计，进度31/34。4.4/4.5/5.2的完整校准验收继续保留未完成。用户要求持续做到完成；当前远端补充验证被GPU2外部任务占用拦截，本轮没有启动GPU测试。

## 调用顺序修正

新增`af.communication_timing.native_call_order_enabled=true`，要求native control启用。默认false保留旧冻结实验及旧runtime指纹；true生成新指纹，派生profile必须显式复核和重新绑定。设备、模型、纯服务参数未变，不根据4.26ms整步差额改数。

1. F从iteration入口提交控制长度接收；控制payload接收等待长度CPU读回。
2. A提交完长度即可提交payload，不等待F回读；payload发送API返回可放行A的CPU计算提交，A设备执行仍等控制流完成。
3. F首层data接收等metadata完成，后续层接收等上一层F发送API返回。A本层返回接收等A本层发送API返回，无需再等待F计算。
4. transport必须等发送数据、双端提交和fabric可用。预贴期间不预占fabric，等待由事件产生并按观察窗口并集统计。
5. 控制估值失败、取消会结束全部预贴进程、释放host资源并允许下一轮运行。

证据：插件`connectors/metadata.py:376–420`为长度/内容send、recv和CPU读回顺序；`connectors/gpu/p2p.py:605–682`为PyNccl自定义收发及返回；`model_executor/models/deepseek_v2.py:135–153`为RemoteFFNProxy先send后recv；`v1/worker/ffn_model_runner.py:201–246`为F逐层recv/compute/send。既有5/20ms延迟对端实验支持API可先返回，不能把API返回解释成设备数据已到达。

调用组内部CPU与kernel交错仍采用聚合近似；本次不宣称逐kernel enqueue回放。现有GPU估值、完整操作覆盖和参数没有修改。

## CPU/GPU同窗口审计

复用PKUa100 `profiling/afd-dbo-fixed-20261007`原始trace，只在服务器读取大文件，取回小摘要。A/F使用相同Kineto baseTimeNanoseconds，按共同epoch和消息序列关联，不能用相同ProfilerStep编号配对。

|batch|A kernel归属|F kernel归属|两方向匹配消息|边界未配对调用|
|---:|---:|---:|---:|---:|
|1|1344/1344|640/640|120|16|
|8|1530/1530|704/704|120|16|

A每组有32个直接Triton `_compute_slot_mapping_kernel`，不存在torch CPU-op注解；通过runtime launch correlation和已审计的输入准备kernel符号归属，没有伪造CPU external id。其余通过correlation及external id关联，未发现kernel早于相应launch的时间倒置。

消息使用唯一prefill burst作为锚点，再按单in-flight、每步两层、方向、FIFO和精确shape配对。边界16个调用不强行补配。batch1/8分别有118/115条接收API在匹配发送API进入前已经返回，支持本次提前提交修正。时钟微小偏差没有独立界定，不能据此进行微秒级跨进程固定成本拟合。

同配置有profiler/无profiler阶段的A step中位数比为1.447/1.565；profiler明显扰动执行，trace仅用于归因和依赖，不进入固定成本profile。摘要：`afd-cpu-gpu-correlation-final-20261010.json`。

## 回放与回归

|batch|旧调用顺序µs|新调用顺序µs|
|---:|---:|---:|
|1|4259.360|4218.451|
|8|4261.027|4220.119|
|4|4260.313|4219.405|
|2|4259.825|4218.917|

各点31步真实路由，所有host参数按实际调用次数唯一收费；新模式降低约40.909µs，未使用整步差额修正参数。18个host字段每步计数与物化值通过审核，见`afd-native-order-replay-audit-final-20261010.json`。

331项AF相关回归通过，包括新增六项原生调用顺序测试、已有等待20/100/120/5与3/7/5等oracle。新增测试覆盖长度读回与发送重叠、CPU提交与设备完成分开、接收提前提交、cache/trace等价、取消，以及长度/内容估值失败后恢复。日志：`outputs/afd-native-order-regression-20261010.log`。

## 尚待完成及继续入口

冻结`afd-native-order-frozen-bundle-20261010.json`，主验收batch1/8/4/2沿用既有证据，另声明此前未用于拟合的batch3做补充验证。新增派生host/operation profile仅改变runtime绑定及profile id，固定参数逐项不变。不能把补充batch3冒充已经完成。

启动目录`profiling/afd-native-order-validation-20261010-v1`在GPU保护阶段失败：GPU2被PID939731占用65830MiB，GPU利用率66%；GPU3空闲。没有启动A/F子进程，不停止外部任务。后续必须使用全新目录v2或更高，保留v1失败日志。

空闲后运行现有`profiling/collect_afd_frozen_validation.py`，参数为`--base /home/youwei/autodl-tmp/af-workspace --output /home/youwei/autodl-tmp/af-workspace/profiling/afd-native-order-validation-20261010-v2 --bundle /home/youwei/autodl-tmp/af-workspace/profiling/afd-native-order-frozen-bundle-20261010.json --supplemental-batch 3`。每次启动仍由driver检查GPU UUID、进程、显存、利用率及18039/18040/19093端口。成功后取回timing-summary与capture JSONL，转换batch3路由，使用新模式回放并核对冻结哈希。

继续完成4.4/4.5的逐项覆盖/来源/统计清单，以及5.2的未校准、协议对齐与校准后同scope分项表。CUDA API/device启动的独立校准、原生stream精细重叠仍需保持明确边界；不能把含等待的kernel occupancy或inclusive控制区间拟合为固定开销，也不能因整步小于10倍而自动关闭剩余任务。

已生成`afd-native-order-fixed-inventory-20261010.json`：18个host字段与7个完整操作字段的数值、来源、覆盖清单；四个batch共124步的完整操作唯一计费检查通过，执行器源码仍匹配冻结bundle。该清单不替代独立分项验收。

已恢复现有`pkua100-gpu-2-3`自动监控为ACTIVE，间隔30分钟；资源状态不变时保持安静，两卡空闲后继续本节实验与剩余任务，成功启动后暂停监控以避免重复启动。当前没有本线程远端GPU作业。
