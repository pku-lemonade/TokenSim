# 当前层级通信版本校准接入与重新冻结

用户于2026-10-12授权适配当前层级通信实现、重新冻结并验收。另一线程已完成层级通信修复、规格同步及归档后，本线程继续；旧冻结包、原生窗口和profile均保留。

## 配置与计费

- tiny Qwen3 MoE、BF16 eager、1A1F decode，不扩展prefill或DSV3。
- 新`--pkua100-local-domains`显式映射A到GPU2、F到GPU3。当前引擎固定8卡逻辑节点，node_count=1，仅使用两个槽；不声称实机有八张卡。真实GPU2/3为NV12，复查确认。
- 本地带宽300GB/s、alpha=1µs沿用硬件目录假设；新公式为alpha+bytes/bandwidth，移除旧H/效率对启动项的影响。它们不是本轮新测的transport参数。
- 校准指纹包含通信域配置和模型版本，修改带宽、延迟、位置会拒绝旧profile。
- 18个host字段、7个完整操作的数值和覆盖保持不变；v3 profile预留batch5，v1/v2的batch4/batch2规则不变。所有既有路由保留assignment，只在审核后派生运行时绑定。
- 31步/点检查：每步2条控制、4条层内data消息，全部使用`domain:node0:local`；data每条为batch×256×2 bytes。host调用数、操作唯一计费、实际物化参数及原生调用顺序全部检查。

## 冻结与可恢复性

新包：`afd-hierarchical-frozen-bundle-20261012.json`，profile为同目录`afd-hierarchical-{host,operation}-profile-20261012.json`。

`outputs/afd-hierarchical-20261012/source-snapshot.zip`保存655项源码、数据和原始证据，逐项哈希写入bundle。测试发现依赖原仓库顶层util/benchmark，另存`support-source-sha256-v2.json`及冻结副本中的4个支持文件；原ZIP与bundle不重写。恢复运行需要二者联合，不能仅凭ZIP宣称独立测试环境完整。Python与包版本另存`local-environment.json`。

冻结副本路径为`outputs/afd-hierarchical-20261012/frozen-source`；当前工作区以后变化不能改变本次回放。352项AF回归、59项通信域/FIFO/placement/cache专项通过；早先两次因遗漏util/benchmark导致的import失败日志保留，不计为通过。新增v3留出规则与域参数变化拒绝测试通过，strict validate通过。

六个原生插件关键文件（控制消息、PyNccl、A/F runner、metadata、DBO）与历史源码哈希一致；vLLM 0.26.0、PyTorch 2.11.0未变。核对记录为`native-source-audit.json`。

## 本轮验收

远端全新目录`profiling/afd-hierarchical-validation-20261012-v1`，采集器`collect_afd_hierarchical_validation_20261012.py`。冻结后依次运行batch1/8、4/2/3、5，每点三个独立无hook/无profiler窗口，最后单独采集batch5路由。GPU空闲与端口检查由每次启动保护执行，空闲不构成资源预留。

本轮时间不用于修改参数；batch5此前未观测。已观测batch3只作回归。原生整步完成后使用冻结副本显式`--native-call-order --pkua100-local-domains`回放，`scripts.report_afd_hierarchical_validation`核对bundle、源码、18窗口、输出哈希、路由、逐项计费及local域。

## 分项证据与边界

`afd-hierarchical-components-20261012.json`记录历史设备分项与新模型比较：计算倍率约1.43–2.03，匹配transport倍率约8.17–9.11，均小于10。历史设备区间与当前原生整步不是同一窗口，不相加重建整步；transport仍为解析估值，旧CUPTI微基准不自动变为新实测profile。

25个参数的覆盖、训练统计与相对训练p50残差均已列出，训练拟合残差不是独立留出分项验收。CUDA API/device启动独立校准，以及host/完整操作的独立分项验收仍缺证据；4.4/4.5/5.2不能因整步通过而自动关闭。

## 冻结后正式结果

18个独立无hook/无profiler窗口完成，共88133个decode步。实测为三个窗口均值的等权平均；模拟为同配置31步真实路由回放均值，单位µs。

| batch | 用途 | 实测整步 | 实测成本接入后 | 模拟/实测 |
|---:|---|---:|---:|---:|
| 1 | 训练点验证 | 2916.280 | 4162.527 | 1.427 |
| 2 | 历史点回归 | 2923.783 | 4162.993 | 1.424 |
| 3 | 历史点回归 | 3047.311 | 4163.245 | 1.366 |
| 4 | 历史点回归 | 2863.307 | 4163.481 | 1.454 |
| 5 | 新留出 | 2882.869 | 4163.651 | 1.444 |
| 8 | 训练点验证 | 3099.380 | 4164.194 | 1.344 |

六个batch的每个窗口均满足max/min<10；这是数量级目标，不是精确误差验收。均值仍高估约34%–45%，batch5高估约44%。新结果没有用于调参，18个host字段及7个operation参数未改。

独立batch5采集已严格转换成31步路由。655项主源码哈希和4项补充依赖哈希复核一致；六点各31步的18项host成本、7项operation唯一计费、2条控制和4条层内数据消息及local域通过。18窗口输出在同batch内一致，纯计时与路由诊断分开。

batch5模拟关键链约为host服务3337.696µs、完整操作711.123µs、计算110.798µs、数据传输4.034µs。这是模型内分解，不能视为原生分项实测。固定成本相关项在模型中占主导，但本轮整步数据不能单独判定高估来自哪个常数、探针扰动或依赖重叠。不得按1.44倍统一缩减参数，也不以整步差额拟合固定费。

历史原始baseline（20261007）batch1/4/8模拟分别189.633/191.061/191.611µs，当时实测中位数3709.440/3862.348/3844.307µs。它们属于旧配置、旧调用顺序和不同时间窗口；仅保留历史，不与本轮均值作同scope归因。当前协议对齐但未加载profile的同路由结果保存在`outputs/afd-hierarchical-20261012/unfitted-b*/tokensim-b*.json`，六点约144.709–146.376µs；其中batch5为145.833µs，接入成本后4163.651µs。未拟合结果仍保留默认提交假设，并非所有开销为零。

机器可读验收：`afd-hierarchical-validation-20261012.json`；原生摘要及batch5路由：`outputs/afd-hierarchical-validation-20261012-v1`；分项覆盖与统计：`afd-hierarchical-components-20261012.json`。恢复冻结环境需要主源码和support两份ZIP。

采集进程退出码0；结束后GPU2/3无计算进程，18039/18040/19093无监听；未停止任何他人进程。监控保持暂停，避免重复采集。

12.1–12.3完成，变更总进度34/37。4.4/4.5/5.2仍缺CUDA API/device启动独立校准、host/operation独立分项留出验收及纯transport profile接入；完整校准不标通过。下一轮补测需先声明新的训练/验证计划，不使用本次batch5留出结果调参后仍称其未见。
