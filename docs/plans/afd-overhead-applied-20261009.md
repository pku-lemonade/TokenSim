# 2026-10-09 原生固定成本补测与实际接入

本次完成任务11.1，进度25/34。继续使用tiny Qwen3 MoE、BF16 eager、GPU2(A)/GPU3(F)、1A1F、decode；不运行新batch2留出，不覆盖旧冻结验收。新增参数均由操作/API边界样本得到，没有用完整step差额拟合。

## 实测参数

所有数值单位µs。每项用batch1/8各三个独立进程窗口，先求窗口中位数再等权拟合。表内每次提交费用按backend和方向匹配，替代原来发送5µs/接收0µs的默认假设。

|覆盖|固定成本|计数|
|---|---:|---|
|采样token ready-input回读|56.342|每步一次|
|Scheduler.schedule|68.159167|每步一次|
|Scheduler.update_from_output + OutputProcessor.process_outputs|34.232333|每步一次|
|控制 torch.distributed NCCL A→F 发送提交|235.485042|每步两次，长度/内容各一次|
|控制 torch.distributed NCCL A→F 接收提交|221.822750|每步两次|
|PyNccl A→F 发送提交|94.381083|每层一次|
|PyNccl A→F 接收提交|83.518833|每层一次|
|PyNccl F→A 发送提交|122.212667|每层一次|
|PyNccl F→A 接收提交|102.099083|每层一次|

回读1200样本包含计时外前序设备同步；原生_to_list返回保证CPU数据可用。补采8:1、1:2、8:2采用新远端目录，复用10月8日已完成的1:0、8:0、1:1。六窗口逐token输出跨日一致。

CPU调度/输出共1200个decode步骤，按三个不重叠函数区间记录，逐步要求每函数恰好调用一次。未开启profiler，不包含输入准备、metadata、codec、采样、回读或GPU提交。输出成本仅覆盖表中两个函数，**不包含runner._bookkeeping_sync其余CPU工作**，没有把“输出处理”字段解释为所有外围成本都已覆盖。

通信提交共2400个零延迟训练样本和9600个延迟对照样本。测量输入设备工作先完成，API返回立即停止计时，完成同步放在计时外；NCCL初始化经过预热，TORCH_NCCL_BLOCKING_WAIT=0。发送端和接收端分别注入5/20ms延迟，所有提前调用的样本均在延迟对端进入API之前返回。延迟数据只用于边界审核，不参与固定参数训练；不将API返回视为数据已到达。

这些是隔离微基准的CPU API成本，仍受CPU调度/频率、runtime及测量工作负载影响。不能把单个参数当成所有机器或生产负载通用常数。PyNccl窗口间波动明显；具体p50/p95和窗口中位数保存在拟合JSON中，不能用本次回放宣称精确预测通过。

## 参数实际接入与同路由对照

|batch|原准备+codec profile|增加回读|再增加调度/输出|再增加通信提交|
|---:|---:|---:|---:|---:|
|1|934.753625|991.095625|1093.487125|1961.782458|
|8|936.029087|992.371087|1094.762587|1963.057920|

每点31个捕获的原生decode路由步骤，参数加载到通用ModelAFExecutor执行，不在报告里额外加数。审计逐步核对七个operation、两个codec、新的调度/输出和每方向/后端提交次数；费用与profile物化值一致，发送/接收默认项不再叠加。

回读依赖GPU token_selection完成，输出整理依赖回读；每步末尾和下一步入口顺序已校验。100/120µs输入ready加11µs回读分别得到111/131µs，回读常数不随输入推迟变化。标准LLMEngine、旧配置、取消清理及trace/cache测试通过。

通信提交带来的本次关键路径增量约868.295µs；发送/接收两端可以重叠，不能把所有表项逐项相加当作整步增量。当前逐步结果仍使用已有调用线程/设备DAG近似，**未完成真实预贴接收时序重建**。

## 产物与复现

- `afd-readback-profile-20261009.json`：旧六项operation加实测回读，v2，batch2验收pending。
- `afd-engine-cpu-profile-20261009.json`、`afd-engine-cpu-fit-20261009.json`：复用codec并加入明确覆盖的CPU外围成本。
- `afd-submit-profile-20261009.json`、`afd-submit-fit-20261009.json`：进一步加入六类原生提交成本。
- `afd-readback-replay-audit-20261009.json`、`afd-submit-replay-audit-20261009.json`：真实路由计费与依赖审核。
- 原样本：`outputs/afd-sampled-token-readback-20261008-v1`、`outputs/afd-engine-cpu-20261009-v1`、`outputs/afd-submit-20261009-v1`。
- 远端分别为`profiling/afd-sampled-token-readback-20261009-v1`、`profiling/afd-engine-cpu-20261009-v1`、`profiling/afd-submit-20261009-v1`，均位于`/home/youwei/autodl-tmp/af-workspace`。

拟合脚本为`fit_afd_sampled_readback`、`fit_afd_engine_cpu`、`fit_afd_submit`。用`python -m scripts.run_tiny_afd_alignment --output <新目录> --batches 1 --host-profile docs/plans/afd-submit-profile-20261009.json --operation-profile docs/plans/afd-readback-profile-20261009.json --routing-replay outputs/afd-live-capture-20261007/replay-b1.json --calibration-environment outputs/afd-live-capture-20261007/environment.json`复现batch1；batch8替换对应batch和路由文件。

插件metadata、p2p、attention_metadata、ffn_worker和qwen3_moe源码与基线哈希一致；两个runner的差异仅为旧诊断副本末尾插入的capture hook。新CPU采集前后检查原插件全部Python文件哈希守恒；回读仅改隔离副本，通信微基准不改插件。

## 未完成项与验收边界

1. 计算提交仍未发布实测profile。当前分析模型在`TokenSim/operator_data/analytical/gpu.py`为算子加入kernel_launch_us，已有CPU前向区间又含runtime enqueue；尚未得到能明确剥离同覆盖启动项及隐式同步的原生调用组样本。不能把0.5–0.6ms整层CPU前向直接加到GPU估值上。F原生Qwen3MoeSparseMoeBlock将gate/router放进FusedMoE，后续必须按这个实际调用组审核，不能把同一inclusive调用分别收费给router和expert。
2. runner bookkeeping及精确原生preposted ready时序仍缺实测对应，11.3保持未勾选；本次CPU输出参数明确为局部覆盖。
3. 三组独立消融、全部参数冻结后的batch1/8、batch4回归及新batch2三窗口正式验收尚未执行，11.4保持未勾选。不用10月7日原生整步与本轮1.96ms回放的比例冒充新验收。

已确认的是新成本被正确接入和局部边界通过；本轮没有宣称完整校准完成。

本地AF相关310项回归通过（`outputs/afd-overhead-regression-20261009.log`），OpenSpec strict validate通过。采样/提交拟合器测试覆盖缺窗口、重复、错误方向/环境、NaN、含等待及延迟对端未返回等拒绝条件。
