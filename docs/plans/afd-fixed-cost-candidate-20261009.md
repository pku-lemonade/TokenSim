# 2026-10-09 原生调用组固定成本候选

继续`align-vllm-afd-protocol-and-timing`，限定tiny Qwen3 MoE、BF16 eager、GPU2(A)/GPU3(F)、1A1F decode。以下为实际加载到通用执行器的候选参数，完整校准仍未通过。旧profile与batch4冻结验收保留。

## 新增覆盖与数值

单位µs，batch1/8各三个独立进程窗口。每个原生函数每窗200样本；先求各层/窗口中位数，再按预先声明规则等权平均，不使用整步差额拟合。

|字段|数值|计数与覆盖|
|---|---:|---|
|compute_a_embedding_us|45.559917|每步一次embedding CPU|
|compute_a_attention_us|434.480500|每层一次input norm+self attention CPU|
|compute_a_post_norm_us|27.458333|每层一次post-attention norm CPU|
|compute_f_moe_block_us|583.046042|每层一次原生router+grouped experts+combine CPU|
|compute_a_final_norm_us|24.270667|每步一次final norm CPU|
|compute_a_lm_head_us|54.013917|每步一次logits CPU|
|compute_a_token_selection_us|69.465833|每步一次_sample CPU|
|engine_runner_bookkeeping_us|41.703833|每步一次_bookkeeping_sync，扣除嵌套_to_list|

计算CPU共14400个原生函数样本，另有1200个bookkeeping样本。原有采样回读56.342µs、调度68.159167µs、frontend输出34.232333µs和六类消息提交继续使用其独立实测来源，见`afd-overhead-applied-20261009.md`。

计时前完成前序设备工作，计时内利用CUPTI runtime/driver回调按线程深度求API区间并集，CPU候选=函数wall−API并集。嵌套runtime→driver只扣一次。这是CUDA API之外的框架组织成本；被扣去的runtime/driver成本未校准，已有设备启动下限仍为分析假设。不能说CPU提交与设备启动已经全部精确分离。空探针中位数0.32–0.51µs仅衡量计时器入口/出口，不证明完整回调扰动很小。

F batch8第三窗每层CPU中位数约710.5µs，其余五窗约552.5–561.5µs。保留所有合格窗口，没有删除慢窗追平整步。完整拟合、分位数与局限在`afd-compute-cpu-fit-20261009.json`；随后做无CUPTI诊断评估该候选的可信度，诊断wall不能直接换成CPU服务。

## 通用执行器接入

- F原生MoE是一整个调用，映射到router之前收费，内部expert/combine不重复收费；与独立router/expert提交字段互斥。
- token设备计算→readback→bookkeeping→frontend输出建立完成依赖，所有每步项恰好一次。
- bookkeeping排除完整嵌套回读，frontend字段只覆盖Scheduler.update_from_output及OutputProcessor.process_outputs；异步调度、logprobs与路由额外D2H不在本次支持域。
- CPU组整体放在对应设备组之前，仍是现有DAG聚合近似，不是逐kernel enqueue回放。具体API成本、接收预贴、stream时序保持缺口。
- trace逐项核对31步的实测值与计数，GPU设备服务没有因添加host成本而修改；cache/trace关闭时完成时刻一致。

## 同路由消融

单位µs/step，都是原生捕获路由；baseline已经包含旧准备操作与codec。

|启用项|batch1|batch8|相对baseline增量约|
|---|---:|---:|---:|
|baseline|934.754|936.029|0|
|仅新增token回读|991.096|992.371|56.342|
|仅新增计算调用组|3184.765|3186.432|2250|
|仅新增消息提交与CPU外围|1947.144|1948.420|1012.391|
|全部启用|4259.360|4261.027|3325|

单独增量不可相加为全部启用效果，host/GPU/完整操作重叠关系会改变。“仅外围”采用synthetic_test零readback保留token-ready依赖，明确是反事实，不是假称实测零成本。正式bundle仍为56.342µs。原始结果见`afd-fixed-cost-ablation-20261009.json`。

## 冻结与验收

`afd-frozen-fixed-cost-bundle-20261009.json`保存完整host/operation profile和执行器源码哈希。先冻结，再按batch1/8训练回验、batch4历史回归、batch2新留出的顺序各运行三组无hook/无profiler窗口；路由采集单独进行。所有启动复查GPU2/3进程、显存、利用率与端口；不预留资源、不干扰其他进程。

正式结果另存`afd-fixed-cost-validation-20261009.json`，其检查包括冻结哈希、逐项计费、输出一致性及每窗口max/min<10。即使完整step通过，仍不能将未测runtime/API、精确接收预贴与同窗口CPU/GPU归因标为已通过。

本地315项AF相关回归通过；随后新增whole-MoE/bookkeeping的cache/trace与设备服务不变检查，专项8项通过。OpenSpec strict validate通过。

### 最终候选对照结果

十二个独立无hook/无profiler窗口共63829个decode步骤；各batch跨窗口输出哈希一致。冻结参数及执行器源码哈希审核通过，未按留出结果调参。

|batch|用途|实测均值ms|模拟ms|模拟/实测|
|---:|---|---:|---:|---:|
|1|训练回验|2.633|4.259|1.618|
|8|训练回验|2.768|4.261|1.539|
|4|历史回归|2.728|4.260|1.562|
|2|新留出|2.738|4.260|1.556|

每个独立窗口的整步max/min均小于10。batch2另外采集31步路由，62个逐层expert histogram、124条hidden消息与62条metadata消息通过审计；hidden每条1024bytes，metadata长度8bytes/内容179bytes。结果见`afd-fixed-cost-validation-20261009.json`及`afd-fixed-cost-heldout-routing-20261009.json`。

无CUPTI同边界诊断每batch/role一个窗口。A侧开启/关闭CUPTI的调用wall比值约1.05–1.31；F侧约0.85–1.24，跨进程CPU频率与调度变化也包含在此差异中。F batch8扣API后的CPU均值仍比无CUPTI完整调用wall高约7%，说明窗口波动/探针偏差不可忽略；没有据此改写参数或声称完成精确CPU校准。两种模式都在调用前同步，不能隔离这种同步对原生重叠的影响。详见`afd-compute-probe-diagnostic-20261009.json`。

分项边界：GPU服务在同路由代入前后逐节点不变，之前的GPU计算和双backend设备服务对照仍为历史独立证据；本次没有新增完整原生同窗口host/device/transport分量的独立验收。CPU费用有原生调用组样本、三窗口统计和实际计数证据，但被排除的CUDA API服务、精确预贴及stream顺序仍未校准。不能把同一拟合样本与参数接近作为独立验证，也不能将4.26ms整步结果解释为所有分项都准确。

11.2–11.4本轮完成；3.1/3.2/4.2/4.4/4.5/5.2保留未完成，总进度28/34。当前已达成整步数量级目标，完整变更未结束，尚不能归档。

## 复现

```powershell
python -m scripts.fit_afd_compute_cpu outputs/afd-compute-cpu-20261009 --base-profile docs/plans/afd-bookkeeping-profile-20261009.json --output <新profile.json> --report <新fit.json>
python -m scripts.run_afd_fixed_cost_ablation --output <新的消融目录>
python -m scripts.audit_afd_fixed_cost_replay --profile docs/plans/afd-compute-cpu-profile-20261009.json --root-prefix outputs/afd-compute-cpu-replay-20261009-b --output <新audit.json>
```

采集原始窗口位于PKUa100的`profiling/afd-compute-cpu-20261009-v1`及`v2`，bookkeeping位于`profiling/afd-bookkeeping-20261009-v1`；本地小摘要分别保存在`outputs/afd-compute-cpu-20261009`与`outputs/afd-bookkeeping-20261009-v1`。已安装插件不作修改。
