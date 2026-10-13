# 整步低估的CPU调用与完成链路排查

本轮只做定位，不改变冻结profile、估值公式或参考插件文件。复用tiny Qwen3两层、GPU2(A)/GPU3(F)、BF16 eager、输入128/输出32、batch1/8。每点分别运行正常计时、cProfile及轻量函数边界计时，各为独立进程；不使用batch4。五轮预热后只记录decode，GPU profiler全部关闭。六次测试的输出token逐项一致，原插件Python源码哈希守恒。

## 直接结论

整步差距不能由GPU算子自身的几十微秒误差解释。已定位的缺口集中在CPU驱动模型/算子、模型入口出口及采样后的同步回读。当前模拟只接入部分准备操作，compute_submit_us、host_receive_submit_us、host_completion_us仍是未测零值，host_submit_us仍为5µs假设。这些是覆盖缺口，不能把零值理解为实测成本为零。

GPU纯计算和CPU前向调用不是同一口径。CPU调用包含Python/框架执行、CUDA提交及可能的设备/对端等待；异步GPU完成可能落在后面的采样回读处，不能把整段CPU前向再叠加GPU kernel时长。

## 同一窗口的非重叠整步拆解

单位µs/step，均值。下列行用入口/出口区间按嵌套关系切分，每一步逐一验证分项之和等于观测墙钟。原始事件中control位于model forward内、逐层forward位于外层execute_model内，报告已经扣除这些嵌套，不能再加inclusive父函数。

| 非重叠区间 | batch1 | batch8 |
| --- | ---: | ---: |
| A输入准备 | 412.45 | 431.01 |
| A attention metadata构建 | 107.55 | 110.56 |
| 控制payload整理与两次发送 | 233.04 | 236.33 |
| Layer 0 CPU前向调用 | 600.73 | 657.35 |
| Layer 1 CPU前向调用 | 526.13 | 576.07 |
| 模型前向其余部分，已扣控制/两层 | 165.24 | 168.99 |
| execute_model其余部分，已扣上述子项 | 248.72 | 263.53 |
| sample_tokens，含完成等待/CPU回读 | 548.93 | 379.36 |
| 其余engine区间 | 153.07 | 239.12 |
| 整步观测 | 2995.87 | 3062.32 |

batch1记录4836步，batch8记录4681步。外层driver另含约1µs hook收集尾部，计时分别2996.86/3063.21µs；不把该尾部写成引擎开销。三类其余区间只是时间所在位置，尚未全部拆成纯CPU服务或阻塞等待。层编号按每步两次decoder调用顺序识别，与GPU kernel独立测量严格区别。

## 具体问题及定位

1. **算子提交费用覆盖不足。** 当前`TokenSim/af/executor.py::_run_planned_compute`只有非零compute_submit_us才安排额外host提交，冻结配置为0；原生一层CPU前向约0.53–0.66ms，GPU层计算仅为其一部分。`vllm/model_executor/models/qwen3_moe.py:409`为本轮计时边界，内部Python/PyTorch调用和不同算子的实际提交次数需要按原生调用归属，不能统一按逻辑专家数收费。
2. **采样结束不等于CPU获得token。** 原生`gpu_model_runner.py:4496 sample_tokens`经`:3668 _bookkeeping_sync`调用`:7757 _to_list`。后者执行pinned CPU缓冲copy、transfer_event.record、transfer_event.synchronize，再tolist。当前`TokenSim/af/model_executor.py::_run_block`在token_selection DAG结束后直接记录block_completed_at，缺少显式D2H/token可用完成节点。当前host_completion_us属于通信消费，不能拿它代替整个采样输出完成。
3. **控制消息提交仍使用粗略假设。** 原生`afd_plugin/connectors/metadata.py:376 send_control_payload`总wall约233/236µs，其中两次torch.send累计约146/147µs；PyTorch `distributed_c10d.py:2600 send`内含isend和work.wait。模拟控制传输、codec和staging已有独立费用，但5µs/次提交不具备覆盖该API的实测证据。146µs并不等于纯CPU或纯网络时间，不能直接替换alpha。
4. **引擎外围费用没有完整进入模型。** 模型外层forward其余、execute_model其余、engine其余合计约567/672µs；包含模型边界操作、调度/请求状态更新/返回等混合工作，其中有些设备操作已有模拟条目，需要先去重，不能全部补成新增固定费。

准备成本并非剩余低估的主要方向：本轮原生输入准备+metadata约520/542µs，当前完整操作常数之和约617µs；后者包含人为自身完成同步，二者范围不完全相同。本轮没有为了贴合原生而调小它们。

## 测量扰动与纠正

| 计时方式 | batch1整步均值µs | batch8整步均值µs |
| --- | ---: | ---: |
| 无hook正常计时 | 3018.84 | 3311.71 |
| cProfile | 3745.03 | 4348.88 |
| 轻量边界计时，driver口径 | 2996.86 | 3063.21 |

cProfile整步比正常窗口高24%/31%。其control payload累计显示1248/1352µs，而轻量诊断仅233/236µs，sample_tokens的分布也明显改变，说明探针会改变CPU/GPU重叠及等待落点。因此cProfile只用于发现函数，不能作为固定成本参数或分项相加证据。轻量与正常窗口均值相差约−0.7%/−7.5%；窗口顺序及运行时状态也会影响结果，不能宣称零扰动或做精确剩余差额拟合。

`torch_isend`在轻量hooks中未捕获事件：torch.send内部引用的模块别名未经过该包装。没有据此宣称isend耗时为0，也没有将send减未捕获项当作wait。decorator的inspect源文件混用了wrapper路径和被包装函数行号；已在本地后续采集器改用inspect.unwrap。上述源码定位另经服务器原文件核对，原实验来源字段保持不修改。

## 后续修复顺序

首先在模拟中明确token采样、CPU回读及token可用完成节点，读回前的依赖等待与读回自身服务分别处理；其次按真实算子/API调用测提交服务，保留CPU提交与GPU执行可重叠，不能把整层0.6ms作为固定费叠加。控制后端单独补测提交/返回边界；最后纳入剩余引擎调度和输出CPU整理。所有新参数仍须ready条件审计、三窗口和覆盖去重；不变更已有batch4冻结验收结论，也不以本次诊断冒充新的校准通过。

## 证据及执行状态

- 小摘要：`afd-cpu-attribution-20261008.json`，逐项原始证据在`outputs/afd-cpu-attribution-20261008-v1`。
- 远端完整数据：`profiling/afd-cpu-attribution-20261008-v1`与`profiling/afd-phase-attribution-20261008-v1`，位于`/home/youwei/autodl-tmp/af-workspace`下；大pstats与phase事件未回传。
- 新脚本：`collect_afd_cpu_attribution.py`、`afd_phase_diagnostics.py`、`summarize_afd_cpu_phases.py`。采集复用空闲/端口检查和仅清理自身子进程的既有driver。
- 最后一组诊断PID为4145659/4145660，已由driver退出清理。随后GPU2出现另一PID4154292，占39169MiB、24%利用率；没有再次启动GPU测试或停止该进程。GPU3空闲。本次未改变监控状态。
