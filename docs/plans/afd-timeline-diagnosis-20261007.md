# AFD完整步约20倍差距：定位记录

已确认差距不只是Attention/MoE算子或NVLink带宽估值问题：当前模拟缺少真实eager引擎的主机输入准备、metadata控制协议、逐算子提交及接收就绪链路；被测插件另外存在CUDA热路径反复导入缺失NPU模块的性能问题。

本轮仅诊断，未修改插件业务逻辑或模拟参数。复用三组GPU2/3 trace，追加无GPU的DBO guard测试，以及一次隔离batch1 decode CPU cProfile（2480步）；诊断进程已退出，最后GPU计算进程列表为空。

## 1. 确认的插件性能问题：重复失败导入

远端插件根为 `/home/youwei/autodl-tmp/af-workspace/src/afd-plugin/afd_plugin`。

- `v1/worker/dbo.py:12–23`：每次RemoteFFNProxy都调用manual_dbo_yield，即使CUDA路径DBO没有实际切换。
- `v1/worker/dbo.py:52–62`：每次guard尝试导入`afd_plugin.v1.worker.npu.ubatching`。
- `v1/worker/npu/ubatching.py:8`：该模块导入`torch_npu`，当前服务器未安装，产生ImportError。
- 失败模块未留在sys.modules中，下一次又进行模块查找、字节码读取和执行。100次guard的cProfile出现100次exec_module/200次_find_and_load，直接证实重复行为。

CPU-only无profiler计时：guard中位数164.73µs/次，原生dbo_enabled检查约0.13µs/次。当前两层每步调用两次。已有torch profiler trace的manual_dbo_yield约0.55–0.70ms/步；新增cProfile约1.11ms/步。后两组受各自采样扰动，不能将1.11ms当成无profiler可节省时间。CPU-only结果说明该问题确实存在，而非只由GPU profiler产生。

建议在平台初始化时选择CUDA/NPU实现，或按平台惰性加载并缓存实现/不可用状态；不要在每层通过失败导入探测平台。必须保持启用DBO时的yield语义，不能直接全局删掉yield。修复需在隔离插件副本中A/B验证，本轮没有实施。

## 2. metadata是独立的每步控制协议，当前模拟缺失

调用链：`v1/worker/attention_model_runner.py:332–341` → `attention_metadata.py:174–218` → `connectors/gpu/p2p.py:807` → `connectors/metadata.py:376`。

`send_control_payload`的具体步骤：

1. `metadata.py:389–396`：编码JSON，构建CPU uint8 tensor，拷到GPU，再构建GPU size tensor。
2. `metadata.py:399–400`：调用两次torch.distributed.send，先传长度，再传payload。
3. F端`metadata.py:410–420`：recv长度 → size_tensor.item() → 动态分配payload → recv payload → object_tensor.cpu().numpy().tobytes() → JSON解码。

cProfile确认每decode步调用一次send_control_payload、两次distributed.send；控制协议发生在模型层循环之前，不是每层各发送一套。在该采样中send_control_payload约1.634ms/步，含构建、拷贝、API及阻塞成本，不是metadata字节的纯传输时间。不能据此给每条消息增加1.634ms，也不能把这个含等待的值直接导入transport_service。

这是明确的模拟覆盖缺项，并不单凭热点就认定协议实现有正确性bug。优化方向是减少CPU/GPU来回、固定/复用控制buffer或消息格式、按实际状态复用形状信息；必须保留事务对应关系、状态变化、warmup/profile/graph标志，不能无条件省略每步控制消息。

## 3. eager主机执行远大于当前host_submit假设

新增decode-only cProfile的A侧热点（batch1，2480步，含采样扰动）：

| 路径 | inclusive平均µs/步 | 定位 |
| --- | ---: | --- |
| metadata发送完整链路 | 1633.99 | 插件`connectors/metadata.py:376` |
| maybe_apply_dbo_yield | 1111.92 | 插件`v1/worker/dbo.py:12` |
| 输入准备 | 652.92 | vLLM`v1/worker/gpu_model_runner.py:1937`，`_prepare_inputs` |
| A数据发送包装 | 187.56 | 插件`connectors/gpu/p2p.py:305` |
| A数据接收包装 | 200.57 | 插件`connectors/gpu/p2p.py:374` |

这些是函数范围而非硬件纯服务。inclusive条目有父子重叠，不把所有热点表行相加；C++扩展调用中也可能包含等待。当前解析模拟计算纯服务不包含Python逐算子dispatch、输入准备、调度和输出处理的完整成本。

本地模拟定位：`TokenSim/config/cluster_config.py:351`默认host_submit_us=5；`TokenSim/af/executor.py:798–817`只针对消息执行该提交阶段。两层四条data消息合计20µs，并没有因此覆盖上表所有host项。`scripts/run_tiny_afd_alignment.py:46`运行analytical_only，属于明确未校准基线。

## 4. F端长同步主要表现为等待，不能再收费一次

原始torch profiler分析：

| Batch | A profile步µs | A CUDA同步区间并集µs | F CUDA同步区间并集µs |
| --- | ---: | ---: | ---: |
| 1 | 6163.11 | 26.12 | 2638.33 |
| 4 | 5142.20 | 25.06 | 2215.53 |
| 8 | 4986.12 | 23.65 | 2216.36 |

F的热点集中在item/_local_scalar_dense/cudaStreamSynchronize；源码控制协议存在GPU长度回读以及payload回CPU，均会要求数据就绪。`v1/worker/ffn_worker.py:189`先接收控制消息，之后才执行模型；`:211–212`还有每轮CUDA synchronize。这些建立了真实等待与执行顺序。未采集逐消息跨进程identity，故尚不能把每个F同步微秒精确归给某一个A函数。

不能把F的2ms等待加到A的主机准备之后作为又一笔独立成本；很可能包含等A提交的同一段时间。也不能把几毫秒的NCCL kernel占用解释成纯传输时间。需要用接收就绪、producer完成、stream依赖和消息完成事件产生等待。

本地`TokenSim/af/executor.py:821–830`只有提供transfer_readiness时才等待额外事件；当前对照没有原生ready回放，使用dag_dependencies_only。等待/传输分离接口已存在，但缺少原生输入事件和host时序，无法自行重现真实停顿。这解释了“代码已做等待分离，为何结果仍不对”。

## 5. 测量与结论的边界

- 无profiler真实步为3.71–3.86ms，模拟约0.19ms。原始torch profile步为约4.99–6.16ms；新增cProfile也将执行减慢。因此不能把上面的采样函数时长直接拟合成无profiler参数。
- A原始trace有约2.0–2.46ms未落在主线程cpu_op/runtime标注并集中，可能含Python、线程调度和采样成本；不将它全部认定为某一类CPU服务。新增cProfile用于找到具体调用来源。
- 本次每进程时间线独立分析，没有把两个进程的绝对时间直接拼接为关键路径。主线程区间采用并集，父子CPU op不会重复计入区间覆盖。
- 独立通信实验中“send局部完成”未覆盖延迟接收，而数据＋回执覆盖5/20ms延迟；正式模型应以接收数据可用的依赖为准。回执RTT不能直接充作单向参数。

## 建议处理顺序

1. 先在隔离插件副本修正NPU重复失败导入，做同配置无profiler A/B；不要将可避免的软件缺陷固化进校准常数。
2. 补metadata协议及host阶段边界：每步输入准备/控制协议、每层算子提交/数据提交、接收回读和每步完成。用真实调用依赖产生等待，覆盖范围唯一。
3. 采集低扰动边界计时，区分host纯工作与阻塞等待，冻结参数后再测留出batch。不要直接使用本次cProfile inclusive时长作为profile值。
4. 再校准层内attention/MoE、布局拷贝与层外GPU辅助项；这些有误差，但不足以单独解释20倍整步差距。

## 证据文件

本地`outputs/afd-timeline-diagnostic-20261007/`包含`timeline-diagnostic.json`、`dbo-cpu-diagnostic.json`、`cpu-summary.json`及两个源码只读快照`dbo.py`、`metadata.py`。同目录原始来源分别是远端`profiling/afd-gpu23-rerun-20261007/`和`profiling/afd-cpu-diagnostic-20261007/c1/`；raw trace和decode-cpu.pstats保留远端。

复现脚本：`diagnose_afd_timeline.py`（只读trace），`diagnose_dbo_cpu.py`（CUDA不可见的CPU-only计时），`prepare_cpu_diagnostic.py`（隔离driver，decode-only cProfile）。本轮未修改被测插件或模拟器代码，未提交。
