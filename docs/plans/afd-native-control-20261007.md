# 原生控制协议、源码审计与时间线摘要

本轮继续实施`align-vllm-afd-protocol-and-timing`。完成任务1.2源码覆盖审计、3.5按需时间线摘要，并实现2.4中的metadata部分；真实路由回放、精确API/stream边界和校准未完成。没有用整步差额生成任何固定常数。

## 原生调用与计费覆盖审计

插件源码根目录：PKUa100 `/home/youwei/autodl-tmp/af-workspace/src/afd-plugin-dbo-cache-20261007`。原生vLLM根目录：`/home/youwei/autodl-tmp/af-workspace/envs/afd026/lib/python3.12/site-packages/vllm`。插件关键文件SHA256与原序列化函数生成的字节样例保存于`afd-native-source-audit-20261007.json`，审计脚本只在CPU提取/执行两个函数，未导入插件或使用GPU。

| 原生路径/定位 | 已确认语义 | 模拟覆盖与仍缺项 |
| --- | --- | --- |
| `v1/worker/attention_metadata.py:70–111,174` | eager model forward之前发送一次DP控制；DP=1无ubatch为stage0 | 每iteration两条control消息；不按layer收费 |
| `connectors/metadata.py:320,376,403` | compact/sorted JSON；先long长度，再uint8内容；接收长度后`.item()`，内容后`.cpu()`和解析 | 精确字节协议、独立staging/回读/解析服务与依赖；时长未校准 |
| `connectors/gpu/p2p.py:807–864` | 控制使用torch.distributed NCCL进程组 | 与PyNccl data分后端计费/缓存；目前均仅catalog物理服务假设 |
| `v1/worker/ffn_worker.py:189–212` | F先阻塞接收控制，执行所有层后CUDA synchronize，再接收下一步控制 | F控制完成门控activation接收；首版单in-flight，未把同步等待收作固定服务 |
| `v1/worker/ffn_model_runner.py:189–246` | 根据控制构造F上下文，按layer/stage接收、计算、发送完整hidden | F Router→grouped expert→combine→返回；逐层真实路由仍未回放 |
| `connectors/gpu/p2p.py:305,374,410,540,628` | data custom op/PyNccl；1A1F每方向每层一个hidden张量 | T×H×2 bytes，不乘top-k；源本地完成≠接收可用，API精细边界待测 |
| 插件`model_executor/models/qwen3_moe.py:165–216,225` | A持有attention、input/post norm；F持有原生MLP且无这两个norm | 当前native composition已排除F重复norm/residual |
| vLLM `model_executor/models/qwen3_moe.py:333–352,409–429` | Q/K norm在A；attention后residual+post norm在A，MLP后residual与下一输入norm融合 | Q/K norm已建模，融合费用和布局拷贝仍需改进，不将融合kernel任意拆分拟合 |
| 同文件`:219–253` | native MoE内部gate/TopK、grouped experts、加权返回；一层一次上层调用包含多个kernel | 模拟grouped节点保留histogram、F combine唯一；当前额外compute调度按DAG节点近似，尚未逐kernel提交对齐 |
| 同文件`:466,485,652`及`v1/worker/gpu_model_runner.py:3648` | embedding、final norm、logits与sampler在A侧完整引擎 | ModelAFExecutor边界已有；索引/slot mapping/copy等辅助项仍不完整，不能摊进F计算 |

F控制接收在server loop顶部建立，确实可以早于A准备完成。data接收调用在控制解析后、逐层循环中建立；当前聚合模型仍采用数据依赖完成后建立接收提交的近似，不声称复现所有预贴/CPU阻塞时刻。原生NCCL waiting kernel占用设备的影响也未作为纯传输成本拟合。

## 已实施控制时间线

`A准备 → 控制编码 → 长度/内容H2D → 长度传输 → F长度回读 → 内容传输`。

内容交付后分为两条可重叠路径：A开始模型计算；F执行内容回读和JSON解析。activation必须同时满足A数据产生和F控制就绪。长度回读阻止内容接收提前开始。控制与data使用相同fabric队列；提交、host准备和完成不占fabric。

两条控制消息后端标记为`torch_distributed_nccl`，data为`pynccl`；新增query backend字段进入service缓存键。无等待时间进入缓存，重复step重新排队。control不再重复收data的host_completion_us，设备回读与CPU解析独立计费。

模型将control内容交付作为A执行model forward的放行点，这是聚合stream完成近似，尚不等同实测send API返回。控制H2D/回读暂使用worker设备资源，不模拟独立copy engine。上述限制保留在设计和未完成任务中。

## 配置与使用

`af.communication_timing.native_control_enabled`默认false以保留旧基线，新native对照脚本默认true。只支持既有1A1F、TP/DP/EP=1、BF16 eager decode、一个未分块stage和一个in-flight iteration；多block或并发事务明确拒绝。

新增固定服务项（µs，默认0且未校准）：`control_encode_us`、`control_h2d_us`、`control_length_readback_us`、`control_payload_readback_us`、`control_decode_us`。只包含各自纯服务，不得包含阻塞等待，也不得与a/f_step_prepare_us覆盖同一段工作。非法值和关闭control却提供非零成本的配置被拒绝。

本地机制测试示例（数值不是校准参数）：

```powershell
python -m scripts.run_tiny_afd_alignment --output outputs/native-control-example --batches 1 8 4 --a-step-prepare-us 10 --f-step-prepare-us 200 --control-length-readback-us 3 --control-payload-readback-us 5 --control-decode-us 10
```

用`--no-native-control`复现之前不含控制消息的基线。报告包含`control_protocol_status`、`fixed_host_assumptions`、data/control各自payload和`calibration_status=uncalibrated`。batch1/4/8每步控制均为8+179 bytes；179来自原序列化函数，代码按内容计算，batch16为8+181。两层data仍为四条消息。

## 时间线摘要

新增`AFTraceRecorder.summarize_timeline(iteration)`，只处理当前iteration索引，不扫描历史。对照报告导出各类service sum、busy区间并集、首步关键链和各步平均关键链组成，分别列host、compute、control device、data/control transport、peer wait、resource queue及unexplained。

关键链根据显式依赖及已知容量1资源的blocking时间戳选取一条，存在多个等长路径时只报告一条。链上互不重叠区间重建makespan；不能把不同类别busy并集再相加。没有trace时保留墙钟、全部标为unexplained；跨iteration资源blocking保留queue，不读取历史为其强行归因。

确定性oracle：两条并行分支10和20，后续5，service sum=35，关键链/makespan=25；外部资源等待与缺失依赖间隙分别标记queue和unexplained。控制oracle：长度service5、长度回读7、内容service5，则内容17完成；F内容回读11+解析200，F在228就绪；A从17计算100，activation在228开始、等待111、transport service仍5。

## 验证与待测

新增控制协议11项及时间线5项测试，覆盖原函数wire逐字节一致、消息数/bytes/dtype/backend、A/F重叠、缓存后端隔离、trace/cache等价、非法配置、取消/错误后资源复用、单stage/单in-flight边界、关键链重建及未知间隙。

验证结果：`python -m unittest discover -s tests -p 'test_af*.py' -q`共252项通过；OpenSpec严格校验、修改文件语法检查与`git diff --check`通过。

本地完整模型batch1/8/4各31步机制运行通过，示例平均step约427.596/429.574/429.024µs，均为假设参数输出，不与原生实测宣称对齐。GPU2目前被其他任务占用约70.9GB，GPU3空闲，未启动双卡校准。

剩余优先项：精确原生host/API/stream边界、control后端专用微基准、真实路由回放、非重叠profile与融合compute覆盖、冻结参数后的三窗口留出验证。本轮不发布已校准profile。
