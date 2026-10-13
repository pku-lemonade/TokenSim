# 逐layer计算对照：TokenSim / vLLM+AFD

复用2026-10-06在PKUa100 GPU0/3采集的真实AFD trace，没有重新启动GPU测试。模型为此前的小型Qwen3 MoE，共2层，代码编号Layer 0和Layer 1；不能外推为DSV3逐层实测。

每个batch统计31个decode步（context 129–159），不含prefill、通信、CPU耗时和等待。本报告统一用每步kernel时长之和的**均值**，使层与层外成本可加；此前总表使用原生中位数，数值略有区别。A和F相加仅表示计算工作量，不是重叠后的实际层延迟。

## 逐层汇总

单位µs/该层/一次decode步。

| Batch | Layer | A 模拟 / 实测 | F 模拟 / 实测 | 合计模拟 / 实测 | 实测/模拟 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 27.31 / 31.94 | 18.52 / 26.51 | 45.83 / 58.45 | 1.28× |
| 1 | 1 | 30.31 / 31.50 | 18.52 / 26.53 | 48.83 / 58.03 | 1.19× |
| 4 | 0 | 27.48 / 46.35 | 19.05 / 29.49 | 46.52 / 75.84 | 1.63× |
| 4 | 1 | 30.48 / 46.06 | 19.05 / 29.36 | 49.53 / 75.41 | 1.52× |
| 8 | 0 | 27.70 / 46.90 | 19.08 / 29.75 | 46.78 / 76.64 | 1.64× |
| 8 | 1 | 30.71 / 47.29 | 19.08 / 29.42 | 49.79 / 76.71 | 1.54× |

为对应原生执行边界，模拟器原有Layer 0的MoE后residual费用移入Layer 1输入norm组，Layer 1的MoE后residual费用移入层外final norm组。这只是报告重分组，模拟器参数、事件和总成本均未修改。原生融合kernel不做任意拆分：Attention residual和post norm一起比较，MoE residual和下一个norm一起比较。未经重分组的模拟层总量保存在同名JSON的raw_logical_layer_simulated_us中。

## 层外GPU计算

| Batch | 层外模拟 | 层外实测 | 差额 |
| --- | ---: | ---: | ---: |
| 1 | 15.05 | 50.55 | 35.50 |
| 4 | 15.06 | 52.88 | 37.82 |
| 8 | 15.07 | 54.70 | 39.63 |

层外包括embedding、最终norm（融合最后一层residual）、logits/argmax，以及索引、拷贝、KV slot mapping等辅助GPU操作。它们每步执行，不能把这部分费用平均摊到每一层再拟合层算子。

## 已定位的差异

1. 两层计算耗时接近，没有某一层异常放大的迹象。逐层整体误差小于此前含层外成本的A侧倍率。
2. batch1每层A侧9个kernel；batch4/8每层12个，新增3个aten::copy_ kernel，每层合计约10–11µs。解析模型没有对应的独立布局拷贝项。这是batch增长差异的直接证据，不应只调大统一启动常数。
3. 原生每层Attention core约8.5–8.8µs，KV cache写入约2.9–3.1µs。模型generation_attention的kv_bytes已包含读写流量，但只用一个解析服务/启动项；不能说KV写入字节完全漏算。报告将原生core和cache写入合并，与模拟attention_core对比。
4. 模拟器把residual和norm分开计费，原生融合执行，部分norm组合反而高估；与此同时attention core、布局拷贝和部分投影低估。统一乘一个系数会掩盖不同方向的误差。
5. Router按原生gate至top-k结束的范围单独比较。专家、路由布局和combine作为整体比较，因为加权与下投影可能融合；不将moe_sum直接当作完整weighted_combine的实测值。
6. 层外实测约50–55µs，当前对应模型约15µs，差额约35–40µs。该差额属于本次GPU kernel口径，不等于CPU调度开销，更不属于网络传输。

## 切分与验证

原始AFD trace没有显式layer annotation。这里用已核对源码的顺序和形状还原：每个decode步A侧恰有5次hidden=256的norm调用（input0/post0/input1/post1/final），F侧恰有2次shape=[batch,256]的vllm::moe_forward。按调用顺序编号Layer 0/1。每个GPU kernel通过CUDA runtime/driver correlation匹配CPU提交，再归入层区间；无法匹配或结构不符会停止分析。

三个batch均识别31个decode步，无未分类层内kernel；层总和＋层外均值与总计算均值守恒，原生总计算中位数也与上一轮报告逐点一致。模拟重分组前后总成本一致。这是源码与形状支持的层序推断，不是原始trace自带的layer ID，也没有回放实际expert routing。

## 每层明细

布局拷贝模拟值0表示当前无独立费用条目，其他模型项是否间接吸收相关成本不在本次拟合范围。真实kernel时长受profiling影响，当前各batch只有一个AFD采样窗口；已有单卡三窗口对照只用于辅助验证总计算稳定性。

### Batch 1，Layer 0

| 部分 | 模拟 µs | 实测 µs |
| --- | ---: | ---: |
| 输入norm＋上一层residual | 3.001 | 2.930 |
| QKV投影 | 3.170 | 3.497 |
| Q norm | 3.001 | 2.886 |
| K norm | 3.000 | 2.812 |
| RoPE | 3.001 | 2.713 |
| Attention core＋KV写入 | 3.049 | 11.444 |
| 输出投影 | 3.085 | 2.866 |
| Attention residual＋post norm | 6.002 | 2.795 |
| 层内布局拷贝 | 0.000 | 0.000 |
| Router | 6.002 | 7.363 |
| 专家＋路由布局＋combine | 12.516 | 19.143 |

### Batch 1，Layer 1

| 部分 | 模拟 µs | 实测 µs |
| --- | ---: | ---: |
| 输入norm＋上一层residual | 6.002 | 2.665 |
| QKV投影 | 3.170 | 3.489 |
| Q norm | 3.001 | 2.846 |
| K norm | 3.000 | 2.746 |
| RoPE | 3.001 | 2.699 |
| Attention core＋KV写入 | 3.049 | 11.431 |
| 输出投影 | 3.085 | 2.873 |
| Attention residual＋post norm | 6.002 | 2.747 |
| 层内布局拷贝 | 0.000 | 0.000 |
| Router | 6.002 | 7.425 |
| 专家＋路由布局＋combine | 12.516 | 19.107 |

### Batch 4，Layer 0

| 部分 | 模拟 µs | 实测 µs |
| --- | ---: | ---: |
| 输入norm＋上一层residual | 3.003 | 2.845 |
| QKV投影 | 3.173 | 4.949 |
| Q norm | 3.003 | 2.923 |
| K norm | 3.002 | 2.828 |
| RoPE | 3.005 | 2.957 |
| Attention core＋KV写入 | 3.194 | 11.567 |
| 输出投影 | 3.087 | 4.937 |
| Attention residual＋post norm | 6.008 | 2.884 |
| 层内布局拷贝 | 0.000 | 10.458 |
| Router | 6.003 | 10.203 |
| 专家＋路由布局＋combine | 13.046 | 19.291 |

### Batch 4，Layer 1

| 部分 | 模拟 µs | 实测 µs |
| --- | ---: | ---: |
| 输入norm＋上一层residual | 6.008 | 2.775 |
| QKV投影 | 3.173 | 5.149 |
| Q norm | 3.003 | 2.884 |
| K norm | 3.002 | 2.832 |
| RoPE | 3.005 | 2.845 |
| Attention core＋KV写入 | 3.194 | 11.521 |
| 输出投影 | 3.087 | 4.924 |
| Attention residual＋post norm | 6.008 | 2.893 |
| 层内布局拷贝 | 0.000 | 10.235 |
| Router | 6.003 | 10.135 |
| 专家＋路由布局＋combine | 13.046 | 19.221 |

### Batch 8，Layer 0

| 部分 | 模拟 µs | 实测 µs |
| --- | ---: | ---: |
| 输入norm＋上一层residual | 3.006 | 2.858 |
| QKV投影 | 3.177 | 4.981 |
| Q norm | 3.006 | 3.067 |
| K norm | 3.003 | 2.968 |
| RoPE | 3.009 | 2.705 |
| Attention core＋KV写入 | 3.389 | 11.853 |
| 输出投影 | 3.090 | 5.224 |
| Attention residual＋post norm | 6.015 | 2.892 |
| 层内布局拷贝 | 0.000 | 10.348 |
| Router | 6.004 | 10.262 |
| 专家＋路由布局＋combine | 13.077 | 19.488 |

### Batch 8，Layer 1

| 部分 | 模拟 µs | 实测 µs |
| --- | ---: | ---: |
| 输入norm＋上一层residual | 6.015 | 2.777 |
| QKV投影 | 3.177 | 5.205 |
| Q norm | 3.006 | 2.953 |
| K norm | 3.003 | 2.972 |
| RoPE | 3.009 | 2.850 |
| Attention core＋KV写入 | 3.389 | 11.810 |
| 输出投影 | 3.090 | 5.022 |
| Attention residual＋post norm | 6.015 | 2.871 |
| 层内布局拷贝 | 0.000 | 10.832 |
| Router | 6.004 | 10.189 |
| 专家＋路由布局＋combine | 13.077 | 19.228 |

## 证据和复现

- 同名JSON保存逐层模拟分项、重分组前层总量、原生均值/中位数/范围、kernel数量和copy shape计数。
- 远端原始trace：`/home/youwei/autodl-tmp/af-workspace/profiling/afd-compute-gpu03-20261006/c{1,4,8}/{attention,ffn}`。
- 远端执行：`envs/afd026/bin/python profiling/summarize_layer_compute.py profiling/afd-compute-gpu03-20261006`，只生成layer-summary.json，不使用GPU。
- 下载小摘要后，本地运行：`python scripts/report_layer_compute.py`。
- 模型位置：`TokenSim/af/operator_provider.py::_attention_components`、`TokenSim/operator_data/workload.py::generation_attention_work`、`TokenSim/operator_data/analytical/gpu.py::GPURooflineModel.estimate`。

本轮只增加诊断脚本和报告，没有修改估值公式、校准参数或提交代码。
