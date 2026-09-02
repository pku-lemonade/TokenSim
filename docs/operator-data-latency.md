# Operator-data 时延模型

本模块按照 `task-TokenSim-Groq-Operator-Data.md` 实现。它把时延工作分成两层：离线工具生成可追溯数据包，TokenSim 运行时只查表、受控插值和组合，不在事件循环中即时调用 Roofline 或 LLMCompass。

## 计算原理

离线公式使用：

```text
compute_time = FLOPs / peak_compute
memory_time  = bytes / peak_bandwidth
roofline_time = max(compute_time, memory_time)
latency = calibration_multiplier * roofline_time + calibration_offset
```

同一 shape 有 A 级算子/层级实测时，实测结果优先；公式结果为 C 级，只补齐缺测 shape。B 级公开端到端基准只检查整体量级，不拆成单算子值。D 级项目假设只能用于敏感性分析。

运行时先 exact match，再按表执行单维内部插值：GEMM 只沿 `m`，Prefill Attention 只沿 `input_seq_len`，Decode Attention 只沿 `context_len`，MoE 只沿 `num_tokens`。Elementwise 不插值，所有表第一版都不外推。失败会报告完整 key，不回退旧模型。

A 阶段组合 QKV GEMM、Attention core、输出投影与必要 elementwise；F 阶段组合 Router、MoE expert 与必要 elementwise。Manifest 的 `component_includes` 用于阻止 fused kernel 重复包含 softmax、SwiGLU、投影或 Router。Dispatch/combine 与 A-to-F/F-to-A 不进入 compute sum。

通信 service time 为：

```text
fixed_latency + payload_bytes / effective_bandwidth
```

Provider 不计算 queue time。全/半双工、容量和多个 AF blocks 的竞争仍由 AF executor 的 SimPy resource 产生。

## 数据包内容

```text
hardware.yaml
generation_meta.yaml
shape_manifests/<experiment-id>.yaml
gemm_perf.parquet                 gemm_analysis.parquet
context_attention_perf.parquet    context_attention_analysis.parquet
generation_attention_perf.parquet generation_attention_analysis.parquet
moe_perf.parquet                  moe_analysis.parquet
elementwise_perf.parquet          elementwise_analysis.parquet
communication_perf.parquet        communication_analysis.parquet
```

`*_perf.parquet` 是运行时窄表；`*_analysis.parquet` 保存公式分项、瓶颈、校准和证据等级。Loader 在仿真前检查 schema/version、单位、类型、来源、非负有限时延、重复 exact key，以及 perf/analysis key 一一对应。

## 生成与校验

安装依赖后，从仓库根目录运行：

```powershell
python -m TokenSim.operator_data.cli generate `
  --input data/operator_data/af_e2e_input.yaml `
  --output .operator-data-example

python -m TokenSim.operator_data.cli validate .operator-data-example
```

样例中的计算值是未校准公式，链路值也是示例，不能作为 Groq/Ada 物理性能结论。实际使用时必须替换 source、硬件版本、kernel/backend、shape manifest、实测 observation 和 calibration/holdout。

AF 集群配置显式选择正式 provider：

```json
{
  "af": {
    "enabled": true,
    "latency_provider": "operator_data",
    "operator_data_path": "../../../.operator-data-example",
    "operator_manifest_path": "../../operator_data/model_operators_example.yaml",
    "operator_data_backend": "analytical",
    "operator_link_group": "af-fabric",
    "operator_group_size": 2
  }
}
```

相对路径以 cluster JSON 所在目录为基准。`latency_provider="injected"` 保留给确定性测试 stub；AF-disabled 和旧 backend 不加载 Parquet reader。

## 验证边界

完整机制 E2E 会真实生成并加载 Parquet，经 `LLMEngine`、scheduler、AF executor、资源竞争和 trace 完成 Prefill/Decode：

```powershell
python -m unittest tests.test_operator_data_latency_provider
```

测试可证明 exact/interpolation、A/F 组合、通信公式、queue/service 分离和 trace 自洽。只有拿到独立 kernel、链路或真实系统 trace 后，才能报告物理准确性；否则结论必须写为“机制与模型自洽通过，物理准确性待实测”。

本 change 的实现验收结果为：operator-data 专项测试 20/20、AF 配置与 executor 回归 48/48、仓库完整 Python 测试 181/181 通过。当前未收到 Ada/Groq 的独立 kernel holdout、链路 benchmark 或真实系统 trace，因此当前精度状态正式记录为：**机制与模型自洽通过，物理准确性待实测**。测试中的 `measured:test` 和 example source 都是确定性 fixture，不是硬件 ground truth。
