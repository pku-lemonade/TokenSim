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

运行时先 exact match，再按表执行受限的单维内部插值：GEMM 只沿 `m`，Prefill Attention 只沿 `input_seq_len`，Decode Attention 只沿单个 request 的 `context_len`，Routed/Shared Expert 各自只沿 `num_tokens`。Weighted combine、residual、encode/decode 等 elementwise operation 不插值，所有表第一版都不外推。查询失败会报告完整 key，不回退旧模型或运行时 analytical 公式。

Compute provider 按互斥 operation 组合 A Attention、A Router、F Dense FFN、F Routed Expert、F Shared Expert、A weighted combine、A residual、encode/decode 和 layout work。Routed Expert 使用已经完成 Top-K 展开的逐 Expert assignment workload；Shared Expert 使用原始 token 数，并以扩大的 intermediate 表示多个 Shared Experts；weighted combine 保留每条 input edge 的 Expert、Top-K index 和 routing weight，再把 `T*K` 条 Routed partials恢复为 `T` 个 token outputs。Residual 只在 A 侧按原始 token 数计算一次。Manifest 的 `component_includes` 用于阻止 fused kernel 重复包含 softmax、SwiGLU、投影或 Router。Collective 和跨域 transfer 均不进入 compute sum。

通信 service time 为：

```text
fixed_latency + payload_bytes / effective_bandwidth
```

运行时通信 query 明确区分 `a_to_f`、`f_to_a`、`all_reduce`、`all_gather` 和 `reduce_scatter`。A-to-F 是 Routed dispatch 与 Shared input 的唯一跨域传输；F-to-A 只返回 Routed/Shared partials，数学 weighted combine 随后在 A 侧单独计时。每个方向一层只有一个 logical stage，但 planner 会按真实 producer、endpoint、stable tensor item 和 global hidden interval 展开独立 legs。Payload 来自逐 token assignments、Shared input 和 tensor ownership，不能由 histogram 或无 owner 的总 token 数猜测，也不会再生成搬运相同数据的 `ep_dispatch`/`ep_combine`。

Endpoint-leg query 携带实际 wire payload；provider 只应用链路固定时延和带宽。Collective query 携带每 rank logical input/output bytes、实际 participants、operation、fabric 和 lane；ring/tree 等 wire-volume 系数只由 provider 应用一次。Provider 不创建 placement、依赖或 queue time。全/半双工、lane capacity 和多个 AF blocks 的竞争由 executor 的 SimPy resources 产生。

五类 dtype 角色独立进入数据键和审计信息：GEMM/Expert 同时保留 weight 与 activation dtype，Attention 额外保留 KV Cache dtype，通信使用对应方向 dtype。传输 dtype 与 activation dtype 不同时，发送 encode 和接收 decode 是独立 compute queries，不隐藏在带宽服务时间中。SwiGLU 公式使用 gate/up/down 三个矩阵；MHA/GQA/MQA 使用实际 Q/K/V/O dimensions；Decode Attention 按每个 request 的 context 和 sliding window 分别计算后聚合。

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

示例 package 是正式 Decode 与 AF-Prefill compatibility 共用的超集，因此仍可包含 legacy F-side `residual_add` record。正式 Decode 的 `dense_ffn`、`routed_expert` 和 `shared_expert` composition 不会消费该 record；residual 只由 A-side `residual` operation计时。

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

完整机制 E2E 会真实生成并加载 Parquet，经 `LLMEngine`、scheduler、AF executor、资源竞争和 trace 完成 Decode AF。Decode-only 配置下 Prefill 保持既有完整模型 aggregate 路径；显式 AF-Prefill 使用 compatibility 路径，不宣称具有新的逐 rank 精度：

```powershell
python -m unittest tests.test_operator_data_latency_provider
```

测试覆盖 exact/interpolation/missing、Dense/MoE/Shared/Combine 的互斥组成、endpoint payload、collective wire volume、queue/service 分离和 trace 自洽。验收数字以当前测试命令的实际输出为准，文档不固化可能过期的通过数量。只有拿到独立 kernel、链路或真实系统 trace 后，才能报告物理准确性；否则结论必须写为“机制与模型自洽通过，物理准确性待实测”。测试中的 `measured:test` 和 example source 都是确定性 fixture，不是硬件 ground truth。
