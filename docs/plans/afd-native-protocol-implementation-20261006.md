# Decode-only 原生协议实施记录

后续计算专项测试已于2026-10-06完成：使用GPU0/3，A/F聚合计算达到数量级一致，未拟合参数。详见 `afd-compute-alignment-20261006.md`；下文GPU占用和未启动测量描述保留为此前实施阶段的历史记录。通信、host及完整校准验收仍待完成。

## 范围与状态

用户确认本轮只校准 decode，暂缓完整流程对照，不修改安装中的 AFD 插件。原生 prefill 仍用于建立 KV/预热，但不参与校准，不能宣称实现了 A 本地 prefill。模拟以 context=129 开始31次 decode。前一轮原始结果保留在 `outputs/afd-1a1f-alignment-20261006/`。

已实现协议与纯计算组合修正、显式分离计时和运行态 readiness 注入接口；尚未完成 metadata、原生路由/stream readiness 回放、host profile、实测校准与留出验收。manifest 为本目录 `afd-native-protocol-manifest-20261006.json`，记录原始结果与原生源码哈希。

## 被测插件源码证据

远程源码根：`/home/youwei/autodl-tmp/af-workspace/src/afd-plugin/afd_plugin`。

| 位置 | 已确认边界 | 尚缺证据 |
| --- | --- | --- |
| `model_executor/models/qwen3_moe.py`，`_checkpoint_weight_roles`、`AFDQwen3MoeDecoderLayer.__init__` | A只构建Attention/norm和RemoteFFNProxy；FFN权重分配给F。当前不能在A本地执行完整prefill | 完整流程不在本次验收范围 |
| 同文件 `compute_ffn_output` | 调用完整 `self.mlp(hidden_states)`，包括gate/专家/数学combine，输出hidden | 内部kernel融合与各段耗时须profiling |
| `model_executor/models/deepseek_v2.py:119`，`RemoteFFNProxy._send_and_receive` | A每层先send_attn_output，再recv_ffn_output；当前无prefill本地分支 | CUDA stream及host实际ready时序 |
| `connectors/gpu/p2p.py:305`，`send_attn_output` | hidden先发送；只有调用者提供router_logits/input_ids时才额外发送；当前Qwen gate-on-F不传router logits | 逐消息原生采集确认 |
| `connectors/gpu/p2p.py:374`，`recv_ffn_output` | 返回hidden，尽可能复用输入buffer | 融合算子与buffer峰值实测 |
| `connectors/gpu/p2p.py:685`，`P2pNcclAFDControlPlane` | DP metadata通过独立control-plane的NCCL group传递，形状按stage缓存；不能假设每层发送一套metadata | 每step发送次数、序列化bytes与纯服务 |

以上来自本次只读SSH核查，源文件哈希见manifest；不是对上游最新提交的推断。GPU2被PID1305614、GPU3被PID1515380占用，本轮没有启动GPU测量、修改插件或停止这些作业。

## 代码行为

- `af.execution_protocol=vllm_afd` 要求decode-only、1A1F、BF16、无shared expert的all-MoE Qwen3。Router固定随当前被测插件在F，旧router_location默认值不改变该选择。旧协议默认仍是 `partial_return`。
- 原生DAG：A Attention/post-norm→hidden→F Router→grouped experts→F weighted combine→hidden→A residual。用完整token identity而非Top-K partial作为返回张量identity。
- 专家估值使用不含router/shared/combine的 `routed_experts` 契约，按真实histogram查询。F combine单独估值，避免沿用通用MoE表中范围不明确的组合。Q/K norm在A计入；F不再计A的norm或residual。实际kernel融合带来的差异仍未校准。
- `communication_timing.enabled=true` 是原生模式必需配置。目前手工默认host_submit=5微秒只是假设，报告明确为uncalibrated。
- `TransferReadiness` 提供当前调用的sender_stream、receiver_stream和receiver_posted事件；executor先等待这些依赖，再申请fabric。trace分别输出readiness_wait_duration、queue_duration、service_duration。事件不进入估值缓存。
- 没有readiness提供者时，trace标记 `dag_dependencies_only`：只覆盖模拟DAG已知依赖，不宣称已回放真实对端等待。这是尚未校准的边界。
- DAG缓存原有配置指纹自然包含新增协议字段；provider按协议构造且保持不可变，修改协议须新建runtime/provider；运行时事件每次独立创建。

## 本地验证

`python -m unittest tests.test_af_native_protocol -q`：7项通过，包含Top-K=1/2的数据量、加权edge oracle、F combine位置和返回依赖、Q/K norm、协议自动Router、非法配置、缓存等价、stream晚到不污染service以及跨环境事件拒绝。

旧路由、并行规划、通信计时、缓存专项32项通过。沙箱内两轮全量测试（437/439项）均有36项临时目录权限错误。经自动审批后在沙箱外重跑，439项全部通过；日志为 `outputs/native-protocol-tests-unsandboxed.log`，临时目录仍限定工作区。没有为通过测试而修改临时目录逻辑。

本地命令：`python -m scripts.run_tiny_afd_alignment --output outputs/afd-native-protocol-20261006 --batches 1 8`。结果为未校准纯解析对照：batch1约189.63微秒，batch8约191.61微秒。两层每方向data分别1024/8192 bytes。该结果尚未对齐原生毫秒级整步，不能作为校准完成证据；完整metadata和CPU引擎开销仍缺失。

## 待完成

1. 原生trace补齐未匹配kernel、消息/路由/stream就绪关联，以及metadata控制平面。
2. GPU空闲后补同款PyNccl服务测量；不能将含host/peer wait的NCCL区间直接导入。
3. 根据证据定义并接入非重复计费的host profile，冻结预测规则后验证batch4。
4. 完成剩余规范场景、取消与竞争专项和性能统计。OpenSpec未完成项保持未勾选；不归档、不提交。
