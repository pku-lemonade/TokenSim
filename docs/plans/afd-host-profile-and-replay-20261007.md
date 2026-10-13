# 原生 AF：本地实现与剩余实测边界

本轮完成通用入口的 host profile 加载、严格路由回放和计费覆盖校验。没有生成实测 profile，没有用整步差值拟合参数，也没有宣称完整 decode 已达到数量级目标。

## 代码变化

- 标准 LLMEngine 的 `vllm_afd` 使用 ModelAFExecutor，保留 embedding、最终 norm、lm_head、token_selection。此前 tiny Qwen3 的 GQA 会走仅逐层执行器，漏掉模型边界。
- `af.engine_overhead_profile` 为内嵌 JSON 对象；`af.calibration_environment` 是本次运行明确提供的软件/执行环境清单，必须与 profile 完全一致。profile 在 executor 初始化验证，在 submit 验证 batch/context/phase；不会在超范围时静默使用默认值。
- `af.routing_replay` 为内嵌 JSON 对象。按 decode_step/layer/request 建索引，复用现有 RoutingAssignment/LayerRoutingSnapshot 验证 token、top-k、专家和权重；不修改原请求，不扫描历史 step，不允许缺项后切换到合成路由。
- 同名模型运行计划、设备、链路或拓扑参数变化使 runtime 指纹变化。profile 内容变化隔离纯服务缓存；DAG 模板已经按不可变 AF 配置隔离。等待仍由每次调用的事件/资源竞争生成。
- 原生模式中的实测算子表必须在包 `meta.extra.timing_scopes[table]` 声明 `device_service`。`routed_experts` 的 `meta.extra.operation_coverage.routed_experts` 可为 `["routed_experts"]` 或 `["routed_experts","weighted_combine"]`。后者将 combine 保留为零服务、零额外host提交的语义完成点，依赖仍保留；analytical fallback 不继承表中的融合声明。不得盲目剥离设备固有 launch。

## Host profile v1

必需根字段：`schema_version=1`、`profile_id`、`unit=us`、`timing_scope=host_service`、`contains_wait=false`、`excludes_device_launch=true`、`protocol=vllm_afd`、`runtime_fingerprint`、`environment`、`batch_range`、`context_range`、`costs`、`evidence`。未知字段拒绝。

`environment` 至少声明 `execution_mode=eager`、`software_sha256`；版本内容应由隔离插件与 vLLM 源码清单生成，不能拿模拟器软件版本代替。

每项成本格式为 `{fixed_us, per_token_us, coverage:[唯一操作名]}`。允许部分覆盖，其他项沿用明确记录的未校准配置。可覆盖：

| 成本字段 | 唯一覆盖 |
| --- | --- |
| a_step_prepare_us | a_input_prepare |
| f_step_prepare_us | f_loop_prepare |
| compute_submit_us | compute_dispatch_excluding_launch |
| host_submit_us | send_submit |
| host_receive_submit_us | receive_submit |
| host_completion_us | data_completion |
| control_encode_us | control_encode |
| control_decode_us | control_decode |

profile 成本替代隐式默认值，不做叠加；手工显式指定同一字段（即使等于默认 5µs）仍报冲突。设备 H2D/回读不属于 host_service，继续保留独立未校准配置，不能塞入这个 profile。compute_submit 按现有 DAG 计算提交边界收费，不冒充原生逐 kernel host 调用采集。

`evidence` 必需：`source`、`kind=measured|synthetic_test`、`sample_count`、`window_count>=3`、`training_batches`、`heldout_batches`、`method=constant|nonnegative_linear`、`residual_us`、`quantiles_us`、`validation_status=pending|heldout_passed|heldout_failed`。quantiles_us 按每个成本保存 `{p50,p95}`。首轮 batch4 留出，不能用于训练；线性模型至少三个不同训练 batch；固定模型的规模项为零。参数、残差、分位数及适用范围内的最大预测值必须有限非负。

加载后的状态始终为 `host_profile_loaded_not_full_engine_validation`：加载成功、来源自称 measured 或某个 host 留出通过，都不能代替完整引擎验收。仓库测试使用 synthetic_test，不提供假实测数值。

## 路由回放 v1

根字段：`schema_version=1`、`runtime_fingerprint`、`environment`、`source`、`kind=captured|synthetic_test`、`assignments`。

每条 assignment：`phase=decode`、`decode_step`、`layer_id`、`request_id`、`token_id`、`top_k_index`、`expert_id`、`routing_weight`、`a_dp_rank=0`、`a_tp_rank=0`。不存模拟 iteration_id；由运行时重新绑定。decode token_id 等于当前 context_len-1。请求 batch 必须与当前 step/layer 采集记录一致，不能拿 batch8 的子集冒充 batch1 原生运行。

仅支持 native control 开启的一块、单 in-flight eager 1A1F。真实采集尚未完成；synthetic_test 只能验证回放机制。加载回放仍不代表消息 API/stream 时间线已对齐。

## 使用

先导出当前运行指纹（不启动模拟）：

```powershell
python -m scripts.run_tiny_afd_alignment --output outputs/afd-runtime --batches 1 8 4 --describe-runtime
```

有合格实测资料后，可加载 profile、环境与匹配 batch 的路由文件：

```powershell
python -m scripts.run_tiny_afd_alignment --output outputs/afd-profile --batches 1 --host-profile host-profile.json --calibration-environment environment.json --routing-replay routing-b1.json
```

CLI 未指定的成本不写入配置，避免默认零被误判为手工覆盖；显式成本依然接受严格冲突校验。输出保留 host_profile/routing_replay 来源与指纹，同时整体 calibration_status 仍为 uncalibrated，因为设备、传输和完整引擎验收未结束。

## 验证与待测

专项测试覆盖标准 engine 模型边界、默认/手工冲突、错误 scope、NaN/负数/溢出、同名链路带宽变化、profile 内容缓存隔离、精确路由权重与 histogram、缺项/重复/错误 token/expert、缓存及 trace 开关等价。

CPU 开销脚本：`python -m scripts.benchmark_afd_local_overhead --output docs/plans/afd-local-overhead-20261007.json`。分别比较控制消息开/关、trace 开/关；wall 取三个重复窗口，tracemalloc 单独运行。该脚本只评估 Python 实现开销，不测 GPU，也不作为通信校准依据。

仍须实测或进一步实现验收：真实逐层路由采集、精确 send/recv API 与 stream 边界、通信双方 ready 的纯服务、同窗口 host/device 归因与 profile 参数、原生 buffer 生命周期/峰值显存核对、batch1/8/4 各三独立窗口以及完整 step 对照。OpenSpec 中这些整项继续未勾选，不能将局部实现等同于全部完成。

### 本次验证结果

- 全量 `python -m unittest discover -s tests -q`：479项通过，130.439秒。沙箱内两次运行均有36项Python临时目录权限错误；改用沙箱外同一测试命令后通过，临时目录仍为仓库outputs/test-tmp。原始日志：outputs/afd-local-regression.log。
- 全量运行后补充的native边界失败恢复/KV归属用例通过；最后scope覆盖/trace来源改动的专项复验通过。AF相关回归262项通过（该次尚未包含最后新增的1项）。
- 完整模型驱动batch1执行31步通过，`--describe-runtime`指纹导出通过；OpenSpec strict校验、diff空白检查和新增模块编译通过。
- CPU基准无其他本任务测试同时运行，31步、batch8、每点三次wall测量；表中内存为独立tracemalloc运行的Python分配峰值，MB按十进制。

| 控制协议 | trace | wall中位数(ms) | Python峰值(MB) | 保存事件 |
| --- | --- | ---: | ---: | ---: |
| 关闭 | 关闭 | 201 | 2.03 | 0 |
| 开启 | 关闭 | 251 | 1.98 | 0 |
| 关闭 | 开启 | 285 | 6.79 | 868 |
| 开启 | 开启 | 331 | 8.34 | 1395 |

控制协议增加约17条trace事件/步；wall增加约17%（trace开启）或25%（trace关闭）。这是本机小模型实现开销，不外推到长时间、大规模负载；内存小幅波动不解读为优化收益。trace开关前后模拟step完全一致；控制消息开启本身增加模拟服务是预期行为。数值来源：docs/plans/afd-local-overhead-20261007.json。
