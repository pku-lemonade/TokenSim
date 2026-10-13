# 实测固定成本接入与测试

本轮完成两项原生CPU codec固定成本的补测和实际接入，不是完整host/通信校准完成。

## 测量与参数

在PKUa100执行原始隔离插件的 `encode_control_payload` / `decode_control_payload`，使用真实AFDDPMetadata CPU tensor及179-byte原生消息，逐次验证round-trip。编码的输入tensor明确为CPU；解码只创建CPU tensor。不包含send/recv、设备staging、读回或调用者包装成本。CUDA_VISIBLE_DEVICES为空，六个进程均断言CUDA未初始化。

batch1/8各三个独立进程窗口，每函数200次预热；每batch/window/function在129–159 context范围采集124次，共1488个原始调用样本。context不是codec输入，按既有decode支持范围标注，不声称测到了context规模效应。采样使用perf_counter_ns，含普通计时及OS调度噪声；按各窗口中位数等权求常数。此为隔离CPU函数微基准，尚未经完整引擎内的低扰动复验。

| 参数 | 实际代入值 | 全样本p50 / p95 |
| --- | --- | --- |
| control_encode_us | 6.163 µs | 5.795 / 8.727 µs |
| control_decode_us | 14.9714167 µs | 13.858 / 19.220 µs |

参数文件 `afd-measured-codec-profile-20261007.json` 保存来源指纹、训练/留出范围、统计和pending验证状态。原始小文件在 `outputs/afd-cpu-codec-fixed-20261007-v1/`；源码SHA与既有原生采集manifest一致。测量脚本 `scripts/measure_afd_cpu_codecs.py` 不修改原插件。

## 实际接入结果

经既有通用ModelAFExecutor profile入口加载，沿用已采集的31步真实路由。每个batch的编码和解析分别恰好31个计费事件，逐事件service核对与参数一致。没有在报告总数上补常数。

| batch | 实测接入前模拟 | 接入后模拟 |
| --- | --- | --- |
| 1 | 229.595529 µs | 235.758529 µs |
| 8 | 231.262598 µs | 237.425598 µs |

解析服务在当前模型中与A计算重叠，所以整步只增加编码的6.163µs，不是两项求和。这只验证现有模拟依赖下的结果，不替代原生调用边界验证。真实引擎此前仍为3.36–3.88ms；主要差额未解决。

修复trace中profile生效仍被硬编码标为未校准的问题：现输出每项生效值、实测/测试/手工/默认来源、覆盖、profile指纹与验证状态。对照驱动标为partial_host_calibration；默认0继续标为unmeasured_default，不能误认为测得0。calibrated字段指该服务参数有实测来源，不表示整步验收通过。

## 测试与剩余工作

- 完整AF相关274项回归通过；新增实测来源、唯一计费和未测零值标记回归。
- 两组真实路由回放逐项核对编码/解析31次计费；保留同配置代入前后结果，JSON摘要见 `afd-applied-fixed-costs-20261007.json`。
- Python编译、OpenSpec严格校验及git diff检查通过。
- 尚未代入的host项：A/F准备、compute submit、send/receive submit、completion。设备staging/回读和纯通信参数也未校准。已有inclusive测量不能证明这些独立服务的数值，不能直接拿来补齐profile。
- 总等待新口径、原生提交/完成依赖、完整同窗口CPU/GPU关联仍待验收。此次没有修改等待调度或宣称这些任务完成。
- batch4留出本轮未执行；当前只有codec局部profile，后续应先完成必要固定成本覆盖再做完整冻结验收。OpenSpec整体保持16/25，4.4/4.5因覆盖不全仍未勾选。
