# AF 通信：提交、传输、同步等待分开计时

一次通信按以下依赖执行：

```text
生产者完成 → CPU 提交排队 → CPU 提交 → 链路排队 → 数据传输
                                                    ↓
消费者可运行 ← 完成通知（可选 CPU 服务）← 传输完成事件
```

CPU 提交结束即释放 host 资源。数据传输只占链路资源，CPU 可以继续提交后续消息。消费者依赖传输完成，不占着 CPU 或链路空等。默认采用事件驱动完成；如果目标后端忙轮询或阻塞整个提交线程，需要另外建模，不能把其时间重新塞入链路常数。

## 当前默认实验假设

配置在 `data/operator_data/af-communication-separated.json`：

| 部分 | 设置 | 占用资源 |
|---|---|---|
| 每条消息提交 | **5 μs**，可配置，未实测校准 | 源端 host |
| 设备传输启动常数 | **0 μs**，暂不额外假设，后续可补测 | 链路 |
| 数据传输 | `payload_bytes / 25 GB/s`，带宽为假设 | 链路 |
| 完成通知额外服务 | **0 μs**，可配置 | 目标端 host；为 0 时不占资源 |
| CPU/链路排队与同步等待 | 调度和依赖实际产生 | 不额外添加固定服务时间 |

5 μs 只收取一次，放在 CPU 提交。两个 0 表示当前额外开销未建模，不代表硬件测得零开销。25 GB/s 沿用之前的理论假设，不是 A800/NVLink 标称或实测带宽。

不考虑重叠的单条消息完成时间为：

```text
提交排队 + 5 μs + 链路排队 + 设备启动常数 + 字节数/带宽
         + 完成通知排队 + 完成通知服务时间
```

`dependency_wait_duration` 记录提交完成后等到传输完成的区间，已经包含链路排队和传输，**仅用于解释等待，不能再加到上式**。消费者在生产者完成之前的依赖等待同样由 DAG 控制。

## 模拟配置与兼容性

AF cluster 可开启：

```json
{
  "af": {
    "communication_timing": {
      "enabled": true,
      "host_submit_us": 5.0,
      "host_completion_us": 0.0,
      "host_resource_scope": "per_worker"
    }
  }
}
```

`per_worker` 表示每个物理 worker 各有一个通信提交线程；`shared` 表示所有端点共用一个提交线程。它们只控制通信提交/完成服务，不代表完整推理引擎所有 CPU 操作。多 rank collective 在各参与者 host 上并行提交，所有提交结束后才申请链路；完成通知同理。

旧配置默认关闭分段模式，以保留既有 provider 行为。本轮 `run_mla_measured_curves.py` 默认开启并读取上述 JSON。负数、NaN、无穷和布尔型耗时会被拒绝。

分段模式要求 provider 明确声明 `details.timing_scope="transport_service"`。算子包通信表支持同名可选列：缺失为 `legacy_unspecified`，包含主机间隙的测量应为 `eager_interval`。后两者不能用于分段模式，会立即报错；不能只给旧 51/66 μs 数据重新贴标签来绕过边界要求。

新增事件是 `host_submit`、原有 `transfer/collective`、`synchronize`，分别保存各自排队和服务时间。原 DAG 节点 ID 由完成事件持有，保证下游等到真正完成才执行。网络字节和 tensor slices 只在传输事件上计数。CPU 提交、链路和完成事件全部参加依赖与资源容量检查。

## A800 复跑

```powershell
python scripts/run_mla_measured_curves.py --root outputs/deepseek-mla-a800-20260916
```

输出另存 `simulation-separated/`，保留旧数据。使用已有 114 个精确算子形状，不重新占 GPU。通信 51/66 μs 拟合和旧双向测量只保留为排除的诊断数据，不参与新通信计时。

初始设置下，4K、batch=32 的 AF=32 / AF=8 时延约 15.28 / 13.72 ms；16K 对应约 16.66 / 14.04 ms。这些是通信参数假设下的单层模拟，不是重新完成通信校准或真实两卡端到端验证。MLA 等计算表仍为 eager 区间，计算侧主机间隙及全局 CPU 发射竞争尚未完全分离。

测试覆盖提交/传输重叠、CPU 与链路各自排队、完成服务归属、半双工、共享 host、零开销兼容、拒绝旧口径，以及正式 MLA TP collective 完成依赖。
