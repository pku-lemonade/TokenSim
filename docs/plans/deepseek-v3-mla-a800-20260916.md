# AutoDL A800：DeepSeek-V3 单层 TokenSim 测试

沿用 [两卡方案](deepseek-v3-mla-1a1f-20260916.md) 的模型和扫描范围，改在 `AutoDL-A800`、账号 `songhan` 上执行。

用户授权：可以中断其自己的任务，不中断其他用户任务。2026-09-16 已核对旧 `qwen3-moe-tiny-random` vLLM 服务属于 songhan，对服务主进程 254204 发送 SIGTERM 后正常退出，两卡均恢复 0 MiB。没有启动新的模型服务。

## 配置

- 两张 NVIDIA A800-SXM4-80GB，各 81920 MiB，NV8 互联。
- A：物理 GPU 0，`GPU-dfe6761a-23f4-c34c-9c03-01da02020e5f`。
- F：物理 GPU 1，`GPU-ce81ab13-d8a0-a49d-c908-f876b8eb2087`。
- V3 一个 MoE transformer 层，BF16、合成权重、uniform top-8 路由；不加载完整 671B 模型。
- 本轮显式设置 `router_location=f`，Combine 在 A。旧通用脚本 configuration 的默认 Router A 保持兼容，本轮调用显式覆盖并核查 trace。
- 功耗沿用可修改的固定假设 250 W/卡、500 W 合计。它不是 A800 实测平均功耗，也不是其 400 W 功率上限。

## 测量及输出

远程根目录：`/home/songhan/autodl-tmp/af-workspace/profiling/deepseek-v3-mla-a800-20260916`。

1. 冒烟测试：11 类算子通过数值检查及 GPU profiling。MLA 使用 FlashInfer FA2 压缩路径；只调用独立算子库，不启动 vLLM 引擎。
2. 正式算子：114 个曲线所需精确形状，另加 4 个未参与拟合的 MLA holdout。A 卡采 106 点，F 卡采 12 个专家形状。F 的 Router GEMM、norm 使用同型号 A 卡的精确形状数据，未声称每个算子都在实际执行卡上单独测量。
3. 通信：从模拟 DAG 提取真实 payload，14336 至 458752 字节共 6 档；分别测 A→F、F→A、同时双向，共 18 点。F→A 按专家分别返回部分结果，另有一个 shared partial 传输；不是每个小批次固定 8 次返回。当前均匀路由下 AF=8 激活 64 个专家，一个小批次有 65 次返回。batch=32、AF=8 共 4 次 A→F、260 次 F→A。保留现有模拟器的传输粒度，不假定已合并消息。
4. 所有正式点使用 eager、预热 20 次、3 个至少 2 秒且至少 100 次的窗口。计时关闭 profiler，另捕获 trace。P50/P95 是 100 次调用均值的分布，不是请求尾延迟。
5. `compute-A/traces/`、`compute-F/traces/`、`communication/traces/` 是真实 GPU profiling；`simulation/traces/` 是模拟时间轴。
6. 通信使用 PyTorch P2P copy、独立 stream 和同设备 event；不是 NCCL，也不是完整单层 GPU 驱动。保存双向同时执行测量用于检查 full-duplex 假设，不把拓扑标签直接当作带宽。

## 曲线

- 流水扫描：总 batch=32，AF 小批次 1、2、4、8、16、32，context 4096/16384，共 12 点。
- 吞吐—交互速度：AF 上限固定 8，并发 1、2、4、8、16、32、64、128，context 4096/16384，共 16 点。
- x=单层等效每用户速率=1/整批单层时长；y=并发×x/0.0005 MW。图中标注并发度。
- 另保存 batch=32、4K 的不拆分与最快 AF 小批次模拟时间轴。

所有 runtime 计算形状必须有精确实测覆盖，缺点即停止，禁止静默填 Roofline。MLA 的 `h + multiplier × Roofline` 拟合单独报告 holdout 误差；绘图的已测点不使用该拟合替换实测值。通信使用非负固定延迟+bytes/bandwidth 拟合，保存误差和带宽能否辨识。

## 结果边界

这是实测算子和通信校准后的 TokenSim 单层曲线，不是两卡完整 DeepSeek-V3 推理，也不是原生双卡层驱动的端到端对照。专家 kernel 使用当前默认配置，尚未调优。通信与计算并发争用及全局 CPU 发射线程限制未完整校准，eager 算子区间含 host gap。吞吐只能称单层等效速率。

后续流程由 `finish_mla_a800_run.py` 等待本轮特定测量 PID 正常结束，再顺序执行通信和 CPU 模拟。它不监控他人任务、不抢占新任务、不创建常驻占卡等待进程。任何测量失败会停止流程，并写入 `pipeline-status.json`。
