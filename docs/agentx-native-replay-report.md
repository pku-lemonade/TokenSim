# AgentX 执行回放与取消生命周期

日期：2026-09-30；分支：`agentx-support`。本次接入原始请求图的执行回放，完成预热交接、依赖、idle clocks、宽限和取消的逻辑及引擎集成。完整模型的混合缓存、两种卸载和耗时路径仍待实现，B200/B300 正式性能验收未完成。

## 修改与运行语义

`NativeAgentXReplay` 消费上一部分核对过的请求图和 snapshot。先发必要 primer，然后每通道再发 10 条压力预热请求，max_tokens=1；压缩 idle 等待，保留跨流前驱和 SPAWN/JOIN。发满配额后停止压力请求，已发请求排空；1800 秒预热 grace 从停止发送后开始，未作为总预热时长上限。错误和取消分别计数。

正式测量接续各流已推进的游标，按该树原始 timestamp 减 sampled origin 恢复下一请求等待，并减全局最小 offset。已发过的压力请求不重发；同一 continuing tree 的 cache scope 跨阶段不变。树完成后从共享顺序池领取下一条，重建独立游标和 cache scope；prompt/hash/OSL 始终来自原始 trace，未追加模拟输出生成新输入。

单树无在途请求时启动 300 秒 watchdog；到期推进该树下一组 pending timer，依赖仍未就绪时继续处理下一等待链。系统整体 idle 使用独立 10 秒 cap，统一移动待发 timer，保留相对间距。有在途请求不作系统 idle 压缩。阶段切换和停止发送后禁止旧 timer 发新请求。

发请求窗口为 3600 秒，之后停止新请求、最多等待 30 秒，再取消未返回请求，最多等待 10 秒取消确认。宽限期成功请求进入指标；错误／取消保留在状态计数与逐请求记录，未计入成功请求的延迟。吞吐使用成功请求最早 start 和最晚 end，不固定除以 3600/3640。没有成功请求时仍可导出正常到期结果，缺失延迟为 null，不伪造成零延迟。

新增引擎取消接口、Request 取消标记和确认事件。取消不将 generation_idx 补成预期 OSL，不被视为成功；已取消请求的迟到 step 不采样、不记录成功。CANCEL 优先于新 ADD；worker 在当前 kernel/load 完成边界清理自己持有的队列和块，再确认。测试验证正在执行的请求在取消后仍持块，到 kernel 结束才释放。未完成 prompt/output 不以完整缓存键发布；所有 owner 确认后才完成取消，重复取消与迟到回调不重复计数或释放。

原始 64-token hashes 按 4 个一组聚合为完整 256-token engine block，保留前缀链和 tree scope；不足一个 engine block 的尾部不伪造 hash。请求终态释放重的逻辑块和 hash 列表。逐请求／详细事件可用流式 sink 输出；日志开关不改变轨迹。

`benchmark.py` 的完整 DSV4 AgentX 分支已选择新图与新回放，强制全部 393 条 trace、无 skip、256 block、预热和 3600 秒窗口，预加载输入也不能绕过检查。旧通用 workload 的回放保持原接口。矩阵脚本迁移、proxy 删除和原生缓存仍待下一部分。

## 验证与数据分析

新增 17 项边界／引擎集成测试，完整测试 **316 项通过**。覆盖 queued/running 取消、kernel 持块、无迟到采样、grace 成功、取消超时与迟到回调、预热交接无重发、历史 API 只推导等待、两个 idle cap、错误前驱完成、日志不改变轨迹，以及缺失指标。

同时执行全部 393 条 trace、98,827 条原始请求图的生命周期诊断，两个主并发均完整推进 3600 秒发请求窗口及排空。该诊断使用 **0.1 秒受控 transport，没有执行硬件／算子模型，不是正式性能验收**；只检查真实数据中的依赖链、配额、游标和停止边界，不报告模拟吞吐误差。

| 生命周期诊断 | c196 | c384 |
|---|---:|---:|
| warmup 发出／成功 | 2171 / 2171 | 4249 / 4249 |
| profiling 发出／成功 | 35758 / 35758 | 71205 / 71205 |
| 发请求窗口（秒） | 3600 | 3600 |
| 包含末次完成的阶段 elapsed（秒） | 3600.014000 | 3600.087433 |
| tree idle 推进次数 | 824 | 1574 |
| 正式硬件性能验收 | 未执行 | 未执行 |

受控 transport 不模拟参考的错误和实际服务速度；因此诊断计数不要求匹配实测 completed 或 idle 次数。所有阶段 `sent = successful + error + cancelled` 守恒，未出现等待死锁。请求图核对中的 211/409 primer，加每通道 10 条压力请求，确实产生 2171/4249 条预热发送。

对完整池运行进行 10 秒 CPU 采样，发现 `_future_timers` 占约 32%、`_issue_ready` 约 18%、`_ready` 约 15%：反复扫描大量 terminal/unspawned 会话。改为活跃游标集合与未完成计数，并保留原始发出顺序。优化前后的两组完整窗口在 phase counts、trace/原始请求覆盖、回收顺序、elapsed、idle 次数与跳过秒数上 **逐项完全相同**。证据在 `data/reference/agentx-replay-logic/`；完整运行、采样和日志在 `results/agentx-dsv4-full/replay-logic/`。

## 尚未完成

真实的一小时模拟仍需要 DSV4 单份压缩 KV、SWA/indexer 状态与共享物理池，解决 B200 的错误容量准入；随后接入实际 SimpleCPU/Mooncake 传输、DP 同步、投机进度与模块耗时。当前旧的 uniform cache 和 GQA 耗时路径不能作为完整 DSV4 性能证据。参考 worker/time 固定诊断、完整矩阵和最终三种子验证也继续保持未完成，不用本次受控生命周期诊断替代。
