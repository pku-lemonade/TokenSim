# 完整操作固定成本已接入

本次完成9.1–9.3：独立operation_service_profile、通用执行器/标准引擎接入、事件替换、真实路由回放和逐步计费核对。整步校准仍未通过，4.4/4.5及正式留出验收保留未完成。

## 实际加载参数

| 完整操作 | 固定成本（µs） | 每步次数 | 覆盖 |
| --- | ---: | ---: | --- |
| h2d | 38.1689167 | 1 | bytearray/frombuffer、内容H2D、长度tensor构建及自身完成 |
| length_readback | 10.946 | 1 | 已就绪长度的item() |
| payload_readback | 19.3659167 | 1 | 已就绪内容的cpu().numpy().tobytes() |

参数文件 `afd-operation-profile-20261007.json`。每项1200样本，batch1/8各三窗口、每窗200样本；六组中位数等权求均值。原始 `outputs/afd-control-staging-20261007-v1/window-*.json` 的混合scope和“不适用于纯host/device”标记未改动；新产物声明完整操作scope、覆盖、来源内容指纹、统计和适用域。

既有codec encode 6.163µs、decode 14.9714167µs独立收费。参数绑定PKUa100 GPU2/3、tiny Qwen3、eager1A1F、179-byte消息、batch1–8/context129–159，不是跨模型硬件默认值。

## 完成边界与调度

原始微基准在计时外synchronize，排除前序GPU工作。H2D使用默认blocking拷贝和CPU数据构建长度tensor；读回返回CPU可见值。实机边界检查在每次API返回后立即query当前stream，再校验数据：batch1/8三项各10次，共60次全部通过。该检查不拟合时间，不导入query/校验成本。

脚本 `scripts/check_afd_operation_boundaries.py`，本地证据 `outputs/afd-operation-boundary-20261007.json`，远端独立目录 `/home/youwei/autodl-tmp/af-workspace/profiling/afd-operation-boundary-20261007-v1`。核对无重复校准进程后启动，脚本再次检查双卡空闲；不使用网络端口，未改原插件或覆盖实验。

完整操作占用worker调用线程，替代旧control_device事件；不额外计copy/内部同步，不独占整张GPU或fabric。staging在发送前；长度读回在长度收到后/内容接收前；内容读回在内容收到后/解析前。外部等待由依赖和资源竞争生成，不缓存、不混入参数。

独立copy-engine争用仍未模拟，当前是单in-flight调用线程近似。不启用profile保留旧device聚合近似；显式手工值即使为0也拒绝同覆盖并存。profile内容进入计算/传输缓存隔离；静态配置进入计划身份。operation_service单列，不混入纯CPU/GPU忙时。

## 真实路由回放

| batch | 仅codec校准（µs/步） | 新增完整操作（µs/步） | 历史实测三窗口（µs/步） |
| --- | ---: | ---: | --- |
| 1 | 235.758529 | 284.873446 | 3360.46–3794.61 |
| 8 | 237.425598 | 286.540515 | 3575.00–3883.86 |

每组同环境、同codec、真实路由31步。`scripts/audit_afd_operation_replay.py`确认每步三项operation各一次、codec各一次、物化成本与profile相同、没有legacy重复事件；证据见同名JSON。operation关键路径均49.114917µs，小于三项总和68.480833µs，因为内容读回与A计算重叠；不能把服务总和加在step末尾。

校准后与历史实测仍差约11.8–13.6倍，未达到小于10倍目标。本次实机只检查完成边界，没有重新测完整引擎step；batch4未运行，留出参数未冻结。

复现（batch8替换相应batch与replay文件，使用新输出目录）：

```powershell
python -m scripts.run_tiny_afd_alignment --output outputs/new-operation-b1 --batches 1 --routing-replay outputs/afd-live-capture-20261007/replay-b1.json --calibration-environment outputs/afd-live-capture-20261007/environment.json --host-profile docs/plans/afd-measured-codec-profile-20261007.json --operation-profile docs/plans/afd-operation-profile-20261007.json
```

## 验证和缺口

新增10项测试覆盖范围/冲突/非法值、标准LLMEngine、唯一计费、旧行为、缓存/trace等价、100+11→111及输入延迟不改常数、调用线程串行且GPU/fabric不独占、取消释放和拟合校验。285项AF回归通过后，补充两项测试，operation专项10项通过；strict validate通过。

最终按模块独立进程完成51个模块、504项测试，全部通过，每个进程退出码均为0；摘要见 `afd-operation-regression-20261007.json`，日志在 `outputs/operation-regression-test_*.log`。单进程全量尝试未留下完整摘要，未计为通过；相关AF测试在同一进程的285项回归另行通过。硬件/角色校验补强后，原始样本重新拟合与保存profile完全一致。

仍缺A/F准备、算子/消息提交及输出处理的可归因成本、无peer等待的transport服务和原生调用边界。不能直接使用旧cProfile的A `_prepare_inputs` 652.92µs：vLLM gpu_model_runner.py中1956提交block table拷贝，2055/2082进行H2D，2118有GPU fill，2152–2154明确non_blocking拷贝。CPU返回不保证全部设备工作完成，wall与已建模设备工作可能重叠。需先明确纯host子段，或扩展完整操作的覆盖/完成边界并去除重复设备成本；首批三项白名单没有覆盖整个输入准备。不能用整步残差代替该缺失参数。

边界检查已退出；监控保持先前暂停状态，未占用或预留GPU。
