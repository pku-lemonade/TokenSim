## 1. 模型与 schema

- [x] 1.1 扩展 `ModelShape` 的 MLA 字段并验证维度；用合法/非法 DeepSeek 配置单元测试验证错误信息。
- [x] 1.2 扩展 Attention operator-data schema 和 key 编码；验证旧 MHA/GQA/MQA manifest 仍可加载。

## 2. 延迟与执行

- [x] 2.1 实现 MLA 完整等长 prefill/decode analytical 公式及 source 标记；用固定 shape 检查 latency、FLOPs 和 KV bytes。
- [x] 2.2 接入 MLA exact lookup，理论数据离线生成、runtime 缺数据显式失败；验证不得静默回退 GQA，实测替换不重复添加固定开销。
- [x] 2.3 将 MLA Attention 接入现有 MoE A/F decode planner；运行 V3 形状单层 smoke test，检查 trace 阶段完整、TP cache 和权重账本正确。

## 3. DeepSeek‑V3 数据与验证

- [x] 3.1 添加与官方快照一致的 V3 元数据、前 3 层 Dense、58 层 MoE 和 shared expert；生成可加载 manifest；功耗放入独立硬件假设配置，synthetic 路由明确说明。
- [x] 3.2 离线生成 MLA operator-data 覆盖完整等长 prefill、decode batch 和假设设备；输出 measurement-contract 和待测 shape 清单。用户已同意实测稍后补充，本任务不声称完成设备采集。
- [x] 3.3 全量回归 341 项通过；后续 runtime/TP 专项 54 项、最终 MLA 11 项复测通过。完成 61 层、虚拟 1A32F、batch=1/4/16/32 的 decode 扫描，输出 `outputs/deepseek-v3-61layers` 的 trace、summary 和限制说明。



