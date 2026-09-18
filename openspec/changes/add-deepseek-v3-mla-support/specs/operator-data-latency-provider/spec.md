## ADDED Requirements

### Requirement: dedicated optional MLA tables
系统 SHALL 在 schema v2 上增加可选 MLA core 和 grouped GEMM 表，保持原有普通 Attention 数据包兼容。

#### Scenario: missing MLA record
- **WHEN** MLA core 的设备、dtype、执行方式、context 向量或 latent/head 维度没有精确匹配
- **THEN** runtime MUST 报告缺失 shape，不回退普通 Attention，不现场调用理论公式

#### Scenario: offline measured precedence
- **WHEN** 同一 MLA shape 同时有公式候选与合格实测观测
- **THEN** 生成器 MUST 选择实测值，不再叠加理论固定开销，并保留来源；exact 表示 shape 命中而非实测证据

#### Scenario: grouped projection timing
- **WHEN** MLA 有 H 组独立逐头矩阵乘
- **THEN** grouped GEMM SHALL 按 H 组工作量建模但只计一次启动开销，不引入跨 head 的无效矩阵乘
