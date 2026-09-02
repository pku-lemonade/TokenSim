## MODIFIED Requirements

### Requirement: Invalid or unsupported AF configuration fails before simulation

系统 MUST 在启动模拟事件循环前完成 AF 字段、阶段、拓扑、link 和共享资源校验，并使用项目统一的配置错误类型报告可操作的字段或对象上下文。合法且已启用的 AF 配置在 executor 与所需 latency providers 可用时 SHALL 进入 AF 执行路径；若运行环境缺少 executor、provider 或必要模型元数据，系统 MUST 在产生 AF 事件前明确报告缺失依赖，不得静默退回非 AF 完整模型时延。

#### Scenario: Configuration has multiple AF errors

- **WHEN** 配置同时包含非法容量和缺失的阶段 link
- **THEN** 系统在模拟开始前拒绝配置
- **AND** 返回的错误信息至少标明一个可操作的字段或拓扑问题

#### Scenario: Valid AF configuration reaches a supported runtime

- **WHEN** 合法且已启用的 AF 配置被交给具有 executor 和所需 providers 的运行入口
- **THEN** 系统创建对应阶段的 AF 执行路径
- **AND** 系统不再以“AF executor 尚未实现”为由拒绝运行

#### Scenario: Valid AF configuration reaches a runtime without AF executor

- **WHEN** 合法且已启用的 AF 配置缺少 executor、provider 或必要模型元数据
- **THEN** 系统在产生 AF 事件前拒绝运行并标明缺失依赖
- **AND** 系统不得使用 Roofline、LLMCompass 或其他完整 iteration 后端静默模拟 AF
