## MODIFIED Requirements

### Requirement: Provider estimates are canonical and cacheable

Operator-data compute与transfer provider MUST 从影响service time、payload或provenance的全部query字段构造规范化identity。对相同identity返回的estimate SHALL 是不可变且可安全缓存的；estimate MUST 保留source id、dataset version、match type、component breakdown、payload bytes和适用的timing scope。缓存机制 MUST 不包含或复用queue、ready、start或end time。

#### Scenario: Cached operator estimate preserves provenance

- **WHEN** 相同规范化compute query第二次命中缓存
- **THEN** 返回与首次查询相同的service duration、source、dataset、match type与components
- **AND** executor仍为第二次执行独立计算queue duration

#### Scenario: Policy and shape changes miss the cache

- **WHEN** device/backend/fallback、dtype、phase、local shape、context、routing workload或collective participants改变
- **THEN** provider不把该query视为先前identity
- **AND** 按新query执行合法lookup或明确失败

#### Scenario: Compact keys preserve full query identity

- **WHEN** 内置 provider 使用静态指纹与动态 key 查询估值缓存
- **THEN** 紧凑 key 与完整规范化 query 在 service time、payload 和 provenance 上保持相同的区分能力
- **AND** source/dataset version、lookup policy、dtype、context、routing workload、participants 与 endpoints 的有效变化不得错误命中
- **AND** key 未命中时按需构造完整 query，兼容 provider 继续使用原完整 query 接口

### Requirement: Transfer estimates declare timing scope

Transfer与collective estimate SHALL 明确声明其service duration的测量边界。纯链路或设备transport时间 MUST 标记为`transport_service`；包含host dispatch、框架间隙或端到端eager interval的数据 MUST 使用不同scope。启用分段通信的executor MUST 仅接受`transport_service`。

#### Scenario: Analytical link estimate is transport-only

- **WHEN** provider通过link/topology alpha-beta模型估算AF transport
- **THEN** estimate标记`timing_scope=transport_service`
- **AND** host submit/completion只由executor配置另行计时

#### Scenario: Eager interval is not relabeled

- **WHEN** 导入数据的测量边界包含host或框架间隙
- **THEN** manifest/estimate保留`eager_interval`或等价非transport scope
- **AND** 不得为通过分段模式校验而改贴transport标签
