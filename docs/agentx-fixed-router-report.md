# Fixed historical worker routing diagnostic

The reference router logs were joined to AIPerf metadata by
`metadata.x_correlation_id` and `header:x-session-id`. The resulting mappings
cover 2,165/2,165 B200 conversations and 4,024/4,024 B300 conversations.

`FixedWorkerRouter` uses the native conversation ID as the lookup key, records
mapped and missing requests, and falls back to the audited consistent-hash
router only for an unknown key. Both full one-hour diagnostics had zero missing
keys:

| SKU | output token/s | mapped requests | missing keys |
|---|---:|---:|---:|
| B200 c196 | 524.650 | 2,542 | 0 |
| B300 c384 | 1,037.582 | 6,122 | 0 |

The runs passed the same final-cache, cancellation and DP-time validator as the
formal checkpoints. They remain routing diagnostics; the formal curve uses the
closed-loop consistent-hash router. Evidence is in
`results/agentx-dsv4-full/diagnostics-fixed-router/` and the mapping provenance
files under `data/reference/agentx-router/`.
