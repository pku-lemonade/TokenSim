# Native attention sensitivity

An explicit `native_attention_context_scale=0.17` was tested using the B200
AIC shape comparison as a diagnostic input. Formal runs keep the default scale
of 1.0; this experiment does not alter reference results.

| SKU | formal output token/s | scale 0.17 output token/s | change | reference error after scale |
|---|---:|---:|---:|---:|
| B200 c196 | 522.135 | 676.538 | +29.6% | −81.2% |
| B300 c384 | 1,027.989 | 1,260.572 | +22.7% | −80.2% |

The scale improves throughput but leaves the result far outside the 25%
threshold. It is therefore retained as an explicit sensitivity knob and not
promoted into the formal matrix. The result rules out a single context
attention scalar as a sufficient fix; remaining error includes the complete
module/backend scope, prefill scheduling, cache pressure and other operator
provenance differences. Evidence is in
`results/agentx-dsv4-full/sensitivity-aic-attn/`.
