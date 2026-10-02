# Final native AgentX ablations

The ablations use the full DeepSeek-V4-Pro model, all 393 traces, warmup,
3,600 simulated profiling seconds, the 30-second grace period, cancellation
cleanup, and three simulation seeds. They are diagnostic comparisons and have
no independently verified reference curve identity.

| SKU / change | mean output token/s | p90 TTFT (s) | p90 ITL (s) | GPU hit | CPU hit |
|---|---:|---:|---:|---:|---:|
| B200 c196, formal SimpleCPU + MTP | 522.135 | 263.624 | 0.401 | 26.734% | 67.396% |
| B200 c196, HBM only + MTP | 232.254 | 581.029 | 0.640 | 68.891% | 0.000% |
| B200 c196, SimpleCPU + MTP disabled | 235.448 | 327.432 | 0.715 | 30.260% | 64.006% |
| B300 c384, formal SimpleCPU + MTP | 1,027.989 | 104.064 | 0.425 | 95.024% | 0.371% |
| B300 c384, HBM only + MTP | 1,005.492 | 103.877 | 0.424 | 95.469% | 0.000% |
| B300 c384, SimpleCPU + MTP disabled | 518.902 | 85.502 | 0.713 | 94.812% | 0.369% |

The HBM comparison isolates offload: it removes CPU hits and sharply reduces
B200 throughput because the GPU pool is capacity-bound, while B300 remains
close to the SimpleCPU result. Disabling MTP approximately halves throughput on
both SKUs because the target forward no longer advances 2/3 tokens per decode
step. Every ablation passed the curve lifecycle validator; raw seed rows,
manifests and the machine-readable summary are in
`results/agentx-dsv4-full/ablations-final/`.
