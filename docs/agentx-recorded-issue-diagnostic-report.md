# Recorded issue-time diagnostic

The native replay now supports `--agentx_issue_mode recorded`. It preserves the
trace timestamp anchor while still honoring dependency and join readiness; it
does not apply tree/system idle compression. This separates request-issue timing
from the formal closed-loop feedback path and is diagnostic only.

Both SKU checkpoints completed the full warmup, 3,600-second profile, grace and
cancellation protocol with seed 0:

| SKU | output token/s | p90 TTFT (s) | p90 ITL (s) | successful requests |
|---|---:|---:|---:|---:|
| B200 c196 | 629.266 | 76.333 | 0.304 | 3,384 |
| B300 c384 | 1,080.462 | 74.621 | 0.304 | 5,854 |

The native lifecycle validator passed both runs. Compared with formal closed-loop
main points, recorded issue timing changes the observed request mix and latency
distribution; it is therefore reported separately and is not used for reference
accuracy or calibration. Evidence is in
`results/agentx-dsv4-full/diagnostics-recorded-issue/`.
