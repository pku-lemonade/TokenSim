# Completed GPU follow-up measurements

Record date: 2026-10-30.
The [follow-up worker](../experiments/lpx/gpu-triton-followup-status.json) completed all 912 requested measurements and 984 turns.
The original priority-1 measurements remain unchanged.
The [measurement record](lpx-gpu-measurements.md) describes the checkpoint, H20 GPU, serving configurations, precision, prompts and differences from the published deployments.
These measurements use the public checkpoint and do not reproduce the unpublished Artificial Analysis (AA) or NVIDIA serving configuration.

## MTP acceptance and output quality

The [combined multi-token prediction (MTP) analysis](../experiments/lpx/gpu-triton-followup-remaining-mtp-acceptance.json) covers all three planned priorities: 976 requests, 1,048 turns and 26 groups, including the original priority-1 runs.
Recomputing the analysis from the six raw-output files produces a byte-identical JSON artifact.
Each document group has 16 turns, each coding group has 89 turns, and each 16K code-completion group has 40 turns.
Equivalent acceptance is derived from each group's mean committed tokens per speculative step.

| Draft length K | Document acceptance range | Coding without thinking | Coding with thinking | 16K code completion | Thinking turns hitting the output cap |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.780721–0.814020 | 0.931032 | 0.932000 | Not planned | 5/89 |
| 3 | 0.745948–0.784691 | 0.913127 | 0.917675 | 0.885962 | 5/89 |
| 5 | 0.732312–0.767748 | 0.905913 | 0.911094 | Not planned | 7/89 |
| 7 | 0.720491–0.760633 | 0.896233 | 0.900837 | 0.870784 | 6/89 |

All 23 capped thinking turns have no final answer; their reasoning tokens remain in the analysis.
No turns trigger the recorded periodic-repetition heuristic.
Every document, coding-without-thinking and 16K code-completion turn stops normally with a nonempty answer.
Acceptance and speed on capped thinking turns describe generated prefixes and do not establish successful task completion.
Validation artifacts accompany each raw-output file and check exact request and turn coverage, input hashes, precision and speculative counter identities.

Coding without thinking exceeds every document group at the same K by at least 0.117012, satisfying expectation 3 across all four draft lengths.
The K=3 document result remains within the preregistered acceptance range, satisfying expectation 2.

## Paired context comparison

Every row pairs all 16 books; differences are per-book acceptance at 100K minus acceptance at 10K, averaged across books.
Counters combine thinking and answer tokens and cannot separate their acceptance.

| K | Quotes mean difference | Quotes higher at 100K | Summary mean difference | Summary higher at 100K |
| --- | --- | --- | --- | --- |
| 1 | -0.008901 | 6/16 | +0.001861 | 11/16 |
| 3 | -0.011240 | 5/16 | -0.007673 | 5/16 |
| 5 | -0.013396 | 2/16 | -0.003991 | 7/16 |
| 7 | -0.016092 | 1/16 | -0.008081 | 6/16 |

Expectation 4 remains satisfied by the K=3 means.
The other draft lengths also have paired mean differences with magnitude below 0.03.

## LP30 pricing

The combined analysis records all 16 AA prices across K, question style and the 5%/10% drafter-cost assumptions.
The token ratio remains 1.006 under the recorded selection rule.
At 10K, every price exceeds the published AA speed by at least 72.2616%.
The smallest lower endpoint of any nonempty common extra-time interval is 194.6319 microseconds per step.
No interval includes an added cost of 25 microseconds or less, satisfying expectation 5 across all four draft lengths.
These are TokenSim prices at measured acceptance, not measured LP30 latency or a fit establishing the published deployment's costs.

## Coding speed spreads

The measured H20 table uses all 89 turns per row, including capped thinking turns.
P80 denotes the 80th percentile of per-turn output speed.
GPU speeds include the serving stack's measured step times.

| K | Thinking | Median TPS | P80 TPS | Maximum TPS | P80/median | Maximum/median |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Off | 85.577 | 89.090 | 93.486 | 1.041056 | 1.092430 |
| 1 | On | 78.229 | 80.673 | 88.255 | 1.031239 | 1.128156 |
| 3 | Off | 149.039 | 158.837 | 176.397 | 1.065741 | 1.183562 |
| 3 | On | 134.215 | 142.702 | 152.892 | 1.063230 | 1.139152 |
| 5 | Off | 191.433 | 210.214 | 234.272 | 1.098106 | 1.223780 |
| 5 | On | 178.594 | 190.113 | 207.281 | 1.064500 | 1.160627 |
| 7 | Off | 220.544 | 251.412 | 300.981 | 1.139963 | 1.364720 |
| 7 | On | 200.123 | 224.080 | 251.877 | 1.119712 | 1.258614 |

The LP30 table uses the same measured acceptance and context-dependent modeled steps.
The drafter column gives its cost as a fraction of a modeled one-token step per draft token.
The ratios describe the acceptance-and-context pricing model and exclude measured GPU latency variation.

| K | Thinking | Drafter cost | Median TPS | P80 TPS | Maximum TPS | P80/median | Maximum/median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Off | 0.05 | 6715.572 | 6820.646 | 6895.365 | 1.016 | 1.027 |
| 1 | Off | 0.10 | 6418.389 | 6518.580 | 6589.962 | 1.016 | 1.027 |
| 1 | On | 0.05 | 6727.872 | 6775.216 | 6886.451 | 1.007 | 1.024 |
| 1 | On | 0.10 | 6430.032 | 6475.327 | 6581.560 | 1.007 | 1.024 |
| 3 | Off | 0.05 | 10400.242 | 10981.519 | 11611.696 | 1.056 | 1.116 |
| 3 | Off | 0.10 | 9300.465 | 9820.154 | 10383.512 | 1.056 | 1.116 |
| 3 | On | 0.05 | 10546.250 | 10867.457 | 11340.399 | 1.030 | 1.075 |
| 3 | On | 0.10 | 9431.732 | 9718.885 | 10141.503 | 1.030 | 1.075 |
| 5 | Off | 0.05 | 12342.074 | 13432.502 | 14461.005 | 1.088 | 1.172 |
| 5 | Off | 0.10 | 10535.331 | 11466.497 | 12344.104 | 1.088 | 1.172 |
| 5 | On | 0.05 | 12590.027 | 13211.962 | 14007.205 | 1.049 | 1.113 |
| 5 | On | 0.10 | 10747.813 | 11279.123 | 11957.736 | 1.049 | 1.113 |
| 7 | Off | 0.05 | 12832.601 | 14639.125 | 16764.909 | 1.141 | 1.306 |
| 7 | Off | 0.10 | 10583.990 | 12074.337 | 13827.128 | 1.141 | 1.306 |
| 7 | On | 0.05 | 13245.801 | 14348.003 | 16043.134 | 1.083 | 1.211 |
| 7 | On | 0.10 | 10926.797 | 11853.910 | 13234.649 | 1.085 | 1.211 |

The [plain coding control](lpx-gpu-measurements.md#coding-speed-spread) supplies the non-speculative baseline and its LP30 price.
All measured H20 and priced LP30 MTP ratios are below NVIDIA's published coding P80/median of 1.158 and maximum/median of 1.464.
This comparison does not identify NVIDIA's thinking setting, workload split or drafter.
Expectation 8 remains satisfied by K=3 without thinking and by the plain control.

## Native n-gram versus replay

The [full comparison](../experiments/lpx/gpu-triton-followup-ngram-comparison.json) covers 144 requests and 153 turns at K=7, including every document and coding-without-thinking request.
Every turn finishes normally with a nonempty answer and no repetition flags.
Native V2 matching and replay retain the documented padding, tokenization, template and control-token differences.

| Set | Turns | Native equivalent acceptance | Replay equivalent acceptance | Native minus replay |
| --- | --- | --- | --- | --- |
| Documents | 64 | 0.266949 | 0.267155 | -0.000205 |
| Coding | 89 | 0.360232 | 0.359954 | +0.000278 |

Both set-level differences are below 0.03, satisfying expectation 7.
The mean absolute per-turn acceptance difference is 0.000245 for documents and 0.002235 for coding.
Agreement at these aggregate levels does not imply identical algorithms or per-turn step counts.
