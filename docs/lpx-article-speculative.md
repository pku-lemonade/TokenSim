# Speculative decoding under the published LPX test conditions

Record date: 2026-10-26.
The [article-condition sweep](lpx-article-sweep.md) found one batch-one layout that reproduces both Artificial Analysis (AA) Gemma 4 31B speeds without speculative decoding: TP32/PP8 with decode context parallelism (DCP) of 8, provided each step costs 14 to 25 µs more than modeled.
NVIDIA's articles, the AA footnotes and press reports neither state nor deny that the endpoint drafts tokens.
This record adds chain speculative decoding to the same conditions and tests how the choice of drafter changes the result.

## How a drafter enters the step

In chain speculative decoding a drafter proposes `K` tokens, and one target forward verifies them together with one target token.
The step time is the drafting time plus the verification forward, and the step commits `tau = (1 - alpha^(K+1)) / (1 - alpha)` tokens at a per-position acceptance `alpha`.
The output speed is `tau` divided by the step time.

A drafter changes the result in three ways:

- **Acceptance.**
  The drafter and the prompt set `alpha`, and published values differ by drafter type and by workload.
- **Drafting cost.**
  A model drafter runs its own layers for every draft token, and a lookup drafter only searches earlier text.
- **Growth of drafting cost with context.**
  A drafter that attends over the whole context gets slower at 100K, which raises the 100K step more than the 10K step.

## Conditions

The target conditions match the [article-condition sweep](lpx-article-sweep.md): one AA request per forward, FP8 weights and KV cache, 256 chips, 96 links of 112 Gb/s per chip and the Hot Chips latency tiers.
No constant time is added beyond the modeled forward and drafting.

| Dimension | Values |
| --- | --- |
| Layouts | TP32/PP8 with DCP 1, 2, 4 and 8; TP16/PP16 with DCP 1, 2 and 4 |
| Attention matrix efficiency | 0.10, 0.30 and 0.50 |
| Draft length `K` | 1, 2, 3, 4, 5 and 7 |
| Drafter | Lookup; multi-token prediction (MTP) at 0.05 and 0.10 of a one-token step per draft token |

| Drafter | Cost per draft token | Growth with context | Published acceptance, prose | Published acceptance, code |
| --- | --- | --- | --- | --- |
| Lookup (n-gram, prompt lookup, suffix matching) | None | None | 0.13 to 0.73 | 0.36 to 0.73 |
| Gemma 4 MTP drafter, 500M parameters, 4 layers | 0.05 or 0.10 of the one-token 10K step | One target global layer's attention increment above 10K per draft token | 0.57 to 0.68 | 0.85 to 0.90 |

The MTP drafter's four layers are about one fifteenth of the target's 60 layers, so 0.05 to 0.10 of a one-token step brackets its fixed cost at batch one.
Its one global-attention layer reads the target's KV cache, so its cost grows with context like one target global layer.
The acceptance ranges come from the [acceptance evidence](lpx-acceptance-evidence.md); no Gemma 4 measurement exists.
Lookup drafting is modeled as always proposing `K` tokens, although a real lookup drafter proposes nothing when it finds no match.

[`scripts/run_lpx_article_speculative.py`](../scripts/run_lpx_article_speculative.py) captures one native verification forward for each layout, efficiency and query count and adds the drafting time of each drafter.
For each combination it reports the shared acceptance range that places both AA speeds within 5%, the acceptance each workload needs alone, the acceptances the two coding values need, and the constant extra time per step that both AA speeds would need at the drafter's published prose acceptance.

## Expectations recorded before the run

1. With lookup drafting, TP32/PP8 with DCP 8 at efficiency 0.30 or 0.50 matches both AA values with one shared acceptance at `K` from 1 to 3 and at no `K` of 4 or more.
2. MTP drafting matches at more draft lengths than lookup drafting on TP32/PP8 with DCP 8, because its drafting time is mostly the same at both contexts and lowers the ratio of the 100K step to the 10K step.
3. Every shared AA acceptance is below 0.45, so no MTP match lies inside the MTP prose range of 0.57 to 0.68.
4. At the MTP prose range, both AA values need at least 100 µs of extra time per step on every layout.
5. No point at efficiency 0.10 matches with any drafter.
6. TP16/PP16 does not match with lookup drafting.
7. With MTP drafting, the 10,996 TPS coding example needs `K` of at least 3.

## Results

Pending.
