# Speculative decoding under the published LPX test conditions

Record date: 2026-10-26.
The [article-condition sweep](lpx-article-sweep.md) found one batch-one layout that places both Artificial Analysis (AA) Gemma 4 31B speeds within 5% without speculative decoding: TP32/PP8 with decode context parallelism (DCP) of 8, provided each step costs 14 to 25 µs more than modeled.
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

The sweep ran on 2026-10-26 in 47 minutes 22 seconds with a peak resident memory of 1.37 GB.
[`experiments/lpx/article-speculative-results.json`](../experiments/lpx/article-speculative-results.json) holds every verification step, drafting time and fit.

```bash
.venv/bin/python scripts/run_lpx_article_speculative.py \
  --summary experiments/lpx/article-speculative-results.json
```

Of 378 combinations of layout, efficiency, draft length and drafter, 11 match both AA values with one shared acceptance at one native token per AA token.
All are TP32/PP8 with DCP 8.

| Efficiency | `K` | Drafter | 100K/10K step ratio | 10K step | Shared AA acceptance | SPEED-Bench median acceptance | 10,996 TPS example acceptance |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 0.30 | 1 | MTP 0.05 | 1.074 | 291.0 µs | 0.019 to 0.033 | 0.394 | unreachable |
| 0.30 | 1 | MTP 0.10 | 1.071 | 304.4 µs | 0.062 to 0.081 | 0.458 | unreachable |
| 0.50 | 1 | MTP 0.05 | 1.063 | 290.5 µs | 0.006 to 0.031 | 0.391 | unreachable |
| 0.50 | 1 | MTP 0.10 | 1.060 | 303.8 µs | 0.050 to 0.079 | 0.454 | unreachable |
| 0.50 | 2 | Lookup | 1.067 | 289.5 µs | 0.007 to 0.027 | 0.298 | unreachable |
| 0.50 | 2 | MTP 0.05 | 1.071 | 316.2 µs | 0.094 to 0.111 | 0.375 | unreachable |
| 0.50 | 2 | MTP 0.10 | 1.065 | 342.9 µs | 0.164 to 0.184 | 0.445 | unreachable |
| 0.50 | 3 | Lookup | 1.080 | 302.1 µs | 0.060 to 0.068 | 0.317 | 0.881 |
| 0.50 | 3 | MTP 0.05 | 1.083 | 342.2 µs | 0.173 to 0.178 | 0.407 | 0.963 |
| 0.50 | 3 | MTP 0.10 | 1.074 | 382.3 µs | 0.256 to 0.267 | 0.484 | unreachable |
| 0.50 | 4 | MTP 0.10 | 1.085 | 421.9 µs | 0.332 to 0.335 | 0.525 | 0.965 |

The 10K step includes drafting.
With the 1.073 Gemma tokens per `o200k_base` token measured on another model's reasoning traces in the [token-ratio record](lpx-token-ratio.md), the upper end of the measured range, the same layout gives 13 matches, adding lookup at `K = 1` on both efficiencies, with shared acceptances of 0.028 to 0.383.

### Comparison with the expectations

1. Partly held.
   Lookup drafting matched only on TP32/PP8 with DCP 8, but only at efficiency 0.50 and `K` of 2 and 3.
   At `K = 1` the 10K step of 277.1 to 277.6 µs is too short even with no accepted tokens, and at efficiency 0.30 the step ratio is 1.104 at `K = 2`, above the 1.0896 bound, because verifying more query rows multiplies the attention matrix work.
2. Held: MTP drafting matched at `K` from 1 to 4 and lookup drafting at `K` of 2 and 3.
3. Held: the largest shared acceptance is 0.335, and no match lies in the MTP prose range of 0.57 to 0.68.
4. Held at 10K and missed at 100K.
   At the MTP prose range, the 10K value needs 111 to 526 µs of extra time per step in every combination.
   In 93 of 252 MTP combinations, mostly at efficiency 0.10 or `K` of 5 and 7, the 100K value would need a shorter step than modeled, so no constant extra time matches both.
5. Held: no point at efficiency 0.10 matched.
6. Held: TP16/PP16 matched with no drafter.
7. Held: with MTP drafting the coding example is reachable only at `K` of 3 and 4.

The derived values in this record are reproduced from the committed results by [`scripts/analyze_lpx_published_conditions.py`](../scripts/analyze_lpx_published_conditions.py), which writes [`experiments/lpx/published-conditions-analysis.json`](../experiments/lpx/published-conditions-analysis.json).

## Interpretation

The drafter changes which setups match, and the main conclusion holds for every drafter.

- **Lookup drafting** adds no drafting time and only lengthens verification.
  It matches at `K` of 2 or 3 on TP32/PP8 with DCP 8 at efficiency 0.50, and at `K = 1` only with the reasoning-text token ratio.
- **MTP drafting** adds 13 to 27 µs per draft token that is nearly the same at both contexts.
  That time plays the role of the 14 to 25 µs that the [article-condition sweep](lpx-article-sweep.md) found missing, so MTP drafting matches at more draft lengths and on both efficiencies at `K = 1`.
- **Every match needs a shared AA acceptance of 0.006 to 0.335**, or 0.028 to 0.383 at the reasoning-text token ratio.
  Published MTP drafters accept 0.57 to 0.68 on prose and lookup drafters 0.13 to 0.73.
  Only lookup drafting at `K = 3` with the reasoning-text ratio, at 0.124 to 0.132, reaches the lowest published lookup value of 0.13.

A drafter that works as published would make the simulated endpoint much faster than AA measured.
At the MTP prose range the AA 10K value would need 111 to 526 µs more per step than the model gives.
Under these conditions the AA values therefore fit best with no speculative decoding, or with drafting that almost never succeeds on AA's prompts.
The coding values need high acceptance at `K` of at least 3, which fits a drafter on code, so they may come from a build or workload where drafting helps, and they do not constrain the AA configuration.
