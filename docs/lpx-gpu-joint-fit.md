# One extra time per step for AA and SPEED-Bench at measured MTP acceptance

Record date: 2026-10-30.
The [GPU follow-up measurements](lpx-gpu-followup-results.md) measure the multi-token prediction (MTP) acceptance of the public Gemma 4 31B checkpoint with its released assistant at draft lengths K of 1, 3, 5 and 7.
Priced on TP32/PP8 with decode context parallelism (DCP) of 8 on 256 LP30 chips, every measured setting makes both Artificial Analysis (AA) values too fast, and the AA values fit only with a constant extra time of about 195 to 576 µs per step in the 14 of 16 settings that fit at all.
The same pricing makes NVIDIA's SPEED-Bench coding median of 4,767 TPS too fast as well.
The [API-outputs record](lpx-api-outputs.md) asked the same question for lookup drafting and found shared extra times of 38 to 61 µs.
This record checks whether one extra time per step places both AA values and the coding median within 5% at the measured MTP acceptance, which would make MTP with a large unmodeled cost per step an alternative to plain decoding with 13 to 24 µs of unmodeled time.

## Method

[`scripts/analyze_lpx_gpu_joint_fit.py`](../scripts/analyze_lpx_gpu_joint_fit.py) reads the [combined MTP analysis](../experiments/lpx/gpu-triton-followup-remaining-mtp-acceptance.json) and writes [`experiments/lpx/gpu-joint-fit.json`](../experiments/lpx/gpu-joint-fit.json).

- Each SPEED-Bench coding turn, with thinking on or off, is priced with the existing `coding_speed` step model: the verification step of K + 1 rows at the turn's mean context, plus the drafter step, plus the extra time.
- The extra time runs from 0 to 1,000 µs per step in steps of 0.5 µs, and the coding window is the range that places the median of the 89 turns within 5% of 4,767 TPS.
- The AA window of each question style and drafter cost is taken unchanged from the combined analysis, at 1.006 Gemma tokens per `o200k_base` token.
- The shared window is the intersection of the coding window and the AA window at the same K and drafter cost.
- At the lower end of each shared window the record reports the coding P80/median and maximum/median, for comparison with NVIDIA's 1.158 and 1.464.
- The 40 16K code-completion turns, measured at K = 3 and 7, are priced at zero extra time and at the smallest shared extra time of the same K and drafter cost, and compared with NVIDIA's 10,996 TPS example.

The extra time is one constant per step, added once to each verification step, following the convention of the API-outputs record.

```bash
.venv/bin/python scripts/analyze_lpx_gpu_joint_fit.py \
  --analysis experiments/lpx/gpu-triton-followup-remaining-mtp-acceptance.json \
  --summary experiments/lpx/gpu-joint-fit.json
```

## Differences from the published runs

| Item | Published runs | This check |
| --- | --- | --- |
| Model build | AA: an updated private build; SPEED-Bench: not stated | Public checkpoint, measured in BF16 on one H20 |
| Drafter and K | Not stated | Released MTP assistant at K = 1, 3, 5 and 7, with an assumed drafter cost of 5% or 10% of a one-token step per draft token |
| Prompts | AA's prompts unpublished; SPEED-Bench split not stated | Study book questions and public SPEED-Bench prompts |
| Thinking and sampling | Not stated | Thinking on for AA-like questions, on or off for coding; temperature zero |
| Step times | Measured on LP30 | Simulated by TokenSim; the extra time stands for every unmodeled cost |

## Expectations recorded before the run

1. The coding median alone fits within 0 to 1,000 µs for all 16 coding rows of K, thinking mode and drafter cost, and the lower end of the coding window rises with K, from below 200 µs at K = 1 to above 500 µs at K = 7.
2. No K = 1 row has a shared window, because the AA values need more extra time than the coding median at that draft length.
3. At least one row at K = 3 or K = 5 has a shared window, and every nonempty shared window starts above 300 µs per step.
4. At the lower end of every shared window the coding P80/median stays below 1.158 and the maximum/median below 1.464.
5. At zero extra time the fastest 16K code-completion turn exceeds 10,996 TPS at K = 3 and 7 for both drafter costs, and at the smallest shared extra time no turn reaches it.
