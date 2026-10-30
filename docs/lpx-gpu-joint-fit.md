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

## Results

The run took 2 seconds on 2026-10-30.
Each cell gives the extra time per step in microseconds; `none` marks an empty window.
Thinking refers to the coding turns; the AA-like turns always think.

| K | Drafter cost | Coding thinking | Coding window | AA summary window | AA quotes window | Shared window |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.05 | Off | 99.0–139.0 | 208.0–234.9 | 217.3–241.8 | none |
| 1 | 0.05 | On | 99.5–140.0 | 208.0–234.9 | 217.3–241.8 | none |
| 1 | 0.10 | Off | 85.5–126.0 | 194.6–221.5 | 204.0–228.4 | none |
| 1 | 0.10 | On | 86.0–126.5 | 194.6–221.5 | 204.0–228.4 | none |
| 3 | 0.05 | Off | 365.5–439.5 | 427.2–458.2 | 465.0–493.9 | Summary: 427.2–439.5 |
| 3 | 0.05 | On | 376.0–450.5 | 427.2–458.2 | 465.0–493.9 | Summary: 427.2–450.5 |
| 3 | 0.10 | Off | 325.5–399.0 | 387.1–418.1 | 424.9–453.9 | Summary: 387.1–399.0 |
| 3 | 0.10 | On | 335.5–410.5 | 387.1–418.1 | 424.9–453.9 | Summary: 387.1–410.5 |
| 5 | 0.05 | Off | 571.5–672.0 | 497.7–525.7 | 563.8–575.8 | Quotes: 571.5–575.8 |
| 5 | 0.05 | On | 591.0–693.5 | 497.7–525.7 | 563.8–575.8 | none |
| 5 | 0.10 | Off | 504.5–605.0 | 430.9–458.9 | 497.0–509.0 | Quotes: 504.5–509.0 |
| 5 | 0.10 | On | 524.0–627.0 | 430.9–458.9 | 497.0–509.0 | none |
| 7 | 0.05 | Off | 689.0–807.0 | 501.0–506.8 | none | none |
| 7 | 0.05 | On | 726.0–848.0 | 501.0–506.8 | none | none |
| 7 | 0.10 | Off | 595.0–713.5 | 407.4–413.2 | none | none |
| 7 | 0.10 | On | 632.5–754.5 | 407.4–413.2 | none | none |

Six of the 16 coding rows share a window with one AA question style, from 387.1 to 575.8 µs per step.
At the lower end of each shared window the coding speeds are:

| K | Coding thinking | Shared with | Coding median | P80/median | Maximum/median |
| --- | --- | --- | ---: | ---: | ---: |
| 3 | Off | Summary | 4,601 TPS | 1.056 | 1.116 |
| 3 | On | Summary | 4,668 TPS | 1.030 | 1.075 |
| 5 | Off | Quotes | 5,003 TPS | 1.088 | 1.172 |

Both drafter costs give the same speeds at the lower end of a shared window, because that end is set by the total step time that places the AA value 5% above AA's measurement.
NVIDIA's coding P80/median is 1.158 and its maximum/median is 1.464.

The 16K code-completion turns give:

| K | Drafter cost | Fastest turn at zero extra time | Fastest turn at the smallest shared extra time |
| --- | --- | ---: | ---: |
| 3 | 0.05 | 10,945 TPS | 4,893 TPS at 427.2 µs |
| 3 | 0.10 | 9,807 TPS | 4,893 TPS at 387.1 µs |
| 7 | 0.05 | 15,255 TPS | No shared window |
| 7 | 0.10 | 12,647 TPS | No shared window |

### Expectations

1. Held: all 16 coding rows fit, with lower ends from 85.5 to 99.5 µs at K = 1 and from 595.0 to 726.0 µs at K = 7.
2. Held: no K = 1 row has a shared window.
3. Held: four K = 3 rows share a window with the summary questions and two K = 5 rows with the quotes questions, all starting at 387.1 µs or more.
4. Held: at the shared windows the coding P80/median is 1.030 to 1.088 and the maximum/median 1.075 to 1.172.
5. Missed in part: at zero extra time the fastest code-completion turn exceeds 10,996 TPS at K = 7 and stays below it at K = 3, at 10,945 and 9,807 TPS; at the smallest shared extra time no turn reaches it.

## Interpretation

The measured MTP acceptance allows one shared explanation of AA's values and NVIDIA's coding median: MTP at K = 3 or 5 on both runs, with about 390 to 580 µs of unmodeled time per step.
That time is 1.4 to 2.2 times the whole modeled plain step of 267 µs at 10K.
The [fixed-cost evidence record](lpx-fixed-cost-evidence.md) finds that published first-generation data leave only about 1 to 2 µs to collective launches, instruction dispatch and synchronization, and leave the host's per-token loop without a size.
The [GroqRack check](groqrack-llama2-7b-validation.md) measured a host loop of 2.7 to 2.8 ms per token on first-generation hardware, so the published evidence does not exclude a per-token cost of this size on LPX.

This shared explanation does not reproduce two other NVIDIA values.
Its coding P80/median of 1.030 to 1.088 and maximum/median of 1.075 to 1.172 stay below NVIDIA's 1.158 and 1.464, because a large constant time per step dilutes the spread of the committed tokens per step.
Its fastest 16K code-completion turn reaches 4,893 TPS, less than half of NVIDIA's 10,996 TPS example.
The 10,996 TPS example fits MTP only with little unmodeled time, which would make the AA values too fast if the AA run drafted with the same acceptance.

The data therefore leave two incomplete explanations.
Plain decoding on AA with 13 to 24 µs of unmodeled time per step needs a weaker drafter than the measured MTP assistant on SPEED-Bench coding, such as lookup drafting, and still leaves the 10,996 TPS example unexplained.
MTP on both runs with about 390 to 580 µs of unmodeled time per step fits the three medians and leaves NVIDIA's spread and the 10,996 TPS example unexplained.
The acceptance comes from the public checkpoint on study prompts, and the published runs may use another build, drafter, draft length, thinking setting and prompts, so neither explanation identifies NVIDIA's deployment.
A measured LP30 step time or host loop would decide between them, since they differ by more than 350 µs per step.
