# Does a lookup drafter reach the 10,996 TPS code-completion example on all 302 prompts?

Record date: 2026-10-30, updated 2026-10-31.
Status: C0 through C4 are complete; all 302 plain and 302 native n-gram outputs passed validation, and the results are published on `lpx-context-parallel`.

## Question

NVIDIA reports three speeds under the Gemma4-31B-LPX name: a SPEED-Bench coding median of 4,767 output tokens per second (TPS), a Hot Chips coding example of 10,996 TPS at a 16K input, and, through Artificial Analysis (AA), 3,382 TPS at 10K and 3,431 TPS at 100K input tokens.
One hypothesis explains all of them with one deployment and one drafter whose acceptance depends on content: answers that copy little from the prompt give AA's speed, writing code gives the coding median, and completing code in a repository gives the example.
That pattern fits a lookup (n-gram) drafter, which copies from the prompt.
Gemma 4 31B's released multi-token prediction (MTP) drafter accepts 0.72 to 0.81 on prose, as the [GPU follow-up results](lpx-gpu-followup-results.md) show, so it does not fit the AA end of the pattern.

On TP32/PP8 with decode context parallelism (DCP) of 8 on 256 LP30 chips, with 13 to 21 µs of unmodeled time per step, the example needs about 3.5 committed tokens per step at a draft length `K` of 3 and about 4.1 at `K = 7`.
The [API-outputs record](lpx-api-outputs.md) measured lookup on 40 of the 302 code-completion prompts of SPEED-Bench's 16K throughput split.
At `K = 7` the 40 requests commit 1.24 to 3.39 tokens per step, and the fastest reaches 9,785 TPS, 11% below the example.
NVIDIA's example is one prompt chosen for a slide, and the fastest of 302 prompts can exceed the fastest of 40.
This test measures all 302 prompts and decides whether the peak part of the hypothesis holds.

The AA part is a separate question.
Lookup on the AA-like book answers needs at least 26 µs of extra time per step, while the coding median without thinking allows at most 21 µs, so the full hypothesis also needs AA prompts that copy less than the book questions.
This test reports the code-completion speeds at 26 µs as well, and it does not test AA's prompts.

## Method

The test reuses the H20 environment, the Triton serving profile and the client of the [GPU measurements](lpx-gpu-measurements.md).

- The prompts are all 302 code-completion requests, built with `--completions 302` in [`scripts/build_lpx_api_prompts.py`](../scripts/build_lpx_api_prompts.py); without the option the builder still writes the 264 earlier requests.
- Gemma 4 31B answers each prompt with plain decoding in BF16, thinking off, temperature 0 and the request's 4,096-token limit.
- [`scripts/measure_lpx_lookup_acceptance.py`](../scripts/measure_lpx_lookup_acceptance.py) replays lookup on the `answer-alone` part at minimum n-gram lengths 1 and 2 and `K` of 1, 3, 5 and 7, as the API-outputs record does.
  The same replay runs on the H20 coding answers without thinking from the [plain coding control](lpx-gpu-measurements.md#coding-speed-spread), and on the answers to the 40 earlier prompts as a separate data set.
- [`scripts/analyze_lpx_cc16k_peak.py`](../scripts/analyze_lpx_cc16k_peak.py) prices every request on the closest setup with the step model of the [coding-spread record](lpx-coding-spread.md).
  For each setting it finds the extra time per step that places the coding median within 5% of 4,767 TPS, and prices the code-completion requests at 0, 13, 21 and 26 µs and at the upper end of that window.
  On the API outputs it gives the earlier 9,785 TPS and the earlier coding windows.
- A second run with vLLM's native n-gram drafter at `K = 7` checks the replay, as task T4 of the [GPU TODO](lpx-gpu-gemma-todo.md) did on the other prompt sets.

The test reports all 8 lookup settings.
A setting that reaches the example counts only together with a nonempty coding window at the same setting, because one deployment uses one drafter setting.

## Expectations recorded before the measurement

1. The build with `--completions 302` gives a request file with SHA-256 `c4815c8625696784735a924c51a042f554c1f27e309539f87126c30c134b776a`, and it contains all 40 earlier code-completion IDs.
2. On the 40 earlier prompts, the H20 answers commit 1.7 to 2.2 tokens per step on average at minimum n-gram 1 and `K = 7`; the API answers committed 1.96.
3. Over all 302 prompts, the mean at the same setting lies within 0.2 tokens per step of the mean over the 40 earlier prompts.
4. At the same setting with no extra time, the fastest of the 302 requests is at least 5% faster than the fastest of the 40 earlier requests, and it reaches 9,785 to 11,500 TPS.
5. No setting with a nonempty coding window has a code-completion request at or above 10,996 TPS at the upper end of that window.
6. Every coding window from the H20 coding answers ends at 21 µs or less, so no lookup setting fits the coding median at the 26 µs that the AA-like book answers need.
7. vLLM's native n-gram drafter at `K = 7` gives an equivalent acceptance within 0.03 of the replay over the 302 requests.

Expectation 5 states the hypothesis's peak part as a prediction that it fails: among the 40 earlier requests only one exceeds 2.8 tokens per step, and the example needs about 4.

## How the results bear on the hypothesis

| Outcome | Meaning |
| --- | --- |
| Expectation 5 missed: a setting with a coding window reaches 10,996 TPS | A lookup-type drafter on one deployment can give both the coding median and the example; the AA part remains open |
| Expectation 5 held, with the fastest request within 5% of 10,996 TPS | The peak lies at the edge of the measured range; another build, a longer draft or one more repetitive prompt could supply it |
| Expectation 5 held, with the fastest request more than 5% below | Lookup with `K` up to 7 on SPEED-Bench's code completion does not give the example at the extra time the coding median allows; the example then needs another drafter, a longer draft or a prompt outside SPEED-Bench |

The step model prices drafts up to `K = 7`, because the LP30 sweep results hold verification steps of at most 8 rows.
Longer drafts need a new sweep with 9 to 16 rows.

## Differences from the published runs

| Item | Published runs | This test |
| --- | --- | --- |
| Model build | Not stated for the example; AA used a private build | Public checkpoint `google/gemma-4-31B-it` in BF16 on one H20 |
| Drafter and `K` | Not stated | Lookup replay at minimum n-gram 1 and 2 and `K` up to 7, and vLLM's n-gram drafter at `K = 7` |
| Prompt | One unpublished prompt with a 16K input | SPEED-Bench's 302 code-completion prompts, of 18,071 to 21,421 raw Gemma tokens before the serving chat template |
| Thinking and sampling | Not stated | Thinking off, temperature 0 |
| Step times | Measured on LP30 | Simulated by TokenSim, with a constant extra time per step |

## GPU work plan

The agent on the GPU machine follows the ground rules of the [GPU TODO](lpx-gpu-gemma-todo.md#ground-rules): it pushes to `lpx-context-parallel`, keeps prompts and rows outside the repository, does not edit the expectations above and reports every setting that ran.
The plain run takes about 2 hours: 302 answers of about 686 tokens, the mean of the earlier sample, at about 45 TPS, the speed of the H20 plain control.
The native n-gram run takes less.

### C0: inputs

- [x] Rebuild both request files from the inputs of the GPU TODO and check their hashes.

```bash
python -I scripts/build_lpx_api_prompts.py \
  --gemma-tokenizer inputs/tokenizer.json \
  --speed-qualitative inputs/speed-qualitative.parquet \
  --speed-16k inputs/speed-16k.parquet \
  --gutenberg inputs/gutenberg \
  --output inputs/requests.jsonl
python -I scripts/build_lpx_api_prompts.py \
  --gemma-tokenizer inputs/tokenizer.json \
  --speed-qualitative inputs/speed-qualitative.parquet \
  --speed-16k inputs/speed-16k.parquet \
  --gutenberg inputs/gutenberg \
  --completions 302 \
  --output inputs/requests-cc302.jsonl
sha256sum inputs/requests.jsonl inputs/requests-cc302.jsonl
```

The first file must keep `7abb2f93587b01649c27e262e041afe96dd68cc055908d80f35610b5de9506ae`, and the second must give the hash of expectation 1.
Check that every `speed-cc16k` ID of the first file appears in the second.

### C1: plain answers to the 302 prompts

- [x] Start the plain Triton server with the arguments of [`gpu-triton-followup-plain-k0-server-command.json`](../experiments/lpx/gpu-triton-followup-plain-k0-server-command.json).
- [x] Run [`scripts/generate_lpx_gpu_outputs.py`](../scripts/generate_lpx_gpu_outputs.py) with `--requests inputs/requests-cc302.jsonl`, `--set speed-cc16k`, `--thinking off`, `--method plain` and the run label `cc302-plain`.
- [x] Commit the outputs without prompts as `experiments/lpx/gpu-cc302-plain-outputs.jsonl`, with the server command and a validation record of coverage, finish reasons and repetition flags, as for the earlier GPU runs.

### C2: lookup replay

- [x] Export the rows of the 302 answers, outside the repository.
- [x] Write a second rows file that holds only the 40 earlier IDs.
- [x] Export the rows of the plain coding control without thinking, `gpu-triton-followup-plain-k0-outputs.jsonl`, joined with `inputs/requests.jsonl`.

```bash
python -I scripts/measure_lpx_lookup_acceptance.py \
  --gemma-tokenizer inputs/tokenizer.json \
  --data speed-coding-plain=rows/speed-coding-plain.json#answer-alone \
  --data speed-cc16k-plain=rows/speed-cc16k-plain.json#answer-alone \
  --data speed-cc16k40-plain=rows/speed-cc16k40-plain.json#answer-alone \
  --per-row \
  --summary experiments/lpx/gpu-cc302-lookup-results.json
python -I scripts/analyze_lpx_cc16k_peak.py \
  --lookup experiments/lpx/gpu-cc302-lookup-results.json \
  --summary experiments/lpx/gpu-cc302-peak.json
```

### C3: native n-gram check

- [x] Start the n-gram server of task T4 at `K = 7` with the Triton profile, run the same 302 requests with `--method ngram --draft-tokens 7` and the run label `cc302-ngram-k7`, and compare its equivalent acceptance with the replay as [`scripts/analyze_lpx_gpu_ngram.py`](../scripts/analyze_lpx_gpu_ngram.py) does.

### C4: recording

- [x] Add a results section to this record: one table per setting with the coding window and the code-completion median, fastest request and count at or above 10,996 TPS at each priced extra time, the same for the 40 earlier prompts, and the outcome of each expectation.
- [x] Add a row to the outcome table of the [progress summary](lpx-aa-reproduction-progress.md) and update its current position.
- [x] Run Black, the unit tests and `git diff --check`, commit and push.

## Plain-output replay results

The [plain-output validation](../experiments/lpx/gpu-cc302-plain-outputs-validation.json) covers all 302 requests: 301 stopped normally and one reached the prescribed 4,096-token limit.
The capped request, `speed-cc16k/6eed5048a6c5445fbb5a19c19dbc64f4`, remains in every relevant calculation.
No answer is empty and no output triggers the recorded repetition checks; these checks do not establish that the generated code is correct.
The run generated 204,651 native tokens, averaging 677.65 per request, with a median measured H20 decode speed of 38.72 TPS.
The throughputs below are simulated LP30 throughputs from replay, not measured H20 or LP30 serving speeds.

The [lookup replay](../experiments/lpx/gpu-cc302-lookup-results.json) contains the coding control, all 302 completion answers and the exact earlier 40-answer subset.
The [pricing artifact](../experiments/lpx/gpu-cc302-peak.json) retains full precision; the tables round TPS to the nearest integer and count threshold crossings before rounding.
Coding windows use the existing 0.5 µs search grid and place the 89-turn, 80-request coding-control median within 5% of 4,767 TPS.
All settings retain the prescribed nominal-16K step model despite the larger measured raw prompt lengths.
The [identity artifact](../experiments/lpx/gpu-cc302-peak-identities.json) maps replay rows and all tied maxima to request IDs.

### Fastest request identities

| Alias | Request ID | Membership | Native output tokens | Finish reason |
| --- | --- | --- | ---: | --- |
| A | `speed-cc16k/793adb1086a5495abc418d940da8e8f9` | Earlier 40 and all 302 | 1,409 | stop |
| B | `speed-cc16k/f1dfde2209794054a8cac4f790f21c57` | Additional 262 | 1,502 | stop |

Both requests have turn index 0 and neither has an empty-answer or repetition flag.
The fastest identity is unchanged across the priced extra times within each setting.

### Minimum n-gram 1, K = 1

Coding extra-time window: 0 to 7 µs.

| Extra time (µs) | 302 median TPS | 302 fastest TPS (A) | 302 ≥10,996 | 40 median TPS | 40 fastest TPS (A) | 40 ≥10,996 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4,853 | 6,364 | 0 | 4,966 | 6,364 | 0 |
| 13 | 4,636 | 6,080 | 0 | 4,744 | 6,080 | 0 |
| 21 | 4,512 | 5,917 | 0 | 4,616 | 5,917 | 0 |
| 26 | 4,437 | 5,820 | 0 | 4,540 | 5,820 | 0 |
| 7 (window upper end) | 4,734 | 6,208 | 0 | 4,844 | 6,208 | 0 |

### Minimum n-gram 1, K = 3

Coding extra-time window: 0 to 23.5 µs.

| Extra time (µs) | 302 median TPS | 302 fastest TPS (A) | 302 ≥10,996 | 40 median TPS | 40 fastest TPS (A) | 40 ≥10,996 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5,279 | 9,419 | 0 | 5,564 | 9,419 | 0 |
| 13 | 5,059 | 9,030 | 0 | 5,331 | 9,030 | 0 |
| 21 | 4,933 | 8,806 | 0 | 5,197 | 8,806 | 0 |
| 26 | 4,857 | 8,672 | 0 | 5,117 | 8,672 | 0 |
| 23.5 (window upper end) | 4,894 | 8,739 | 0 | 5,157 | 8,739 | 0 |

### Minimum n-gram 1, K = 5

Coding extra-time window: 0 to 17 µs.

| Extra time (µs) | 302 median TPS | 302 fastest TPS (B) | 302 ≥10,996 | 40 median TPS | 40 fastest TPS (A) | 40 ≥10,996 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5,150 | 10,507 | 0 | 5,515 | 10,351 | 0 |
| 13 | 4,949 | 10,101 | 0 | 5,300 | 9,954 | 0 |
| 21 | 4,833 | 9,867 | 0 | 5,176 | 9,725 | 0 |
| 26 | 4,763 | 9,726 | 0 | 5,101 | 9,587 | 0 |
| 17 (window upper end) | 4,890 | 9,983 | 0 | 5,237 | 9,838 | 0 |

### Minimum n-gram 1, K = 7

Coding extra-time window: 0 to 7 µs.

| Extra time (µs) | 302 median TPS | 302 fastest TPS (B) | 302 ≥10,996 | 40 median TPS | 40 fastest TPS (A) | 40 ≥10,996 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4,926 | 11,841 | 1 | 5,371 | 10,966 | 0 |
| 13 | 4,746 | 11,411 | 1 | 5,173 | 10,574 | 0 |
| 21 | 4,641 | 11,162 | 1 | 5,058 | 10,345 | 0 |
| 26 | 4,578 | 11,012 | 1 | 4,989 | 10,208 | 0 |
| 7 (window upper end) | 4,828 | 11,606 | 1 | 5,262 | 10,751 | 0 |

### Minimum n-gram 2, K = 1

Coding extra-time window: empty; this setting cannot support the joint coding-and-peak hypothesis.

| Extra time (µs) | 302 median TPS | 302 fastest TPS (A) | 302 ≥10,996 | 40 median TPS | 40 fastest TPS (A) | 40 ≥10,996 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4,636 | 6,196 | 0 | 4,749 | 6,196 | 0 |
| 13 | 4,426 | 5,918 | 0 | 4,534 | 5,918 | 0 |
| 21 | 4,306 | 5,759 | 0 | 4,410 | 5,759 | 0 |
| 26 | 4,234 | 5,664 | 0 | 4,337 | 5,664 | 0 |

### Minimum n-gram 2, K = 3

Coding extra-time window: 0 to 6 µs.

| Extra time (µs) | 302 median TPS | 302 fastest TPS (A) | 302 ≥10,996 | 40 median TPS | 40 fastest TPS (A) | 40 ≥10,996 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5,098 | 9,048 | 0 | 5,333 | 9,048 | 0 |
| 13 | 4,875 | 8,667 | 0 | 5,100 | 8,667 | 0 |
| 21 | 4,747 | 8,448 | 0 | 4,966 | 8,448 | 0 |
| 26 | 4,670 | 8,317 | 0 | 4,886 | 8,317 | 0 |
| 6 (window upper end) | 4,993 | 8,868 | 0 | 5,223 | 8,868 | 0 |

### Minimum n-gram 2, K = 5

Coding extra-time window: 0 to 6 µs.

| Extra time (µs) | 302 median TPS | 302 fastest TPS (B) | 302 ≥10,996 | 40 median TPS | 40 fastest TPS (A) | 40 ≥10,996 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5,112 | 10,486 | 0 | 5,385 | 10,009 | 0 |
| 13 | 4,896 | 10,069 | 0 | 5,158 | 9,611 | 0 |
| 21 | 4,774 | 9,829 | 0 | 5,027 | 9,381 | 0 |
| 26 | 4,701 | 9,684 | 0 | 4,949 | 9,243 | 0 |
| 6 (window upper end) | 5,010 | 10,290 | 0 | 5,278 | 9,821 | 0 |

### Minimum n-gram 2, K = 7

Coding extra-time window: 0 to 6 µs.

| Extra time (µs) | 302 median TPS | 302 fastest TPS (B) | 302 ≥10,996 | 40 median TPS | 40 fastest TPS (A) | 40 ≥10,996 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5,000 | 11,949 | 1 | 5,346 | 10,699 | 0 |
| 13 | 4,795 | 11,495 | 1 | 5,126 | 10,294 | 0 |
| 21 | 4,681 | 11,232 | 1 | 5,000 | 10,060 | 0 |
| 26 | 4,612 | 11,074 | 1 | 4,924 | 9,919 | 0 |
| 6 (window upper end) | 4,900 | 11,735 | 1 | 5,243 | 10,508 | 0 |

### Native n-gram comparison

The [native-output validation](../experiments/lpx/gpu-cc302-ngram-k7-outputs-validation.json) covers all 302 requests: 301 stopped normally and one reached the prescribed 4,096-token limit.
The capped request is the same ID as in plain decoding, `speed-cc16k/6eed5048a6c5445fbb5a19c19dbc64f4`, and remains in the comparison.
No native answer is empty and no output triggers the recorded repetition checks.
Native decoding generated 206,306 tokens, averaging 683.13 per request, with a median measured H20 decode speed of 38.79 TPS.
The collector released its owned server after completing both runs.

The [comparison artifact](../experiments/lpx/gpu-cc302-ngram-comparison.json) uses the V2 GPU n-gram implementation at `K = 7`, minimum n-gram 1 and maximum n-gram 3.
It replays the native run's own exported combined text, preserving the earlier T4 comparison semantics; it does not compare native output against the separate plain-decoding answers used in C2.
The [source and control audit](../experiments/lpx/gpu-cc302-ngram-source-audit.json) verifies the four previously recorded source hashes and the plain and MTP control counter checks.
Each native request's tau is `(completion_tokens - 1) / (emitted_steps - 1)`, including decode steps with no proposal and excluding the prefill token and step.
The table averages per-request tau and derives equivalent acceptance from that mean, as T4 does.
This differs from the aggregate token-to-step ratio used for expectations 2 and 3.

| Metric over 302 requests | Native | Replay of native answers | Native minus replay |
| --- | ---: | ---: | ---: |
| Arithmetic mean tau | 1.795107 | 1.791638 | +0.003470 |
| Equivalent acceptance from mean tau | 0.443768 | 0.442675 | +0.001094 |

The mean absolute per-request acceptance difference is 0.001204 and the maximum is 0.008676, on `speed-cc16k/cb3573b06c7e40d386cf49ac29fa72ac`.
No individual request has an absolute acceptance difference of 0.03 or more.
Expectation 7 holds at the prescribed set level.

Both implementations select the longest n-gram and then the most recent match, but V2 fills missing draft slots with the last sampled token while replay uses variable proposal lengths.
Replay tokenizes exported text without serving chat-template tokens or hidden generated control tokens, and the combined-text export retains two separator newlines even with thinking off.
The comparison therefore supports approximate acceptance agreement for this configuration without establishing identical tokenization or per-request decoding paths.
The minimum-n-gram-2 settings remain replay-only, and this check does not validate simulated LP30 step times or identify NVIDIA's unpublished drafter.

### Expectations and interpretation

| Expectation | Outcome | Evidence |
| --- | --- | --- |
| 1: input hash and inclusion | Held | The expanded hash matches and all 40 earlier IDs are included among 302 unique completions |
| 2: earlier-40 average tau of 1.7 to 2.2 | Held | Aggregate tau is 2.020054; arithmetic mean per-request tau is 1.932896, also inside the interval |
| 3: all-302 average within 0.2 of earlier 40 | Held | Aggregate tau is 1.899691 versus 2.020054, a difference of 0.120363; arithmetic means are 1.802121 and 1.932896, a difference of 0.130775 |
| 4: at least 5% faster and 9,785 to 11,500 TPS | Missed | At minimum n-gram 1 and K = 7, the peak rises 7.97% from 10,966.49 to 11,840.79 TPS, exceeding the predicted upper bound |
| 5: no peak reaches 10,996 at a coding-window upper end | Missed | K = 7 reaches 11,605.59 TPS at 7 µs for minimum n-gram 1 and 11,735.05 TPS at 6 µs for minimum n-gram 2; each has one qualifying request |
| 6: every coding window ends at 21 µs or less | Missed | Minimum n-gram 1 and K = 3 extends to 23.5 µs; no setting fits the coding median at 26 µs, so that narrower conclusion still holds |
| 7: native equivalent acceptance within 0.03 of replay | Held | Native equivalent acceptance is 0.443768 versus 0.442675 for replay of its own answers, a difference of +0.001094 over all 302 requests |

The aggregate and arithmetic definitions give the same outcomes for expectations 2 and 3.
At minimum n-gram 1 and K = 7, request B commits 1,501 replay tokens in 367 steps, or 4.089918 tokens per step.
Expanding the prompt set supplies a normally terminated request that supports the coding-median-and-example part of the lookup hypothesis on the modeled deployment.
This result changes the earlier conclusion drawn from the 40 API answers; it does not establish which drafter NVIDIA used.
The jointly supporting settings have coding windows of 0 to 7 µs and 0 to 6 µs, so neither shares the 13 to 21 µs range cited for the earlier API-output fit.
Both K = 7 settings still exceed the example at 26 µs, but their coding medians no longer fit at that extra time.
No setting fits the H20 coding median at the 26 µs needed by the AA-like book answers, and the AA part remains unresolved.
At the two settings that reach the example, minimum n-gram 1 or 2 with `K = 7`, no extra time places both AA values within 5% on the [AA-like book answers](lpx-api-outputs.md), as [`api-aa-answer-alone-analysis.json`](../experiments/lpx/api-aa-answer-alone-analysis.json) records, and plain decoding needs at least 15.8 µs.
One deployment with one of these settings would therefore need AA prompts on which lookup accepts less than on the book answers.
The native comparison supports the text replay's acceptance approximation at minimum n-gram 1 and K = 7, with the accounting and tokenization differences recorded above.
The public checkpoint, BF16 outputs, prompt lengths, thinking setting and simulated step costs remain different from the unpublished LP30 deployment conditions.

### Validation and publication status

The full unit-test rerun passes all 776 tests, Black reports all 530 Python files formatted, and `git diff --check` passes.
All eight report tables were checked against the 78 priced results, and all 302 native acceptance records and set-level means were independently recomputed from the recorded counters.
The seven preregistered expectations are unchanged.
The GPU machine committed the study as `7540b49` and `1439744` but could not push them: its credential helper timed out, and its configured account lacks write access to the repository.
The commits were pushed from another machine after an independent check, which reproduced `gpu-cc302-peak.json` from the committed lookup replay with identical values and passed all 776 unit tests, Black and `git diff --check`.

## Collection and recurring checks

The [offline analysis command](../scripts/analyze_lpx_gpu_cc302.py) performs the prescribed exports and replay after the matching collection stages finish.
It checks the input hashes and full output coverage before exporting, requires each earlier ID exactly once, and keeps prompt-containing rows outside the repository.
It uses a separate analysis lock and does not acquire the GPU runner lock or launch inference.
Run `--phase lookup` once plain collection finishes, including while native collection is active, and `--phase native` after both collections finish.
The recorded lookup waiter has completed its replay.
The [native analysis waiter](../experiments/lpx/gpu-cc302-native-analysis-command.json) completed `--phase native` after both collections validated.
Its [status](../experiments/lpx/gpu-cc302-native-analysis-status.json) records successful completion; inspect the existing artifacts before starting any additional analysis process.
The native comparison verifies the earlier serving-source hashes and plain/MTP counter identities before reusing the T4 accounting interpretation.

```bash
/tmp/lpx-gpu/repo-venv/bin/python -u scripts/analyze_lpx_gpu_cc302.py \
  --work-dir /tmp/lpx-gpu \
  --requests /tmp/lpx-gpu/inputs/requests-cc302.jsonl \
  --phase all
```

The analysis also writes `gpu-cc302-peak-identities.json`, preserving the exported row order and all tied fastest-request identities for each priced setting without including prompt text.
The main pricing artifact retains the existing analyzer's output format and values.

The historical 1.96 value cited in expectation 2 is the aggregate ratio of output tokens to replay steps: 1.962380203 in the [API replay artifact](../experiments/lpx/api-lookup-results.json).
The arithmetic mean of the same 40 per-request ratios is 1.887417275.
Expectations 2 and 3 use the aggregate ratio to retain the cited historical definition; results will also report the arithmetic mean and identify any differing expectation outcome.
The existing pricing field `mean_tau` remains the arithmetic per-request mean.
These aggregation definitions are recorded while C1 collection is active and before replaying the new code-completion answers.

The [input audit](../experiments/lpx/gpu-cc302-inputs.json) verifies both recorded hashes, 302 unique code-completion requests and inclusion of all 40 earlier IDs.
The [input-context audit](../experiments/lpx/gpu-cc302-input-contexts.json) counts raw user-prompt tokens without special tokens or the serving chat template.
Across all 302 prompts the range is 18,071 to 21,421 tokens and the median is 19,559.5; the earlier 40 retain the previously reported range of 18,253 to 21,286 tokens.
These counts describe the input coverage and do not change the prescribed nominal-16K LP30 pricing model.
The original requests file remains unchanged; the default build is checked through a separate `requests-rebuilt.jsonl` file.
The [coding export record](../experiments/lpx/gpu-cc302-coding-export.json) identifies the 80-request, 89-turn control export outside the repository.

The [collector](../scripts/run_lpx_gpu_cc302.py) runs plain decoding and then native n-gram decoding under the existing `runner.lock`.
It records new server commands separately from the earlier study, keeps warmups separate from measurements using the prescribed request limits, validates all 302 outputs and releases its owned server after collection.
Its `complete` status describes collection only; the full workflow remains open for replay, analysis, documentation and publication.

```bash
/tmp/lpx-gpu/repo-venv/bin/python -u scripts/run_lpx_gpu_cc302.py \
  --work-dir /tmp/lpx-gpu \
  --requests /tmp/lpx-gpu/inputs/requests-cc302.jsonl \
  --stage all
```

The user requested status checks every 30 minutes until all tasks finish.
The [periodic timer](../scripts/monitor_lpx_gpu_periodic.py) submits a message to this existing Codex task every 1,800 seconds, including while GPU collection is running.
An immediate connection check successfully queued message `01a18c80-ed9e-7c11-b91a-6e46ecae9f9c` to task `01a18c06-9d7b-7971-a894-a413d645f829`.
On 2026-11-02, the existing timer was stopped and restarted for resumed task `01a19a00-a82b-7bb1-b373-ff0394c84652`, preserving its 1,800-second deadline and 126 prior deliveries.
The previous timer state is preserved outside the repository as `/tmp/lpx-gpu/cc302/periodic-state-before-resume-20261102.json`.
The resumed timer successfully queued message `01a19a05-9f0c-7362-b413-7625950c6406` to the resumed task at 10:10 Shanghai time, bringing its total to 127 deliveries.
The [timer command record](../experiments/lpx/gpu-cc302-timer-command.json) records the detached process and its external state and log paths.
The timer persists its deadline across restarts, rejects a second process on the same state file and retries known queue failures after 60 seconds.
A timed-out submission is recorded as uncertain and is not immediately duplicated.

```bash
/tmp/lpx-gpu/repo-venv/bin/python -u scripts/monitor_lpx_gpu_periodic.py \
  --workflow experiments/lpx/gpu-cc302-workflow.json \
  --task docs/lpx-cc16k-peak.md \
  --state /tmp/lpx-gpu/cc302/periodic-state.json \
  --thread 01a19a00-a82b-7bb1-b373-ff0394c84652 \
  --codex /home/tiger/.local/bin/codex \
  --interval-seconds 1800
```

Inspect the existing timer before running that command; it resumes a stopped timer and must not accompany another live timer on the same state file.
The timer stops only when `gpu-cc302-workflow.json` records the full workflow as `complete`, after C0 through C4 finish.
Queue submission requires the host and Codex app server to remain available; a message queued during an active turn may execute after that turn finishes.
The timer guarantees neither recovery after a host restart nor execution at an exact wall-clock instant.
