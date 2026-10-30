# Does a lookup drafter reach the 10,996 TPS code-completion example on all 302 prompts?

Record date: 2026-10-30.
Status: expectations and GPU work plan recorded; no measurement has run.

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
| Prompt | One unpublished prompt with a 16K input | SPEED-Bench's 302 code-completion prompts, of 18,253 to 21,286 Gemma tokens in the earlier sample |
| Thinking and sampling | Not stated | Thinking off, temperature 0 |
| Step times | Measured on LP30 | Simulated by TokenSim, with a constant extra time per step |

## GPU work plan

The agent on the GPU machine follows the ground rules of the [GPU TODO](lpx-gpu-gemma-todo.md#ground-rules): it pushes to `lpx-context-parallel`, keeps prompts and rows outside the repository, does not edit the expectations above and reports every setting that ran.
The plain run takes about 2 hours: 302 answers of about 686 tokens, the mean of the earlier sample, at about 45 TPS, the speed of the H20 plain control.
The native n-gram run takes less.

### C0: inputs

- [ ] Rebuild both request files from the inputs of the GPU TODO and check their hashes.

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

- [ ] Start the plain Triton server with the arguments of [`gpu-triton-followup-plain-k0-server-command.json`](../experiments/lpx/gpu-triton-followup-plain-k0-server-command.json).
- [ ] Run [`scripts/generate_lpx_gpu_outputs.py`](../scripts/generate_lpx_gpu_outputs.py) with `--requests inputs/requests-cc302.jsonl`, `--set speed-cc16k`, `--thinking off`, `--method plain` and the run label `cc302-plain`.
- [ ] Commit the outputs without prompts as `experiments/lpx/gpu-cc302-plain-outputs.jsonl`, with the server command and a validation record of coverage, finish reasons and repetition flags, as for the earlier GPU runs.

### C2: lookup replay

- [ ] Export the rows of the 302 answers, outside the repository.
- [ ] Write a second rows file that holds only the 40 earlier IDs.
- [ ] Export the rows of the plain coding control without thinking, `gpu-triton-followup-plain-k0-outputs.jsonl`, joined with `inputs/requests.jsonl`.

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

- [ ] Start the n-gram server of task T4 at `K = 7` with the Triton profile, run the same 302 requests with `--method ngram --draft-tokens 7` and the run label `cc302-ngram-k7`, and compare its equivalent acceptance with the replay as [`scripts/analyze_lpx_gpu_ngram.py`](../scripts/analyze_lpx_gpu_ngram.py) does.

### C4: recording

- [ ] Add a results section to this record: one table per setting with the coding window and the code-completion median, fastest request and count at or above 10,996 TPS at each priced extra time, the same for the 40 earlier prompts, and the outcome of each expectation.
- [ ] Add a row to the outcome table of the [progress summary](lpx-aa-reproduction-progress.md) and update its current position.
- [ ] Run Black, the unit tests and `git diff --check`, commit and push.
