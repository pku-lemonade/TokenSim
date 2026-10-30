# TODO: measure Gemma 4 31B on a GPU machine for the LPX study

Record date: 2026-10-30.
Status: T0–T6 measurements and exploratory analyses are complete under the separately recorded Triton profiles; results and analyses are committed in `08d2dec`.
The [completed follow-up results](lpx-gpu-followup-results.md) cover all MTP priorities, context pairs, n-gram comparisons and coding spreads, including 23 capped thinking turns without final answers.
The [T6 profiling report](lpx-gpu-profile-results.md) covers both contexts and execution modes, with explicit residual activity and instrumentation limits.
The prior monitor delivered a failure continuation, but that session encountered an application error; work resumed in a new session at the user's request.

The [code-completion peak plan](lpx-cc16k-peak.md) adds tasks C0 to C4 for all 302 code-completion prompts, with its own expectations.

This page is a work plan for an agent on a machine with NVIDIA GPUs.
It runs the public Gemma 4 31B checkpoint with its released multi-token prediction (MTP) drafter and measures what the LPX study could only borrow from other models: Gemma's own drafter acceptance, its dependence on context length, the per-request speed spread it causes and the fixed costs of a real decode step.
Each finished task updates the checkbox in this page.

## Background

Artificial Analysis (AA) measured the Gemma4-31B-LPX deployment at a median of 3,382 output tokens per second (TPS) at 10K input tokens and 3,431 TPS at 100K input tokens, each from 50 sequential requests at concurrency 1.
The [final summary](lpx-aa-final-summary.md) and the [progress summary](lpx-aa-reproduction-progress.md) state the outcome so far.
The closest simulated setup is TP32/PP8 with decode context parallelism (DCP) of 8 on 256 LP30 chips, decoding one token per step without speculative decoding.
It needs 13 to 24 µs of unmodeled time per step at 1.006 Gemma tokens per AA token.

The case against speculative decoding for AA rests on one borrowed number.
The [article-condition speculation record](lpx-article-speculative.md) finds matches only at per-position acceptances of 0.006 to 0.335, and the MTP acceptance it compares with, 0.57 to 0.68 on prose, comes from Qwen3-Next on SPEED-Bench.
The [acceptance-evidence record](lpx-acceptance-evidence.md) notes that no published acceptance exists for Gemma 4 31B with its own MTP drafter, by task or by context length.
These GPU measurements supply that number.

A GPU cannot measure LP30 step times, so the 13 to 24 µs gap and the LP30 cross-tray latency stay simulator questions.
Every result below is priced on LP30 with the existing TokenSim results, as the earlier lookup records do.

## Ground rules

- Read [CONTRIBUTING.md](../CONTRIBUTING.md) and [AGENTS.md](../AGENTS.md) before changing anything; documentation follows the sentence layout and prohibited-wording rules there.
- Check out `origin/lpx-context-parallel`, commit to it, and push back to `lpx-context-parallel`.
- The expectations in [Expectations](#expectations) are recorded in this commit, before any measurement.
  Do not edit them after a run; report each one as held, partly held or missed.
  Record any new expectation in its own commit before the run it concerns.
- Report every setting that ran, and do not single out the best of several settings without saying how many were tried and that the choice was made after the results.
- Do not call a fit a reproduction of a published value, and do not say a stand-in result settles a published question.
  Each result section lists the [differences from the published runs](#differences-from-the-published-runs).
- Do not commit prompts, model weights, Hugging Face tokens or other credentials.
  Commit generated outputs without prompts, as [`experiments/lpx/api-gemma-outputs.jsonl`](../experiments/lpx/api-gemma-outputs.jsonl) does.
- Run Black, the unit tests and `git diff --check` before each commit that changes code.
- Keep new scripts at 500 lines or fewer and reuse the helpers named in [Pricing on LP30](#pricing-on-lp30).

## Environment

### Hardware

| Item | Minimum | Preferred |
| --- | --- | --- |
| GPUs | One 80 GB GPU (H100 or A100) running BF16 weights with a short context, for the smoke tests | Two 80 GB GPUs with tensor parallelism (TP) of 2, or one H200 or B200 |
| Context | 120,000 tokens per sequence, for a 100K prompt plus a 16,384-token output | Same |
| Disk | 150 GB for the target and drafter checkpoints and the vLLM cache | Same |

The vLLM Gemma 4 recipe lists at least one 80 GB GPU for BF16 weights and uses TP2 in its MTP example.
The BF16 weights take about 59 GB, so a 120,000-token context on one 80 GB GPU may not fit; use TP2 or an FP8 KV cache in that case and record the choice.

### Software

- vLLM with Gemma 4 MTP support.
  The recipe installs the nightly wheel or the `vllm/vllm-openai:latest` image.
  Record the exact version, the CUDA version and the driver version.
- The target checkpoint `google/gemma-4-31B-it` and the drafter checkpoint `google/gemma-4-31B-it-assistant` from Hugging Face.
  Both are public under Apache 2.0 as of 2026-10-28.
  Record the commit hash of each checkpoint.
- The repository environment from CONTRIBUTING.md, plus `tokenizers==0.21.1`, `pyarrow==17.0.0` and `tiktoken==0.9.0` for the prompt and measurement scripts.

Sources: [vLLM Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html), [vLLM MTP documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/), [vLLM speculative-decoding metrics](https://docs.vllm.ai/en/stable/api/vllm/v1/spec_decode/metrics/) and [Google's MTP drafter announcement](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/).

### Server

One server serves one configuration.
`--max-num-seqs 1` keeps every decode step at batch size 1, as in AA's measurement.
Prefix caching is off so that the 10K prompt, which is the start of the 100K prompt, does not change the 100K prefill.

```bash
vllm serve google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --max-model-len 120000 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --reasoning-parser gemma4 \
  --seed 1 \
  --speculative-config '{"method": "mtp", "model": "google/gemma-4-31B-it-assistant", "num_speculative_tokens": 3}'
```

The plain-decoding server omits `--speculative-config`.
The n-gram server of task T4 uses `'{"method": "ngram", "num_speculative_tokens": 7, "prompt_lookup_min": 1, "prompt_lookup_max": 3}'`.
Flag names change between vLLM releases, so check them with `vllm serve --help` and record the command that ran.
A startup log that shows `SpeculativeConfig(method='draft_model', ...)` for the assistant checkpoint means the release lacks Gemma 4 MTP support.

### Requests

Thinking is switched per request with `"chat_template_kwargs": {"enable_thinking": true}` or `false`, and the reasoning text returns in the `reasoning` field of the message.
Every request uses temperature 0, seed 1 and the request's `max_tokens`.
Requests stream, so the client records the arrival time of each chunk.

## Inputs

The prompts are the 264 requests of the [API-outputs record](lpx-api-outputs.md), rebuilt with [`scripts/build_lpx_api_prompts.py`](../scripts/build_lpx_api_prompts.py).

```bash
mkdir -p inputs/gutenberg
curl -sL -o inputs/speed-qualitative.parquet \
  https://huggingface.co/datasets/nvidia/SPEED-Bench/resolve/main/qualitative/test-00000-of-00001.parquet
curl -sL -o inputs/speed-16k.parquet \
  https://huggingface.co/datasets/nvidia/SPEED-Bench/resolve/main/throughput_16k/test-00000-of-00001.parquet
for id in 2701 1342 1400 98 145 1260 345 1399 1228 3300 1497 205 3207 1404 4280 944; do
  curl -sL -o inputs/gutenberg/pg$id.txt https://www.gutenberg.org/cache/epub/$id/pg$id.txt
done
curl -sL -o inputs/tokenizer.json \
  https://huggingface.co/google/gemma-4-31B-it/resolve/main/tokenizer.json
python -I scripts/build_lpx_api_prompts.py \
  --gemma-tokenizer inputs/tokenizer.json \
  --speed-qualitative inputs/speed-qualitative.parquet \
  --speed-16k inputs/speed-16k.parquet \
  --gutenberg inputs/gutenberg \
  --output inputs/requests.jsonl
sha256sum inputs/*.parquet inputs/tokenizer.json inputs/gutenberg/*.txt inputs/requests.jsonl
```

Keep `inputs/` outside the repository or leave it untracked.
The files used on 2026-10-27 have these SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `requests.jsonl` | `7abb2f93587b01649c27e262e041afe96dd68cc055908d80f35610b5de9506ae` |
| `speed-qualitative.parquet` | `4f76bc45bdffb38712a5c26d7a7c9a9a791484b392a7b64a6ffb9abe48e02e26` |
| `speed-16k.parquet` | `7e72b4b352c9ebb65f7847c62c446d93a8d2ff049c74d9a158a3c7e3d6ceb8b1` |
| `tokenizer.json` | `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` |
| `pg98.txt` | `d54c2b80d40a40b982cd88852c6180bb944d95acdb028af3d0e01a1750681784` |
| `pg145.txt` | `dc2e0107e6ae07e8e33da934b2ec4e8e600826a2e0cd48df265119cd02e5fb50` |
| `pg205.txt` | `2d9a76a2e3e8195c69430516ebd33c4d0757a53ad432ff6186b7b794e6fe99f9` |
| `pg345.txt` | `96cd16eacdbfebae8fdda5591f66e0cc8ee76be18e0cd1aca02bc00615782d28` |
| `pg944.txt` | `e13d57170a6dd9f7d97ddbc14cbde961dffbd95e8861ed199e5f00a525e7e52a` |
| `pg1228.txt` | `362142cb7c6f90850477520cfe2a62fe423ae92ea8eb6d9cd8b30ddb39de60b9` |
| `pg1260.txt` | `13414dee2951c3ee731d76d2ffd822016b2479c892162760c5d0eb2aa5fa7631` |
| `pg1342.txt` | `3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75` |
| `pg1399.txt` | `a3e29e08c15c63bb22069f4256316c2ebebd422ecb5c33462ddc00ae00b27571` |
| `pg1400.txt` | `9a637118af8e953e9764ec603d9b0a032883384d465acac2e27966a80cf1c6f8` |
| `pg1404.txt` | `0b92fadc9d7afd767eefd27805a20e44417ff26d1fb2f964229f39dcc7b2a65d` |
| `pg1497.txt` | `3d2e397122881d7737a48fb80e71cddf4c78cda54ceac7e0905fe945dfab48fd` |
| `pg2701.txt` | `b5e88b71a1cda068e9ab1d94344c9d932338ba6da1a66def97a1aecd9cfcc87d` |
| `pg3207.txt` | `3de1e492641d939567a8b0de827fb13e1ad992324f9ac8b204f60475009b7294` |
| `pg3300.txt` | `e91d52dca43fd0fd69baf7776b00a5d8e74b69675b6b608ca8a0378aaf3ae58d` |
| `pg4280.txt` | `9e44b1bc6042da4aeb669d4f3f585037a5488a2908e7ec225f9431c43f6720fc` |

Project Gutenberg sometimes revises its files.
If a book's hash differs, keep the new file and record the difference, because the request hash then differs too.

## Client and per-request record

Extend [`scripts/generate_lpx_gemma_outputs.py`](../scripts/generate_lpx_gemma_outputs.py), or add a small script that reuses its request loop, with these changes:

- Send requests to a local OpenAI-compatible endpoint given by an `--api-base` option, without the OpenRouter `provider`, `usage` and `reasoning` fields.
- Map the request's `reasoning` flag to `chat_template_kwargs.enable_thinking`.
- Stream each response and record the arrival time of every chunk.
- Run at concurrency 1 and read the server's `/metrics` endpoint before and after each turn, storing the differences of the speculative-decoding counters.
- Keep the resumable JSON Lines output and the `--export` rows format, so that [`scripts/measure_lpx_lookup_acceptance.py`](../scripts/measure_lpx_lookup_acceptance.py) and [`scripts/measure_lpx_token_ratio.py`](../scripts/measure_lpx_token_ratio.py) read the outputs unchanged.

In recent vLLM releases the counters are `vllm:spec_decode_num_drafts`, `vllm:spec_decode_num_draft_tokens`, `vllm:spec_decode_num_accepted_tokens` and `vllm:spec_decode_num_accepted_tokens_per_pos`, the last with one series per draft position.
Older releases use other names, so confirm them on the running server.

Each turn's record holds:

| Field | Content |
| --- | --- |
| `id`, `turn`, `set` | Request identity, as in the API outputs |
| `config` | `plain`, `mtp` or `ngram`, with `K`, weight and KV-cache precision |
| `prompt_tokens`, `completion_tokens` | From the response usage |
| `reasoning`, `content`, `finish_reason` | Generated text and stop reason |
| `ttft_s` | Time from sending the request to the first chunk |
| `decode_s` | Time from the first chunk to the last chunk |
| `output_tps` | `(completion_tokens - 1) / decode_s`, the AA definition of output speed |
| `drafts`, `draft_tokens`, `accepted_tokens`, `accepted_per_position` | Counter differences over the turn |

Check on a few turns that `completion_tokens` equals `drafts + accepted_tokens` plus a small constant for the prefill token, and document the identity that holds.
The committed tokens per step are `tau = (completion_tokens - 1) / drafts`, and [`equivalent_acceptance`](../scripts/measure_lpx_lookup_acceptance.py) converts `tau` and `K` to one per-position acceptance.

## Tasks

Run the tasks in order.
Before T1, measure the decode speed of one 10K request and estimate the run time of each task, because the full plan generates several million tokens at batch size 1.
If time is short, run the priority-1 rows before the others and record which rows did not run.

### T0: setup and smoke tests

- [x] Start the plain server, send one 10K `aa-docs` request with thinking on and record the software versions, checkpoint hashes and GPU model.
- [x] Start the MTP server at `K = 1` and confirm the startup log reports the MTP method and that `/metrics` exposes the speculative-decoding counters.
- [x] Send the first 10 `speed-coding` prompts without thinking to the plain server and to the MTP server at `K = 3`, and record how many outputs are identical and, for the others, the first differing token position.
- [x] Confirm the counter identity of [Client and per-request record](#client-and-per-request-record).

### T1: MTP acceptance by draft length

Run each set at concurrency 1 on the MTP server.
`K` takes the values 1, 3, 5 and 7, for which [`experiments/lpx/article-speculative-results.json`](../experiments/lpx/article-speculative-results.json) holds LP30 steps.

| Priority | Set | Thinking | `K` |
| --- | --- | --- | --- |
| 1 | `aa-docs`, 64 requests | On | 3, 7 |
| 1 | `speed-coding`, 80 prompts | Off | 3, 7 |
| 2 | `speed-coding`, 80 prompts | On | 3, 7 |
| 2 | `aa-docs` | On | 1, 5 |
| 3 | `speed-cc16k`, 40 prompts | Off | 3, 7 |
| 3 | `speed-coding` | Off and on | 1, 5 |

- [x] Complete priority 1 under the separate Triton profile: 64 document requests and 80 coding prompts at each of K=3 and K=7.

The [Triton acceptance artifact](../experiments/lpx/gpu-triton-mtp-acceptance.json) contains 288 requests and 306 turns, including nine follow-up coding turns per configuration.
All turns stop normally without empty answers or repetition flags.
The original serving-profile results remain separate.
The [combined analysis](../experiments/lpx/gpu-triton-followup-remaining-mtp-acceptance.json) also completes priorities 2 and 3, retaining all quality flags.

- [x] Write prompt-free turn records, using separate `gpu-triton-mtp-k*-outputs.jsonl` and `gpu-triton-followup-mtp-k*-outputs.jsonl` files to preserve each run's identity.
- [x] Write `experiments/lpx/gpu-triton-followup-remaining-mtp-acceptance.json` with, for each set, context, question style, thinking mode and `K`: the turn count, mean `tau`, equivalent acceptance, per-position acceptance and the per-turn rows `[decode_tokens, counts]`, where `counts` has `K + 1` entries and only the last, the number of drafts, is nonzero because MTP verifies `K + 1` rows in every step.
  That row format is the one [`lookup_speeds`](../scripts/analyze_lpx_coding_spread.py) and the API-outputs analysis read.

### T2: acceptance at 10K and 100K

- [x] From the available T1 `aa-docs` turns at K=3 and K=7, pair each book's 10K and 100K turns per question style and `K`, as `paired_acceptance` in [`scripts/analyze_lpx_api_outputs.py`](../scripts/analyze_lpx_api_outputs.py) does for lookup, and report the mean difference and the number of books whose acceptance is higher at 100K.
- [x] Report the acceptance of the thinking and the answer parts separately where the counters allow it; if they do not, report the combined value only and say so.

The [completed pairing](lpx-gpu-measurements.md#completed-triton-priority-1-results) covers all 16 books for both styles at K=3 and K=7.
Counters support combined thinking and answer acceptance only.
The [completed follow-up pairing](lpx-gpu-followup-results.md#paired-context-comparison) adds K=1 and K=5 for both styles and all 16 books.

### T3: FP8 weights

AA's deployment ran FP8 weights.

- [x] Repeat the priority-1 `aa-docs` row at `K = 3` with FP8 target weights, for example with `--quantization fp8`, and record the precision of the drafter and the KV cache.
- [x] Check whether vLLM rejects MTP with FP8 weights and record the outcome; skip the task if rejected.

The installed vLLM accepts FP8 target weights with the BF16 assistant and KV cache, and all 64 document turns complete normally without quality flags.
The [completed comparison](lpx-gpu-measurements.md#completed-fp8-comparison) gives combined FP8 minus BF16 equivalent acceptance of -0.008278, with all four group differences below 0.03 in magnitude.

### T4: vLLM n-gram drafting against the lookup replay

The lookup records replay an n-gram drafter on fixed outputs; this task checks the replay against a running drafter.

- [x] Run the priority-1 `aa-docs` and `speed-coding` rows on the n-gram server at `K = 7`, with `prompt_lookup_min = 1` and `prompt_lookup_max = 3`.
- [x] Export the same turns and replay them with `scripts/measure_lpx_lookup_acceptance.py` at `n_min = 1` and `K = 7`, and compare committed tokens per step turn by turn.
- [x] Describe any difference in matching rules between vLLM and the replay, from the vLLM source, before interpreting the comparison.

The active T4 run pins V2: native and replay matching both prefer the longest n-gram and most recent occurrence, but V2 pads unavailable draft slots with the last sampled token.
The V1 CPU implementation prefers the earliest occurrence.
The [source audit](../experiments/lpx/gpu-triton-followup-ngram-source-audit.json) and [measurement record](lpx-gpu-measurements.md#acceptance-and-lp30-pricing) retain these distinctions and the tokenization differences.
The [completed coding comparison](lpx-gpu-measurements.md#completed-n-gram-coding-comparison) covers all 80 requests and 89 turns, with native equivalent acceptance 0.360232 and replay 0.359954.
Their difference of 0.000278 is below 0.03.
The [full comparison](lpx-gpu-followup-results.md#native-n-gram-versus-replay) also completes all 64 document turns, with a difference of -0.000205.

### T5: per-request speed spread at concurrency 1

The [coding spread comparison](lpx-gpu-measurements.md#coding-speed-spread) covers all 89 turns for plain decoding and priority-1 K=3/K=7 MTP, including LP30 pricing at both drafter costs.
H20 P80/median is 1.005874 for plain and 1.065741 for K=3, satisfying expectation 8.
The [follow-up coding tables](lpx-gpu-followup-results.md#coding-speed-spreads) complete all remaining T1 coding configurations at both drafter prices.

- [x] From the `speed-coding` turns of T1 and of one plain run of the same prompts, compute the median, the 80th percentile (P80) and the maximum of `output_tps` per configuration, and the P80/median and maximum/median ratios.
- [x] Price the same turns on LP30, as in [Pricing on LP30](#pricing-on-lp30), and report the same ratios.
- [x] Compare both with NVIDIA's SPEED-Bench coding ratios of 1.158 (P80) and 1.464 (maximum), and state that the GPU ratios include GPU step times and the LP30 ratios include only the measured acceptance.

### T6: fixed costs of a decode step

This task is exploratory and has no expectation.
It sizes the per-step costs that the LP30 cost model may omit, on a real serving stack.
The [profiling preflight](../experiments/lpx/gpu-t6-preflight.json) confirms Nsight Systems 2025.6.3 and the installed vLLM PyTorch-profiler configuration and endpoints.
The endpoints require profiling to be enabled at server startup, so this task needs a separate plain server after acquiring the GPU lock.
The [T6 runner](../scripts/run_lpx_gpu_profile.py) is queued after the full follow-up sweep, with separate queue and GPU locks.
It collects four profiled cases, four unprofiled controls and two warmups, keeping traces outside the repository.
Both graph cases and the eager 10K case have been collected; the latter was recovered from valid traces after its profiler-stop request timed out.
The missing eager 100K case completes with optional CUDA summary-table generation disabled.
All four cases have GPU partitions and separate, clock-aligned host/streaming audits in the [profiling report](lpx-gpu-profile-results.md).
Follow the [trace-analysis workflow](lpx-gpu-trace-analysis.md) after collection, including explicit prefill/decode boundaries, overlap-aware GPU categories and separate CPU/streaming interpretation.

- [x] On the plain server, run one 10K and one 100K `aa-docs` prompt with `max_tokens` of 256 under the PyTorch profiler or Nsight Systems.
- [x] Report mean decode costs and fractions for model forward, the 262,144-token vocabulary head, logits/sampling, gaps and unclassified GPU activity, with separate overlapping CPU scheduling and output-streaming scope measurements.
- [x] Repeat with `--enforce-eager` to show the effect of execution mode on the gaps, retaining paired unprofiled controls and instrumentation-overhead caveats.
- [x] Record that GPU host costs do not transfer to LP30 directly and serve only as a size reference for the per-step cost audit.

Inclusive CPU and streaming scopes overlap GPU execution and cannot form additional additive rows of the GPU partition.
The report leaves causal attribution of uninstrumented host costs unresolved instead of equating all GPU gaps with scheduling.

### T7: TokenSim on the same GPU (optional)

- [x] Check whether TokenSim's operator data in `data/operator_data/` covers the GPU type and whether the Gemma 4 model in [`data/models/gemma-4-31b-text.yaml`](../data/models/gemma-4-31b-text.yaml) runs on it.
- [x] If both hold, compare the simulated batch-one decode step at 10K and 100K with the plain-server step of T6; otherwise record why the check did not run.

The GPU is an H20 and `data/operator_data/h20` does not exist, so T7 does not run.

## Pricing on LP30

Add `scripts/analyze_lpx_gpu_acceptance.py`, which prices the measured MTP acceptance on TP32/PP8 with DCP 8 at attention matrix efficiency 0.50, as the API-outputs analysis does for lookup.

- Read the LP30 steps with `load` from [`scripts/analyze_lpx_published_conditions.py`](../scripts/analyze_lpx_published_conditions.py).
  The rows of `article-speculative-results.json` with `layout_name` `tp32-pp8`, `attention_mfu` 0.5 and `drafter` `mtp-0.05` or `mtp-0.10` give `step_us` per workload for each `K`, including the drafter's own time at 5% or 10% of a one-token step per draft token.
  Report both drafter costs.
- Price each AA workload as `tau / step_us * 1e6 / ratio`, with `tau` measured at its own context and the token ratio of 1.006 from [`experiments/lpx/api-token-ratio-results.json`](../experiments/lpx/api-token-ratio-results.json), or the ratio measured on the new outputs if it differs by more than 0.01.
- Reuse `window` and `priced` from [`scripts/analyze_lpx_api_outputs.py`](../scripts/analyze_lpx_api_outputs.py) to report the AA errors without extra time and the extra time per step that places both AA values within 5%.
- For coding, add the drafter time of each row to the verification steps of `context_model` in [`scripts/analyze_lpx_coding_spread.py`](../scripts/analyze_lpx_coding_spread.py), price each turn and report the median against 4,767 TPS together with the P80 and maximum ratios.
- Add unit tests for any new function that computes a step or a speed.

## Expectations

These expectations are recorded before any GPU measurement.

1. T0: at least 9 of the 10 plain and MTP output pairs are identical.
2. T1: MTP at `K = 3` on `aa-docs` with thinking on gives an equivalent acceptance of 0.45 to 0.80.
3. T1: at the same `K`, `speed-coding` without thinking accepts at least 0.05 more than `aa-docs`.
4. T2: the mean paired difference between 100K and 10K acceptance is below 0.03 for both question styles at `K = 3`.
5. Pricing: at the measured `aa-docs` acceptance and either drafter cost, TP32/PP8 with DCP 8 is more than 30% too fast at AA 10K without extra time, and no `K` places both AA values within 5% with 25 µs or less of extra time per step.
6. T3: FP8 weights change the `aa-docs` acceptance at `K = 3` by less than 0.03.
7. T4: vLLM's n-gram drafter and the replay differ by less than 0.03 in equivalent acceptance on each set.
8. T5: on the GPU, MTP at `K = 3` gives a `speed-coding` P80/median ratio of 1.03 to 1.20, and plain decoding gives below 1.02.

## How results bear on the open questions

| Result | Reading |
| --- | --- |
| `aa-docs` acceptance at `K = 3` of 0.5 or more | Gemma's own drafter would make the closest setup much faster than AA at both contexts, so drafting with this drafter fits AA worse than plain decoding unless the LP30 step is much longer than modeled |
| `aa-docs` acceptance of 0.35 or less | The main support for plain decoding in AA weakens, and speculative setups on TP32/PP8 return as candidates |
| 100K acceptance at least 0.05 above 10K | A real drafter could supply part of AA's 1.45% higher speed at 100K, which lookup drafting could not |
| A `speed-coding` P80/median ratio near 1.158 with MTP | A real drafter at batch size 1 can produce NVIDIA's coding spread on the public build |
| Step costs outside the transformer layers of 10 µs or more per step | Supports the existence of per-step costs that the LP30 cost model omits, and sizes them for the cost audit |

None of these results identifies NVIDIA's build, drafter or settings.

## Differences from the published runs

| Item | Published runs | GPU measurements |
| --- | --- | --- |
| Model build | AA: an "updated private, pre-release" Gemma4-31B-LPX build; SPEED-Bench: not stated | Public `google/gemma-4-31B-it` checkpoint |
| Hardware | LP30 chips in an LPX rack | NVIDIA GPUs; LP30 speeds come from TokenSim pricing |
| Precision | FP8, according to press reports | BF16, and FP8 in T3 |
| Drafter and `K` | Not stated | Google's released MTP assistant at `K` of 1 to 7, and vLLM's n-gram drafter |
| Prompts | AA's prompts are unpublished; SPEED-Bench's split is not stated | Project Gutenberg books with questions written for this study, and SPEED-Bench's qualitative coding and 16K code-completion prompts |
| Thinking mode | Not stated | On for `aa-docs`; on and off for `speed-coding`; off for `speed-cc16k` |
| Sampling | Not stated | Temperature 0, seed 1 |
| Load | AA: concurrency 1; SPEED-Bench: not stated | Concurrency 1 and batch size 1 |

## Recording

- [x] Write `docs/lpx-gpu-measurements.md` with the environment, commands, input hashes, each expectation's outcome, result tables, the differences table above and an interpretation limited to what the data shows.
- [x] Commit the outputs and analyses as `experiments/lpx/gpu-*.json` and `experiments/lpx/gpu-*.jsonl`.
- [x] Update the outcome table, current position and open work of the [progress summary](lpx-aa-reproduction-progress.md).
- [x] Add the publication commit to the progress summary's commit table after committing the results.
- [x] Update the [final summary](lpx-aa-final-summary.md) only where a result changes a stated conclusion.
- [x] Tick the boxes in this page and change its status line.

The [progress summary](lpx-aa-reproduction-progress.md#commits) records the publication commit.
Generated outputs and analyses are committed without input prompts, model weights, credentials or raw profiler traces.
