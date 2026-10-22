# LPX integration verification

Integration date: 2026-10-21.
The branch is `lpx-experiment`.
Local code, research reports and portable results were saved in commit `533a487` before integration.
The integration incorporated all 24 remote commits through original revision `7ce890be3fe2abe0e16fa6f862fa2c0e830cd053`.
Those original revision identifiers describe the integration before the subsequent branch-scoped language cleanup rewrote history.
Raw experiment outputs and the local virtual environment remain on the original machine under ignored paths.

## Preserved capabilities

The local [fidelity study](lpx-fidelity-model.md) retains its execution-unit accounting, routed communication, live-memory checks, token conversion, nonuniform pipeline partitions and audited results.
The remote changes retain explicit collective algorithms, feed-forward network (FFN) batching, expert tensor-parallel validation, consistent execution assumptions, capacity-search options, literature reviews and experiment exports.
Tensor parallelism (TP) splits work within a layer, pipeline parallelism (PP) assigns layers to stages, and expert parallelism (EP) distributes experts.

Three files required textual conflict resolution:

| File | Integrated behavior |
| --- | --- |
| `scripts/lpx/native_config.py` | Preserves local activation validation, colocated dense reuse and stage partitions together with remote collective selection, expert mapping and explicit FFN batching settings. |
| `scripts/lpx/native_simulation.py` | Forwards dense-reuse and batching settings together and retains maximum-context accounting. |
| `scripts/lpx/replay_profiles.py` | Retains streaming refinement and invocation costs, rejects dynamic FFN batching in fixed directed acyclic graph profiles, and stops capture when the requested forward operation completes. |

The dense-reuse multi-replica test now declares the supported EP-only FFN layout.
Its rank mapping still checks local and remote replica ownership.
The remote rejection of unsupported mixed expert TP/EP prefill remains covered by its original tests.

## Validation

The complete merged suite passes 626 tests.
Black formatting and its check pass for 466 Python files.
Tracked source, test and script files satisfy the 500-line limit.
Tracked JSON files parse successfully, and `git diff --check` passes.
Seven LPX command-line entry points pass their `--help` smoke checks.

The representative verification uses a fresh checkpoint directory:

```bash
.venv/bin/python scripts/validate_lpx_fidelity.py \
  --config experiments/lpx/fidelity-study.json \
  --tp 16 \
  --pp 8 \
  --output outputs/lpx-integration-20261021/validation
```

The [historical portable results](../experiments/lpx/fidelity-results.json) retain their original source fingerprint and provenance.
The full 180-layout search and 14-ablation study are historical runs.
This integration verification repeats the 128-chip representative candidate, including complete prefill and serial decode checks.

The [integrated portable results](../experiments/lpx/integration-validation.json) record the fresh run, input assumptions, source fingerprints, historical comparison and independent checks.
The integrated source fingerprint is `0b74d51dbed7c566573e6fd936ddd293297b5a858722d6475f828bbf788170e4`.
The historical fingerprint remains `2686611479adfc9af8ce401de376b995ee9d9fb2b21d57bd9c2824b412236119`.
The fresh run does not reuse an earlier checkpoint.

Throughput is output tokens per second (TPS), with time to first token excluded.
Both runs use the bundled unmeasured identity conversion between native and Artificial Analysis token units.

| Input / output tokens | Historical TPS | Integrated TPS | Relative change |
| --- | ---: | ---: | ---: |
| 1,000 / 1,000 | 5,908.784586764318 | 5,908.784586764318 | 0% |
| 10,000 / 1,500 | 5,578.406484068661 | 5,578.406484068661 | 0% |
| 100,000 / 2,000 | 3,611.167910635546 | 3,611.167910635546 | 0% |

The layer counts remain `[6, 8, 9, 7, 7, 10, 7, 6]` at TP16 PP8.
Peak modeled per-chip memory remains 440,169,038 bytes, with minimum headroom of 59,830,962 bytes.
The complete 100K prompt executes 6,250 chunks and reaches context 100,000.
Its simulated prefill time remains 6.3975110678706395 seconds.
Independent checks verify three latency curves, 67 fitting probes, 43 withheld probes and 247,335 subsequent-token steps across three serial workload replays.
Each workload uses one client, five warm-up requests and 50 measured requests.
The fresh validation takes 416.55 host seconds on the current machine.
Host elapsed time depends on machine load and is recorded separately from simulated latency.
These checks establish consistency of the integrated simulator under the selected assumptions; hardware accuracy and endpoint allocation remain uncalibrated.

## English documentation and branch history

The language cleanup translates and renames the research documents and model provenance note.
The current tracked UTF-8 text and filenames contain no Han ideographs.

| File | Content |
| --- | --- |
| [DeepSeek model catalog](../data/models/deepseek-v3-671b-mla-model.yaml) | English simulation provenance note with unchanged model ID and numerical parameters. |
| [Groq hardware evidence](groq-hardware-evidence.md) | Literature inventory, hardware generations and limits on published evidence. |
| [LPX model corrections](lpx-model-corrections.md) | Literature-constrained implementation corrections and retest results. |
| [LPX study guide](lpx-study-guide.md) | Calibration, methodology, commands and experiment records. |

Model loading, package data and document references use the renamed paths.
The [language validation record](../experiments/lpx/integration-language-validation.json) records scan scope and content-preservation checks.
The cleanup removes retired file paths from reachable history on `lpx-experiment` only.
Other branches, external clones and hosting-service caches can retain historical copies.
Original experiment fingerprints and numerical results retain their generation-time provenance.
The scan covers tracked UTF-8 text and filenames, excluding ignored outputs, virtual environments, Git metadata, binaries and non-UTF-8 files.
Image text is outside this text scan's scope.
