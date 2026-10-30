# T6 trace analysis

The [GPU TODO](lpx-gpu-gemma-todo.md#t6-fixed-costs-of-a-decode-step) requires decode attribution after the queued graph/eager collection finishes.
[`analyze_lpx_gpu_trace.py`](../scripts/analyze_lpx_gpu_trace.py) computes an overlap-aware GPU wall-time partition from a collected case and an explicit selection file.
It does not infer decode boundaries or semantic labels from kernel names, and its output remains `analysis_status: gpu_partition_only`.
The [measured T6 results](lpx-gpu-profile-results.md) use inspected generation annotations, CUDA correlations and matrix shapes, with separate host/streaming audits and instrumentation limits.

## Inspect the collected case

Use the `gpu-triton-t6-{mode}-{context}-result.json` manifest for the requested graph/eager mode and 10K/100K context.
Its `outputs` list contains the unprofiled control followed by the profiled request.
Its `traces` list identifies the external trace files and their SHA-256 hashes.
The analyzer verifies every output and trace record before reading the selected trace and checks input identities against the profiled output.
Raw traces and prompt-containing files must remain outside the repository.

Inspect worker and frontend traces separately before aligning their clocks.
In the installed vLLM worker, `annotate_profile` constructs `execute_context_..._generation_...` annotations from the scheduler's context and generation request counts.
With detailed annotations, `sq`, `sk`, `sqsq` and `sqsk` record scheduled-token and sequence-length quantities.
These annotations provide evidence for separating chunked prefill from generation, but CPU scope boundaries alone do not prove that asynchronous GPU work has completed.
Use kernel, launch and flow/correlation evidence to audit the decode interval and its final completion boundary.
Exclude warmup, prefill and profiler export activity from decode windows.

## Selection format

Create a selection file outside the repository after inspecting the actual trace.
`trace` must identify exactly one trace in the case manifest, and `trace_sha256` must match that trace's recorded hash.
`decode_steps` uses Chrome trace timestamps in microseconds, regardless of a viewer's display units.
There must be one window for each actual decode token, equal to the profiled completion-token count minus the prefill token.
Windows must be finite, positive, ordered and contiguous so that inter-step gaps cannot disappear from the reported wall time.
The `basis` fields must explain the boundary and category evidence without copying prompt text.
The analyzer validates the count and structure of the windows; their semantic correctness still requires trace inspection.

The following two-step selection is synthetic and illustrates the schema only.
Replace its timestamps, event indices, hash and evidence with values from the measured case; a 256-token completion requires 255 decode windows.

```json
{
  "trace": "/tmp/lpx-gpu/trace.json.gz",
  "trace_sha256": "REPLACE_WITH_RECORDED_SHA256",
  "basis": "Replace with prefill/decode boundary and final GPU completion evidence",
  "decode_steps": [
    {
      "start_us": 10,
      "end_us": 20
    },
    {
      "start_us": 20,
      "end_us": 30
    }
  ],
  "categories": [
    {
      "name": "transformer_layers",
      "basis": "Replace with module, launch and correlation evidence",
      "event_indices": [42, 43]
    }
  ]
}
```

Category event indices are zero-based positions in the original `traceEvents` array, including metadata events in that indexing.
Supported semantic categories are `transformer_layers`, `vocabulary_head` and `logits_sampling`.
Every assigned event must be a GPU activity intersecting the selected decode interval, and each event can have only one assignment.
Omit a category when the trace does not support its identification; unassigned activity remains `unclassified_gpu`.
Do not classify an ambiguous matrix multiplication as the vocabulary head based only on its kernel name.
Graph replay can lack the eager trace's semantic annotations, so audit any mapping between modes using shapes, execution order and correlation evidence.

## Run the analysis

```bash
/tmp/lpx-gpu/repo-venv/bin/python scripts/analyze_lpx_gpu_trace.py \
  --case experiments/lpx/gpu-triton-t6-graphs-10000-result.json \
  --selection /tmp/lpx-gpu/t6-graphs-10000-selection.json \
  --summary experiments/lpx/gpu-triton-t6-graphs-10000-gpu-partition.json
```

The analyzer refuses to overwrite the case, selection, traces or generated outputs.
The summary records their provenance and retains the control/profile streamed-timing comparison.
It also records per-step partitions and clipped activity-name duration sums for inspection.

## Interpret the partition

GPU kernel, memory-copy and memory-set activity contributes to the union of GPU-active wall time on one device lane.
Overlapping activity within the same category counts once.
Concurrent activity in distinct categories contributes to `gpu_category_overlap` instead of being charged twice.
The remaining wall time is `no_gpu_activity`; the tool does not equate that interval with CPU scheduling, output streaming or device idleness outside the selected trace.
Each partition reports total microseconds, mean microseconds per selected decode step and its fraction of the selected wall interval.
Activity-name sums are non-additive diagnostics because concurrent kernels and copies can overlap.
CPU scopes are excluded from this GPU partition and cannot be added to it as if CPU and GPU execution were serial.

To finish T6, audit the boundary selections and semantic assignments, examine CPU scheduling and frontend streaming with justified clock alignment, and report overlap and unclassified time explicitly.
Compare the graph/eager cases and unprofiled controls at both context lengths, accounting for profiler overhead and any control/profile token differences.
Keep the GPU partition and client delivery timing distinct.
Document remaining attribution limits instead of assigning unexplained time to a convenient category.
GPU host costs are a size reference and do not directly transfer to LP30.
