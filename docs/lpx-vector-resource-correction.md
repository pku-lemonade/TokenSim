# LP30 vector-resource reservations for scalar kernels

With explicit LP30 fabric refinement enabled, native decode assigns the physical vector execution module (VXM) resource to every LP30 compute event.
The previous operation-name list covered attention, attention output, dense feed-forward work and the language-model head.
Normalization, residual, conversion and selection events retained only their worker resource.
Chip-to-chip (C2C) reductions separately reserved the VXM resource, so independent requests could execute a scalar kernel and a reduction concurrently on the same modeled vector unit.
The [streaming refinement](../scripts/lpx/streaming.py) now derives vector reservations from compute-event type and physical LP30 endpoints.
The compute-event binding leaves wire transfers and non-LP30 compute unchanged.
Collective reductions retain their explicit VXM work.

The correction preserves operator service durations and directed acyclic graph (DAG) dependencies.
It completes resource exclusions in the existing model, which reserves each declared compute resource for the complete modeled kernel duration.
This remains a conservative kernel-level model and does not establish physical execution-unit occupancy or compiler scheduling.

## Verification

The [resource tests](../tests/test_lpx_vector_resources.py) reproduce the missing scalar bindings and their effect on contention.
A unit-test control submits a one-time-unit normalization and a one-time-unit reduction on the same chip.
The unbound model completes both at time one, while the corrected model completes them at times one and two.
Additional cases cover scalar operation types, existing bindings, transport events and other hardware.

Fresh native captures at tensor parallelism (TP) 16 and decode context parallelism (DCP) 4 with batch one, and TP32/DCP8 with batch two, match the frozen node rows after only the missing vector bindings are inserted.
Those controls add 64 and 128 vector bindings respectively and preserve their isolated durations.
Two complete 60-layer serving programs each add 1,920 missing bindings before replay.
Key-value (KV) storage and useful model floating-point operation utilization (MFU) retain the conventions in the [main search record](lpx-maximum-mfu-search.md).

| Decode control | LP30 chips | Resident requests | Warm-up completions | Measured outputs | Corrected MFU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing separate KV format | 163 | 10 | 80 | 320 | 3.039141553% |
| Experimental shared KV format | 163 | 14 | 112 | 448 | 3.419600500% |

The corrected controls preserve the displayed decode records at their original issue intervals.
Their two measurement halves agree within 1e-10 relative.
Other configurations can change because the corrected exclusion alters queue ordering.
Older frozen profiles retain their original resource lists and need recapture or an explicit vector-binding transformation before use with this model.
New native profile cache identities include the modified source and cannot reuse the old input identity.

The focused vector, streaming and fidelity checks pass all 31 tests.
The complete suite runs 878 tests, with 876 passing and the same two macOS environment issues recorded before this change.
`test_watch_paths_require_unique_names` differs because `/tmp` resolves to `/private/tmp`.
`test_stop_refuses_foreign_process` requires the unavailable `nvidia-smi` command.
The formatter leaves all 580 Python files compliant, and the staged diff check passes.

The [integrated-boundary study](lpx-integrated-boundary-search.md) retains the placement experiments, corrected controls and graph fingerprints associated with this correction.
The shared-cache format still requires matching model-quality evidence, and the global maximum remains uncertified.
