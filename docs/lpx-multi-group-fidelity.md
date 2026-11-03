# Physical groups and routed A/F traffic on LP30

Pure-LP30 attention/feed-forward (A/F) separation requires distinct physical resources for its A and F worker groups.
The streaming refinement previously used each group's local worker index as its fabric chip number.
For example, `A:0` and `F:0` shared `lp30:0:vxm` even though the runtime assigned them to different physical workers.
A two-layer research configuration contained forty LP30 workers but only thirty-two distinct fabric indices.
This alias could create false vector-unit contention and assign unrelated chip-to-chip (C2C) operations to the same ports.
The old route inventory also included local indices from other accelerator types.

## Physical coordinate assignment

The refinement now assigns LP30 groups disjoint coordinate ranges in their declared worker order.
Each group's range spans its largest local index plus one, preserving sparse local indices.
The initial LP30 group's coordinates remain its native local indices.
Only LP30 workers enter the LP30 route inventory.
A group containing indices zero through thirty-one followed by a group containing zero through seven therefore maps to coordinates zero through thirty-nine.
The coordinate packing is a declared simulation placement rule and requires physical placement validation.
It does not supply disclosed LPX wiring or SRAM bank mappings.

| Physical identity | Earlier coordinate | Corrected coordinate |
| --- | ---: | ---: |
| `A:0` | 0 | 0 |
| `A:31` | 31 | 31 |
| `F:0` | 0 | 32 |
| `F:7` | 7 | 39 |

A regression schedules independent scalar operations on distinct group chips at the same time.
The corrected operations reserve distinct vector execution module (VXM) resources and complete concurrently.
Another regression rejects a route that attempts to use a non-LP30 worker's local index as an LP30 transit chip.
The existing single-group coordinate convention is preserved.

## A/F transfers and injection accounting

Pure-LP30 `af_a_to_f` and `af_f_to_a` transfers now use the same routed endpoint and port resources as pipeline handoffs and refined collectives.
The earlier native endpoint names did not share the refinement's C2C injection locks.
The source and destination identities determine transfer direction, lane assignment and propagation scope.
Injection resources are released after wire transmission, while consumers wait for the propagation delay.
This conservative model serializes simultaneous message phases that share an exclusive endpoint resource.
Different schedules or vector-streamed A/F transfers require separate programs and resource bounds.

Copies within one physical chip and transfers involving a non-LP30 endpoint retain their native transport model.
Measured A/F costs are rejected by this analytical rerouting path.
A completion-only record from a multiphase transfer is also rejected because it does not identify an independent wire phase.
These checks avoid repricing a measured transport or charging a completion record as another data transfer.

The [NVIDIA LPX architecture description](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/) places decode attention on Rubin GPUs and latency-sensitive FFN or expert work on LPUs in its heterogeneous A/F example.
That example does not calibrate a pure-LP30 role split.
The [SHIP paper](https://proceedings.mlsys.org/paper_files/paper/2026/hash/9c20f16b05f5e5e70fa07e2a4364b80e-Abstract-Conference.html) studies a prior Groq generation and supplies architectural motivation for pipeline and memory investigations.
Its results are separate from LP30 timing calibration.

## Validation scope

Twelve new tests cover disjoint group coordinates, sparse indices, cross-group compute overlap, route inventory, both A/F directions, shared C2C contention, local copies, mixed hardware and unsupported transfer records.
Five coordinate tests fail on the earlier mapping.
The A/F transport tests produce four failures and one error before that transport refinement is enabled.
All forty-three focused group, transport, vector, streaming and fidelity tests pass with the corrections.
Pinned Black formatting and checking cover 582 Python files.
The full suite passes 888 of 890 tests, with the same two macOS environment failures involving `/tmp` resolution and the unavailable `nvidia-smi` binary.
Fresh single-group TP16/DCP4/batch-one and TP32/DCP8/batch-two captures match the frozen node rows after applying the previously documented scalar VXM correction.
Their isolated durations remain 15.218782676 and 18.818555436 microseconds.

The production configuration guard for context parallelism in non-colocated A/F domains remains enabled.
Exploratory A/F component graphs explicitly identify any research-only override of that guard.
Such graphs provide conditional memory and timing evidence and do not demonstrate production compiler support.
Static bank allocation, physical routes, unit calibration and model-quality equivalence remain outside these code checks.
