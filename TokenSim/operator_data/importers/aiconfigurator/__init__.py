"""Import NVIDIA AIConfigurator perf tables (Apache-2.0) into operator tables.

Two directory layouts are accepted:

* the upstream database,
  ``aic-core/src/aiconfigurator_core/systems/data/<system>/<family>/<backend>/<version>/*_perf.parquet``;
* a flat collector run directory, where ``collector/collect.py`` and
  ``network/collect_comm.sh`` leave ``*_perf.parquet`` (or, with ``--keep-csv``,
  the ``*_perf.txt`` CSV staging files) next to each other.

Latencies there are in **milliseconds**; ours are microseconds.

Mapping:

=========================  ==========================  ===============================
AIConfigurator file        our table                   notes
=========================  ==========================  ===============================
gemm_perf                  gemm                        gemm_dtype -> dtype (alias map)
context_attention_perf     context_attention           isl -> input_seq_len
generation_attention_perf  generation_attention        context_len = isl + step
moe_perf                   moe                         topk -> top_k, moe_tp/ep_size
nccl_perf / oneccl_perf    collective                  alltoall -> all_to_all, nodes=1
custom_allreduce_perf      collective (kernel=custom)  all_reduce only
moe_a2a_perf               ep_all2all                  DeepEP dispatch/combine (latency already in us)
wideep_deepep_{ll,normal}  ep_all2all                  SGLang DeepEP (two rows per source row)
wideep_*moe_perf           moe (backfill only)         wide-EP expert kernels
=========================  ==========================  ===============================
"""

from __future__ import annotations


from .system import import_aiconfigurator_system
from .discovery import (
    _map_dtype,
    _read_parquet,
    _has_table,
    _version_key,
    _latest_version,
    _select,
    _candidates,
    _version_label,
    _system_gpus_per_node,
)
