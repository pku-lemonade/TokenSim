# Prefix Cache

TokenSim simulates block-granular prefix KV-cache reuse when paged attention is
enabled and the workload supplies stable content identifiers. The simulator does
not tokenize prompts or hash prompt text. The workload must provide one
`hash_ids` entry for every full input block that should be reusable.

## Quick Start

The included trace has three requests with partially shared prefixes:

```bash
./benchmark.py \
  --batching paged-attn \
  --block_size 16 \
  --request_count 3 \
  --workload_type qwen_jsonl \
  --dataset_path ./dataset/prefix_reuse_example.jsonl \
  --cluster ./data/clusters/1_a100/h1.json \
  --model ./data/models/llama-7b.yaml \
  --qps 10 \
  --verbose none
```

The run reports 3 hit blocks and 6 missed blocks (`hit_rate=0.333...`).

For `qwen_jsonl`, each line can contain:

```json
{
  "request_id": 1,
  "input_length": 48,
  "output_length": 4,
  "hash_ids": ["a", "b", "x"],
  "cache_salt": "optional-version",
  "reuse_group": "tenant-a"
}
```

For `json_pairs`, put the same metadata in the optional third element:

```json
[
  [48, 4, {"hash_ids": ["a", "b", "c"], "reuse_group": "tenant-a"}],
  [48, 4, {"hash_ids": ["a", "b", "x"], "reuse_group": "tenant-a"}]
]
```

With `--block_size 16` and `input_length 48`, the three identifiers represent
the three complete input blocks. Identifiers beyond the full input blocks are
ignored, and full blocks without an identifier cannot be reused, so supply at
least `floor(input_length / block_size)` identifiers. An incomplete final block
is always computed and remains request-private.

## When a Request Gets a Hit

A request uses the local prefix cache only when all of these conditions hold:

1. The run uses `--batching paged-attn`. Static and dynamic batching do not
   create a block manager and therefore do not perform prefix lookup.
2. The request supplies `hash_ids` for its full input blocks.
3. An earlier request on the same worker has committed matching full blocks.
   Input blocks are committed once that request has computed its prompt
   (sampled its first token; grouped caches publish pages as chunks
   complete), output blocks when it finishes. A block may be reused while the
   earlier request is still running or after it finishes, as long as the block
   has not been evicted from GPU memory. Freed blocks keep their cached
   content until the memory is reallocated; uncached free blocks are
   allocated first, then cached ones in least-recently-used order.
4. The model, `cache_salt`, and `reuse_group` are identical. These values scope
   the cache key and prevent reuse across models, versions, or tenants.
5. The block identifiers form the same chain from the first block onward.
   Lookup stops at the first mismatch; a later matching identifier does not
   restart the hit sequence.

For example, after `["a", "b", "c"]` is cached, `["a", "b", "x"]` hits two
blocks, while `["a", "z", "c"]` hits only the first block. A request without
`hash_ids` follows the baseline path and reports neither reuse hits nor misses.
When every prompt block hits, the last prompt token is still computed so that
the request can sample.

Models with grouped KV caches (several cache-page geometries, such as
compressed attention with sliding-window layers) key their pages with the
same chained identifiers. A hit must be resident in every cache group, so the
usable prefix can be shorter than the longest matching chain.

The local prefix index belongs to one worker. Data-parallel placement or a
disaggregated prefill/decode layout can send related requests to different
workers. `--sticky_sessions` or `--router_policy consistent_hash` keep a
conversation on one replica (see [Parallelism](parallelism.md)); use a
`MooncakeStoreConnector` when reuse must be modeled through an external store
(see [Mooncake](mooncake.md)).

## Reusing Generated Output

`output_hash_ids` lets completed decode blocks become part of a later prompt.
Output blocks are registered when the request finishes, and only when:

- the input length is block-aligned;
- all full input blocks have `hash_ids`;
- the decode produces at least one full block; and
- `output_hash_ids` supplies identifiers for those full output blocks.

Run the included example with:

```bash
./benchmark.py \
  --batching paged-attn \
  --block_size 16 \
  --request_count 2 \
  --workload_type qwen_jsonl \
  --dataset_path ./dataset/prefix_reuse_output_example.jsonl \
  --cluster ./data/clusters/1_a100/h1.json \
  --model ./data/models/llama-7b.yaml \
  --qps 10 \
  --verbose none
```

The first request declares input block `p` and output block `o`. The second
request arrives at its trace timestamp (1 s), after the first request has
finished, declares `["p", "o"]` as its input chain and reuses both blocks
(2 hit blocks and 1 missed block for the run).

## Metrics

The result JSON exports:

| Field | Meaning |
| --- | --- |
| `reuse_hit_blocks` | Full input blocks reused across all requests (local GPU and external store hits) |
| `reuse_miss_blocks` | Hash-addressable full input blocks not reused |
| `reuse_hit_tokens` | Reused tokens, including external store hits |
| `effective_prefill_tokens` | Tokens that still require prefill compute |
| `prefix_cache_hit_rate` | `hit_blocks / (hit_blocks + miss_blocks)` |

Mooncake-specific local, memory, and SSD hit counters are documented in the
[Mooncake guide](mooncake.md).
