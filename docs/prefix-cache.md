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
  --model ./data/psla/llama-7b.json \
  --qps 10 \
  --verbose none
```

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
the three complete input blocks. Extra identifiers are ignored. Supplying at
least `floor(input_length / block_size)` identifiers is recommended; an
incomplete final block is always computed and remains request-private.

## When a Request Gets a Hit

A request uses the local prefix cache only when all of these conditions hold:

1. The run uses `--batching paged-attn`. Static and dynamic schedulers do not
   create a block manager and therefore do not perform local prefix lookup.
2. The request supplies `hash_ids` for its full input blocks.
3. An earlier request on the same worker has committed matching full blocks.
   A block may be reused while the earlier request is still running or after it
   finishes, as long as the block has not been evicted or moved off GPU.
4. The model, `cache_salt`, and `reuse_group` are identical. These values scope
   the cache key and prevent reuse across models, versions, or tenants.
5. The block identifiers form the same chain from the first block onward.
   Lookup stops at the first mismatch; a later matching identifier does not
   restart the hit sequence.

For example, after `["a", "b", "c"]` is cached, `["a", "b", "x"]` hits two
blocks, while `["a", "z", "c"]` hits only the first block. A request without
`hash_ids` follows the baseline path and reports neither reuse hits nor misses.

The local prefix index belongs to one worker. Data-parallel placement or a
disaggregated prefill/decode layout can send related requests to different
workers; use a `MooncakeStoreConnector` when reuse must be modeled through an
external store. See [Mooncake](mooncake.md).

## Reusing Generated Output

`output_hash_ids` lets completed decode blocks become part of a later prompt.
Output blocks are registered only when:

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
  --model ./data/psla/llama-7b.json \
  --qps 10 \
  --verbose none
```

The first request declares input block `p` and output block `o`. The second
request declares `["p", "o"]` as its input chain and can reuse both blocks if
they are still resident.

## Metrics

The result JSON exports:

| Field | Meaning |
| --- | --- |
| `reuse_hit_blocks` | Full input blocks reused across all requests |
| `reuse_miss_blocks` | Hash-addressable full input blocks not reused |
| `reuse_hit_tokens` | Reused tokens, including external store hits |
| `effective_prefill_tokens` | Tokens that still require prefill compute |
| `prefix_cache_hit_rate` | `hit_blocks / (hit_blocks + miss_blocks)` |

Mooncake-specific local, memory, and SSD hit counters are documented in the
[Mooncake guide](mooncake.md).
