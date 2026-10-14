# Prefix Cache

With paged attention, TokenSim reuses cached attention key/value (KV) blocks when requests share a matching prompt prefix.
The simulator does not tokenize prompts or hash prompt text.
The workload must provide one `hash_ids` entry for every full input block that should be reusable.

## Quick Start

The included trace contains three requests that share part of their prompt prefixes.

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

For `qwen_jsonl`, each line can contain a request with the following fields.

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

For `json_pairs`, put the same metadata in the optional third element of each request record.

```json
[
  [
    48,
    4,
    {
      "hash_ids": ["a", "b", "c"],
      "reuse_group": "tenant-a"
    }
  ],
  [
    48,
    4,
    {
      "hash_ids": ["a", "b", "x"],
      "reuse_group": "tenant-a"
    }
  ]
]
```

With `--block_size 16` and `input_length 48`, the three identifiers represent the three complete input blocks.
Supply at least `floor(input_length / block_size)` identifiers to make every full input block reusable.
Extra identifiers are ignored, and a full block without an identifier must be computed again.
An incomplete final block is computed for that request and cannot be shared with other requests.

## When a Request Gets a Hit

A request uses the local prefix cache only when all of the following conditions hold.

1. The run uses `--batching paged-attn`.
   Static and dynamic batching do not create a block manager and therefore do not perform prefix lookup.
2. The request supplies `hash_ids` for its full input blocks.
3. An earlier request on the same worker has computed matching full blocks and made them available for reuse.
   Uniform caches publish input blocks after prompt processing produces the first output token.
   Grouped caches publish pages as each prompt chunk finishes, and output blocks are published when the request finishes.
   A block may be reused while the earlier request is still running or after it finishes, as long as the block has not been evicted from GPU memory.
   Released blocks keep their cached content until another allocation overwrites it.
   The allocator uses free blocks with no cached content before reusing cached blocks in least-recently-used order.
4. The model, `cache_salt`, and `reuse_group` are identical.
   These values are part of the cache key and keep different models, versions and tenants from sharing blocks.
5. The block identifiers match in order, starting with the first block.
   Lookup stops at the first mismatch, even if a later block identifier matches.

For example, after `["a", "b", "c"]` is cached, `["a", "b", "x"]` hits two blocks, while `["a", "z", "c"]` hits only the first block.
A request without `hash_ids` computes its prompt without prefix reuse and reports neither reuse hits nor misses.
When every prompt block hits, the last prompt token is still computed so that the request can sample.

Grouped KV caches store several types of cache pages, such as compressed attention pages and sliding-window pages.
They use the same ordered block identifiers to identify reusable prefixes.
A prefix can be reused only if the pages needed by every cache group are still in memory, so the reusable prefix can be shorter than the matching identifier sequence.

The local prefix index belongs to one worker.
Data parallelism or separate workers for prompt processing (prefill) and token generation (decode) can send related requests to different workers.
`--sticky_sessions` or `--router_policy consistent_hash` keep a conversation on one model replica, as described in [Parallelism](parallelism.md).
Use `MooncakeStoreConnector` to share prefixes through an external store, as described in [Mooncake](mooncake.md).

For AF with `A TP > 1`, "one worker" means one A DP replica's logical cache.
The scheduler performs prefix lookup once on the leader, while every hit or
miss maps to one local shard on each A TP rank. Reference counts, eviction,
free, preemption, and swap remain aligned across the complete shard set. An F
rank never owns prefix-cache blocks.

## Reusing Generated Output

`output_hash_ids` lets completed decode blocks become part of a later prompt.
Output blocks are registered when the request finishes, provided all of the following conditions hold.

- The input length is a multiple of the block size.
- All full input blocks have `hash_ids`.
- Token generation produces at least one full block.
- `output_hash_ids` supplies identifiers for those full output blocks.

Run the included example with the following command.

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

The first request declares input block `p` and output block `o`.
The second request arrives at the recorded time of 1 second, after the first request has finished.
It declares `["p", "o"]` as its input block identifiers and reuses both blocks.
The run reports 2 hit blocks and 1 missed block.

## Metrics

The result JSON includes the following metrics.

| Field | Meaning |
| --- | --- |
| `reuse_hit_blocks` | Full input blocks reused across all requests (local GPU and external store hits) |
| `reuse_miss_blocks` | Hash-addressable full input blocks not reused |
| `reuse_hit_tokens` | Reused tokens, including external store hits |
| `effective_prefill_tokens` | Tokens that still require prefill compute |
| `prefix_cache_hit_rate` | `hit_blocks / (hit_blocks + miss_blocks)` |

Mooncake-specific local, memory, and SSD hit counters are documented in the [Mooncake guide](mooncake.md).
