# DeepSeek V3 serving cluster

`1a32f.json` places one attention worker and 32 FFN workers on catalog A100 SXM 80 GB devices.
Physical communication uses five eight-slot nodes with dedicated A/F placement.
The F domain uses TP=32 and EP=32 over the same 32 ranks.
Dense/shared FFNs use TP, and routed experts use one rank per expert shard.
The model architecture and provenance are declared in `data/models/deepseek-v3-671b-mla.yaml`.

Host submission costs 5 microseconds per message and completion costs zero by configuration.
These values are simulation assumptions.
Physical bandwidth and contention come from `af.communication_domains`.
The `nvlink3` catalog name does not imply that 33 GPUs share one physical node.

```bash
python benchmark.py \
  --cluster data/clusters/deepseek_v3/1a32f.json \
  --model deepseek-v3-671b-mla \
  --latency_backend operator_table \
  --latency_fallback analytical_only \
  --qps 1 \
  --request_count 1 \
  --prefill_mean_len 16 \
  --prefill_range_len 0 \
  --decode_mean_len 2 \
  --decode_range_len 0 \
  --batching paged-attn \
  --distribution burst \
  --block_size 16 \
  --verbose none \
  --results_path outputs/deepseek-v3-serving
```

MLA serving requires whole, equal-length prefill without cached prefixes or chunking.
Use the [MLA guide](../../../docs/deepseek-v3-mla.md) for the fixed-concurrency decode runner and operator-data requirements.
