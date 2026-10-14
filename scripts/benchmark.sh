#!/usr/bin/env bash

python benchmark.py \
    --batching paged-attn \
    --block_size 16 \
    --request_count 100 \
    --prefill_mean_len 128 \
    --decode_mean_len 128 \
    --cluster ./data/clusters/1_a100/h1.json \
    --qps 50 \
    --model ./data/models/llama-7b.yaml
