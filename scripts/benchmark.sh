#!/usr/bin/env bash

./benchmark.py \
    --batching paged-attn \
    --block_size 16 \
    --swap_policy eager \
    --prompt_count 100 \
    --prompt_lens_mean 128 \
    --generation_lens_mean 128 \
    --cluster ./data/clusters/1_a100/h1.json \
    --qps 50 \
    --psla ./data/psla/llama-7b.json
