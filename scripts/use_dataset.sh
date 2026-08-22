#!/usr/bin/env bash

./benchmark.py \
    --batching paged-attn \
    --block_size 16 \
    --request_count 20 \
    --cluster ./data/clusters/1_a100/h1.json \
    --dataset_path ./dataset/example.json \
    --workload_type json_pairs \
    --qps 50 \
    --max_parallem_sum 100 \
    --verbose none \
    --model ./data/psla/llama-7b.json
