#!/usr/bin/env bash

python benchmark.py \
    --batching paged-attn \
    --block_size 16 \
    --request_count 20 \
    --cluster ./data/clusters/1_a100/h1.json \
    --dataset_path ./dataset/example.json \
    --workload_type json_pairs \
    --qps 50 \
    --max_num_seqs 100 \
    --verbose none \
    --model ./data/models/llama-7b.yaml
