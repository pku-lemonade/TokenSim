#!/usr/bin/env bash
set -euo pipefail

destination="${1:-./dataset/agentx}"
mkdir -p "${destination}"

export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

if command -v hf >/dev/null 2>&1; then
    hf download semianalysisai/cc-traces-weka-062126 \
        traces.jsonl \
        --repo-type dataset \
        --local-dir "${destination}"
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download semianalysisai/cc-traces-weka-062126 \
        traces.jsonl \
        --repo-type dataset \
        --local-dir "${destination}"
else
    echo "Install huggingface_hub first: pip install -U huggingface_hub" >&2
    exit 1
fi
