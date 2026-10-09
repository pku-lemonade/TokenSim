#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${1:-${repository_root}/dataset/agentx}"
revision="23f152f6f0f9399a85901b89a6458def0ef16729"
checksum="29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276"
mkdir -p "${destination}"

if [[ -f "${destination}/traces.jsonl" ]] \
    && printf '%s  %s\n' "${checksum}" "${destination}/traces.jsonl" | sha256sum --check --status; then
    echo "AgentX traces already match the pinned dataset."
    exit 0
fi

if ! command -v hf >/dev/null 2>&1; then
    echo "Install the download extra first: pip install -e '.[agentx]'" >&2
    exit 1
fi
hf download semianalysisai/cc-traces-weka-062126 traces.jsonl \
    --repo-type dataset --revision "${revision}" --local-dir "${destination}"
printf '%s  %s\n' "${checksum}" "${destination}/traces.jsonl" | sha256sum --check
