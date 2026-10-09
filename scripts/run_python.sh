#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${TOKENSIM_PYTHON:-}" ]]; then
    exec "${TOKENSIM_PYTHON}" "$@"
fi

for candidate in python3.12 python3; do
    if command -v "${candidate}" >/dev/null 2>&1 \
        && "${candidate}" -c 'import sys; raise SystemExit(sys.version_info < (3, 11)); import simpy, pydantic, numpy' 2>/dev/null; then
        exec "${candidate}" "$@"
    fi
done

echo "Install the TokenSim dependencies with Python 3.12, or set TOKENSIM_PYTHON to a compatible interpreter." >&2
exit 1
