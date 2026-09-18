#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${TOKENSIM_PYTHON:-}" ]]; then
    exec "${TOKENSIM_PYTHON}" "$@"
fi

if python3 -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))' 2>/dev/null \
    && python3 -c 'import simpy; from TokenSim.hardware import HardwareContext' 2>/dev/null; then
    exec python3 "$@"
fi

if command -v conda >/dev/null 2>&1 \
    && conda run -n tokensim11 python -c 'import simpy; from TokenSim.hardware import HardwareContext' >/dev/null 2>&1; then
    exec conda run --no-capture-output -n tokensim11 python "$@"
fi

if command -v python3.11 >/dev/null 2>&1 \
    && python3.11 -c 'import simpy; from TokenSim.hardware import HardwareContext' 2>/dev/null; then
    exec python3.11 "$@"
fi

echo "TokenSim requires Python 3.11 with requirements.txt installed." >&2
echo "Activate that environment or set TOKENSIM_PYTHON=/path/to/python." >&2
exit 1
