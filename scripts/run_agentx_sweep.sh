#!/usr/bin/env bash
set -euo pipefail

echo "run_agentx_sweep.sh is retired; use the versioned native matrix instead:" >&2
echo "  python scripts/run_agentx_experiments.py --action run --group curve" >&2
echo "The matrix owns the full model, 393-trace pool, 3,600-second window, and topology." >&2
exit 2
