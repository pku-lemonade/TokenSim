#!/usr/bin/env python3
"""Retired AgentX comparison entrypoint.

Use ``scripts/run_agentx_experiments.py``. The former command hard-coded a
64-GPU proxy model, a short profile and an incompatible cache geometry; keeping
it executable would make it too easy to create results with the wrong scope.
"""

if __name__ == "__main__":
    raise SystemExit(
        "run_agentx_comparison.py is retired; use "
        "python scripts/run_agentx_experiments.py --action run --group checkpoints"
    )
