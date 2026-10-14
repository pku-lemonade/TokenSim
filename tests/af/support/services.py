"""Trace-independent observations for AF differential tests and wall profiling."""

from contextlib import contextmanager
from dataclasses import asdict
from unittest.mock import patch

from TokenSim.af.execution.executor import AFExecutor


@contextmanager
def observe_services():
    """Observe scheduling without adding events, resource requests or timeouts."""
    rows = []
    original = AFExecutor._run_resources_stage

    def observed(self, block, layer_id, stage, resources, estimate_factory):
        result = yield from original(
            self, block, layer_id, stage, resources, estimate_factory
        )
        ready, start, end, estimate = result
        rows.append(
            dict(
                iteration=block.iteration_id,
                block=block.block_id,
                layer=layer_id,
                node=stage,
                ready=ready,
                start=start,
                end=end,
                queue=start - ready,
                estimate=asdict(estimate),
            )
        )
        return result

    with patch.object(AFExecutor, "_run_resources_stage", observed):
        yield rows
