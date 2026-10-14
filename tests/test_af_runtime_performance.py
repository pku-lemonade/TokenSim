import unittest
from unittest.mock import patch

import simpy

from TokenSim.af.planning import RankLocalWork
from TokenSim.config.cluster_config import AFCommunicationTimingConfig
from tests.af_runtime_oracle import observe_services
from tests.test_af_communication_timing import WireOnly
from tests.test_af_executor import (
    FixedComputeProvider,
    _executor,
    _load_cluster,
    _requests,
)


class TraceConstructionTest(unittest.TestCase):
    def test_trace_off_skips_construction_and_preserves_all_services(self):
        for timing_enabled in (False, True):
            observations = []
            for trace_enabled in (True, False):
                with self.subTest(timing=timing_enabled, trace=trace_enabled):
                    env = simpy.Environment()
                    cluster = _load_cluster(trace_enabled=trace_enabled)
                    cluster.af.communication_timing = AFCommunicationTimingConfig(
                        enabled=timing_enabled, host_submit_us=5, host_completion_us=2
                    )
                    compute = FixedComputeProvider(0.002, 0.003)
                    executor = _executor(env, cluster, compute, WireOnly(), layers=2)
                    iteration = executor.submit(0, _requests(env, 3))
                    with observe_services() as services:
                        if trace_enabled:
                            env.process(executor.run_iteration(iteration))
                            env.run()
                        else:
                            with (
                                patch(
                                    "TokenSim.af.executor_compute.AFTraceEvent",
                                    side_effect=AssertionError("trace constructed"),
                                ),
                                patch.object(
                                    RankLocalWork,
                                    "as_dict",
                                    side_effect=AssertionError("trace serialized"),
                                ),
                            ):
                                env.process(executor.run_iteration(iteration))
                                env.run()
                    # Host details describe observation only; provider estimates,
                    # including payload/provenance, must otherwise be identical.
                    for row in services:
                        if (
                            row["estimate"]["source_id"]
                            == "assumption:communication-host-cost"
                        ):
                            row["estimate"]["details"] = {}
                    observations.append(
                        (
                            iteration.makespan,
                            iteration.block_completed_at,
                            services,
                            compute.queries,
                        )
                    )
                    self.assertEqual(bool(executor.trace.events), trace_enabled)
            self.assertEqual(observations[0], observations[1])
