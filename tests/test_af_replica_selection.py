from __future__ import annotations

import unittest

from TokenSim.af.planning import IndependentRoundRobinAFReplicaSelector
from TokenSim.config.cluster_config import (
    DomainParallelConfig,
    DomainRankMapping,
    ExecutionDomainConfig,
    LogicalInstanceConfig,
    PhysicalWorkerRef,
)


def _domain(
    name: str,
    capability: str,
    *,
    dp_size: int,
    expert_parallel_size: int = 1,
) -> ExecutionDomainConfig:
    instances = []
    for dp_rank in range(dp_size):
        instances.append(
            LogicalInstanceConfig(
                name=f"{name}-dp{dp_rank}",
                rank_mapping=[
                    DomainRankMapping(
                        domain_rank=dp_rank,
                        tp_rank=0,
                        pp_rank=0,
                        dp_rank=dp_rank,
                        ep_rank=(
                            dp_rank if expert_parallel_size > 1 else 0
                        ),
                        physical_worker=PhysicalWorkerRef(
                            group_name=f"{name}-pool",
                            worker_index=dp_rank,
                        ),
                    )
                ],
            )
        )
    return ExecutionDomainConfig(
        name=name,
        capability=capability,
        phases=("prefill", "decode"),
        worker_group=f"{name}-pool",
        parallel=DomainParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=dp_size,
            expert_parallel_size=expert_parallel_size,
        ),
        logical_instances=instances,
    )


class IndependentAFReplicaSelectorTest(unittest.TestCase):
    def test_consecutive_blocks_rotate_a_and_f_independently(self):
        a_domain = _domain("a", "af_attention", dp_size=2)
        f_domain = _domain("f", "af_ffn", dp_size=3)
        selector = IndependentRoundRobinAFReplicaSelector()

        selections = [
            selector.select_for_block("decode", a_domain, f_domain)
            for _ in range(6)
        ]

        self.assertEqual(
            [item.a_owner.name for item in selections],
            ["a-dp0", "a-dp1", "a-dp0", "a-dp1", "a-dp0", "a-dp1"],
        )
        self.assertEqual(
            [item.f_service.name for item in selections],
            ["f-dp0", "f-dp1", "f-dp2", "f-dp0", "f-dp1", "f-dp2"],
        )

    def test_selecting_a_does_not_advance_f_stream(self):
        a_domain = _domain("a", "af_attention", dp_size=2)
        f_domain = _domain("f", "af_ffn", dp_size=2)
        selector = IndependentRoundRobinAFReplicaSelector()

        self.assertEqual(selector.select_a_owner("decode", a_domain).name, "a-dp0")
        self.assertEqual(selector.select_a_owner("decode", a_domain).name, "a-dp1")
        self.assertEqual(selector.select_f_service("decode", f_domain).name, "f-dp0")
        self.assertEqual(selector.select_f_service("decode", f_domain).name, "f-dp1")

    def test_ep_one_routes_only_inside_selected_f_service_replica(self):
        a_domain = _domain("a", "af_attention", dp_size=1)
        f_domain = _domain("f", "af_ffn", dp_size=2, expert_parallel_size=1)
        selector = IndependentRoundRobinAFReplicaSelector()

        first = selector.select_for_block("decode", a_domain, f_domain)
        second = selector.select_for_block("decode", a_domain, f_domain)

        self.assertEqual(first.expert_parallel_mode, "expert_tp")
        self.assertEqual(first.routed_expert_instances, (first.f_service,))
        self.assertEqual(second.routed_expert_instances, (second.f_service,))
        self.assertIs(first.dense_ffn_instance, first.f_service)
        self.assertIs(first.shared_expert_instance, first.f_service)

    def test_rank_reuse_routed_scope_spans_world_not_service_replica(self):
        a_domain = _domain("a", "af_attention", dp_size=2)
        f_domain = _domain("f", "af_ffn", dp_size=3, expert_parallel_size=3)
        selector = IndependentRoundRobinAFReplicaSelector()

        first = selector.select_for_block("decode", a_domain, f_domain)
        second = selector.select_for_block("decode", a_domain, f_domain)

        expected_world = ("f-dp0", "f-dp1", "f-dp2")
        self.assertEqual(first.expert_parallel_mode, "rank_reuse")
        self.assertEqual(first.f_service.name, "f-dp0")
        self.assertEqual(second.f_service.name, "f-dp1")
        self.assertEqual(
            tuple(item.name for item in first.routed_expert_instances),
            expected_world,
        )
        self.assertEqual(
            tuple(item.name for item in second.routed_expert_instances),
            expected_world,
        )


if __name__ == "__main__":
    unittest.main()
