"""Af config topology."""

from __future__ import annotations

from copy import deepcopy
import unittest

from TokenSim.config.af.communication import (
    AFCommunicationFabricConfig,
    AFCommunicationLaneConfig,
)
from TokenSim.config.af.domains import DomainCollectiveConfig
from TokenSim.config.af.links import AFLinkConfig
from TokenSim.config.af.settings import AFConfig
from TokenSim.config.cluster_config import ClusterConfig, WorkerGroupConfig
from TokenSim.errors import ConfigurationError
from tests.af.support.config_validation import (
    _af_group,
    _cluster,
    _explicit_decode_cluster_raw,
    _full_link,
)


class AFLinkTopologyTest(unittest.TestCase):
    def test_full_and_half_duplex_capacity_validation(self):
        full = _full_link()
        half = _full_link(
            duplex="half",
            a_to_f_capacity=None,
            f_to_a_capacity=None,
            half_duplex_capacity=2,
        )
        self.assertEqual(full.effective_capacities, (1, 1))
        self.assertEqual(half.effective_capacities, (2,))

        invalid = [
            {"a_to_f_capacity": None},
            {"duplex": "simplex"},
            {"a_to_f_capacity": 0},
            {"f_to_a_capacity": True},
            {"half_duplex_capacity": 1},
            {
                "duplex": "half",
                "a_to_f_capacity": None,
                "f_to_a_capacity": None,
                "half_duplex_capacity": None,
            },
        ]
        for overrides in invalid:
            with (
                self.subTest(overrides=overrides),
                self.assertRaises(ConfigurationError),
            ):
                _full_link(**overrides)

    def test_one_or_two_links_build_phase_map(self):
        cluster = _cluster()
        self.assertIs(cluster.phase_to_af_link["prefill"], cluster.af.links[0])
        self.assertIs(cluster.phase_to_af_link["decode"], cluster.af.links[0])

        links = [
            _full_link(name="prefill-link", phases=("prefill",), resource_group="p"),
            _full_link(name="decode-link", phases=("decode",), resource_group="d"),
        ]
        split = _cluster(links=links)
        self.assertEqual(split.phase_to_af_link["prefill"].name, "prefill-link")
        self.assertEqual(split.phase_to_af_link["decode"].name, "decode-link")

    def test_missing_or_ambiguous_phase_link_is_rejected(self):
        with self.assertRaisesRegex(ConfigurationError, "decode.*exactly one"):
            _cluster(links=[_full_link(phases=("prefill",))])
        with self.assertRaisesRegex(ConfigurationError, "prefill.*exactly one"):
            _cluster(
                links=[
                    _full_link(name="one"),
                    _full_link(name="two", phases=("prefill",), resource_group="two"),
                ]
            )

    def test_endpoint_validation_includes_context(self):
        with self.assertRaisesRegex(ConfigurationError, "af-link.*missing"):
            _cluster(links=[_full_link(attention_group="missing")])
        wrong_ffn = _af_group("wrong-ffn", "af_attention")
        with self.assertRaisesRegex(ConfigurationError, "af-link.*ffn.*af_ffn"):
            _cluster(
                links=[_full_link(ffn_group="wrong-ffn")],
                extra_groups=[wrong_ffn],
            )
        with self.assertRaisesRegex(ConfigurationError, "must differ"):
            _cluster(links=[_full_link(ffn_group="attention")])

    def test_hardware_names_do_not_infer_capabilities(self):
        groups = [
            WorkerGroupConfig(
                name="attention",
                role="prefill",
                hardware="Ada",
                num_workers=1,
            ),
            WorkerGroupConfig(
                name="ffn",
                role="decode",
                hardware="Groq",
                num_workers=1,
            ),
        ]
        with self.assertRaisesRegex(ConfigurationError, "af_attention"):
            ClusterConfig(
                num_workers=2,
                networks={"net1": "network"},
                worker_groups=groups,
                af=AFConfig(enabled=True, links=[_full_link()]),
            )

    def test_partial_af_requires_complete_other_phase(self):
        decode_link = _full_link(phases=("decode",))
        with self.assertRaisesRegex(ConfigurationError, "Decode-only.*prefill"):
            _cluster(phases=("decode",), links=[decode_link])
        decode_cluster = _cluster(
            phases=("decode",),
            links=[decode_link],
            extra_groups=[
                WorkerGroupConfig(role="prefill", hardware="device", num_workers=1)
            ],
        )
        self.assertIn("prefill", decode_cluster.worker_groups[-1].capabilities)

        prefill_link = _full_link(phases=("prefill",))
        with self.assertRaisesRegex(ConfigurationError, "Prefill-only.*decode_full"):
            _cluster(phases=("prefill",), links=[prefill_link])

    def test_unknown_link_network_is_rejected(self):
        with self.assertRaisesRegex(ConfigurationError, "af-link.*unknown network"):
            _cluster(links=[_full_link(network="missing")])


class AFCommunicationResourceTest(unittest.TestCase):
    def test_full_duplex_has_independent_directional_resources(self):
        resources = _cluster().af_communication_resources

        self.assertEqual(set(resources), {("fabric", "a_to_f"), ("fabric", "f_to_a")})
        self.assertEqual(resources[("fabric", "a_to_f")].capacity, 1)
        self.assertEqual(
            resources[("fabric", "a_to_f")].queue_duration_field, "queue_duration"
        )
        self.assertEqual(
            resources[("fabric", "a_to_f")].service_duration_field, "service_duration"
        )

    def test_half_duplex_has_one_shared_resource(self):
        link = _full_link(
            phases=("decode",),
            duplex="half",
            a_to_f_capacity=None,
            f_to_a_capacity=None,
            half_duplex_capacity=2,
        )
        cluster = _cluster(
            phases=("decode",),
            links=[link],
            extra_groups=[
                WorkerGroupConfig(role="prefill", hardware="device", num_workers=1)
            ],
        )
        resources = cluster.af_communication_resources

        self.assertEqual(set(resources), {("fabric", "shared")})
        self.assertEqual(resources[("fabric", "shared")].capacity, 2)

    def test_shared_resource_group_must_have_identical_definition(self):
        links = [
            _full_link(name="prefill-link", phases=("prefill",)),
            _full_link(
                name="decode-link",
                phases=("decode",),
                a_to_f_capacity=2,
            ),
        ]
        with self.assertRaisesRegex(
            ConfigurationError, "fabric.*prefill-link.*decode-link"
        ):
            _cluster(links=links)

        links[1] = _full_link(name="decode-link", phases=("decode",))
        cluster = _cluster(links=links)
        self.assertEqual(len(cluster.af_communication_resources), 2)

    def test_shared_fabric_rejects_network_and_duplex_conflicts(self):
        for overrides in (
            dict(network="net2"),
            dict(
                duplex="half",
                a_to_f_capacity=None,
                f_to_a_capacity=None,
                half_duplex_capacity=1,
            ),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ConfigurationError, "conflicts"),
            ):
                _cluster(
                    networks={"net1": "test-network", "net2": "test-network"},
                    links=[
                        _full_link(name="prefill-link", phases=("prefill",)),
                        _full_link(name="decode-link", phases=("decode",), **overrides),
                    ],
                )

    def test_explicit_fabric_registry_supports_independent_collective_lane(self):
        fabric = AFCommunicationFabricConfig(
            resource_group="fabric",
            network="net1",
            duplex="full",
            lanes=(
                AFCommunicationLaneConfig("a_to_f", 1),
                AFCommunicationLaneConfig("f_to_a", 1),
                AFCommunicationLaneConfig("collective", 2),
            ),
        )
        link = AFLinkConfig(
            name="af-link",
            phases=("prefill", "decode"),
            attention_group="attention",
            ffn_group="ffn",
            resource_group="fabric",
        )

        cluster = _cluster(
            links=[link],
            communication_fabrics=[fabric],
        )

        self.assertEqual(link.network, "net1")
        self.assertEqual(link.duplex, "full")
        self.assertEqual(
            set(cluster.af_communication_resources),
            {
                ("fabric", "a_to_f"),
                ("fabric", "f_to_a"),
                ("fabric", "collective"),
            },
        )
        self.assertEqual(
            cluster.af_communication_resources[("fabric", "collective")].capacity,
            2,
        )

    def test_fabric_registry_rejects_missing_conflicting_and_invalid_lanes(self):
        reference_only = AFLinkConfig(
            name="af-link",
            phases=("prefill", "decode"),
            attention_group="attention",
            ffn_group="ffn",
            resource_group="missing",
        )
        with self.assertRaisesRegex(ConfigurationError, "missing communication fabric"):
            _cluster(links=[reference_only])

        with self.assertRaisesRegex(ConfigurationError, "duplicate lane"):
            AFCommunicationFabricConfig(
                resource_group="fabric",
                network="net1",
                duplex="full",
                lanes=(
                    AFCommunicationLaneConfig("same", 1),
                    AFCommunicationLaneConfig("same", 1),
                ),
            )

        fabric = AFCommunicationFabricConfig(
            resource_group="fabric",
            network="net1",
            duplex="full",
            lanes=(
                AFCommunicationLaneConfig("a_to_f", 1),
                AFCommunicationLaneConfig("f_to_a", 1),
            ),
        )
        with self.assertRaisesRegex(ConfigurationError, "conflicts.*af-link"):
            _cluster(
                links=[_full_link(a_to_f_capacity=2)],
                communication_fabrics=[fabric],
            )


class AFDomainParallelConfigTest(unittest.TestCase):
    def test_f_world_size_does_not_multiply_by_ep(self):
        for ep_size, expected_mode, expected_expert_tp, expected_ep_ranks in (
            (1, "expert_tp", 2, [0, 0, 0, 0]),
            (4, "rank_reuse", 1, [0, 1, 2, 3]),
        ):
            with self.subTest(ep_size=ep_size):
                cluster = ClusterConfig(**_explicit_decode_cluster_raw(f_ep=ep_size))
                domain = cluster.execution_domains["f-domain"]
                mappings = [
                    mapping
                    for instance in domain.logical_instances
                    for mapping in instance.rank_mapping
                ]

                self.assertEqual(domain.parallel.world_size, 4)
                self.assertEqual(domain.required_rank_count, 4)
                self.assertEqual(domain.expert_parallel.mode, expected_mode)
                self.assertEqual(
                    domain.expert_parallel.expert_tensor_parallel_size,
                    expected_expert_tp,
                )
                self.assertEqual(
                    [mapping.ep_rank for mapping in mappings], expected_ep_ranks
                )
                self.assertEqual(
                    next(
                        group.num_workers
                        for group in cluster.worker_groups
                        if group.name == "ffn"
                    ),
                    4,
                )

    def test_intermediate_ep_size_is_rejected(self):
        with self.assertRaisesRegex(
            ConfigurationError,
            "f-domain.*expert_parallel_size=1.*TP\\*DP=4.*got 2",
        ):
            ClusterConfig(**_explicit_decode_cluster_raw(f_ep=2))


class AFCollectiveConfigTest(unittest.TestCase):
    @staticmethod
    def _with_collective_fabric(raw: dict) -> dict:
        raw["af"]["communication_fabrics"].append(
            {
                "resource_group": "a-collective-fabric",
                "network": "net1",
                "duplex": "half",
                "lanes": [{"name": "collective", "capacity": 2}],
            }
        )
        return raw

    @staticmethod
    def _collective(**overrides) -> dict:
        values = {
            "operation": "all_gather",
            "after_stage": "a_compute",
            "participants": [0, 2],
            "resource_group": "a-collective-fabric",
            "resource_lane": "collective",
            "network": "net1",
        }
        values.update(overrides)
        return values

    def test_replica_local_participants_map_to_each_dp_instance(self):
        raw = self._with_collective_fabric(
            _explicit_decode_cluster_raw(a_tp=4, a_dp=2, f_tp=1, f_dp=1)
        )
        raw["af"]["execution_domains"][0]["communication_plan"] = [self._collective()]

        cluster = ClusterConfig(**raw)
        domain = cluster.execution_domains["a-domain"]
        collective = domain.communication_plan[0]

        self.assertEqual(domain.resolve_collective_participants(collective, 0), (0, 2))
        self.assertEqual(domain.resolve_collective_participants(collective, 1), (4, 6))
        self.assertEqual(collective.operation, "all_gather")
        self.assertEqual(
            cluster.af_communication_resources[
                ("a-collective-fabric", "collective")
            ].capacity,
            2,
        )

    def test_collective_participants_and_stage_are_strict(self):
        with self.assertRaisesRegex(
            ConfigurationError, "participants contains duplicates"
        ):
            DomainCollectiveConfig(
                operation="all_reduce",
                after_stage="a_compute",
                participants=(0, 0),
            )
        with self.assertRaisesRegex(ConfigurationError, "operation must be one of"):
            DomainCollectiveConfig(
                operation="tp_collective",
                after_stage="a_compute",
                participants=(0, 1),
            )

        out_of_range = self._with_collective_fabric(
            _explicit_decode_cluster_raw(a_tp=4, a_dp=2, f_tp=1, f_dp=1)
        )
        out_of_range["af"]["execution_domains"][0]["communication_plan"] = [
            self._collective(participants=[0, 4])
        ]
        with self.assertRaisesRegex(ConfigurationError, "out-of-range.*\\[4\\].*TP=4"):
            ClusterConfig(**out_of_range)

        duplicate_stage = self._with_collective_fabric(
            _explicit_decode_cluster_raw(a_tp=4, a_dp=2, f_tp=1, f_dp=1)
        )
        duplicate_stage["af"]["execution_domains"][0]["communication_plan"] = [
            self._collective(),
            self._collective(operation="reduce_scatter"),
        ]
        with self.assertRaisesRegex(ConfigurationError, "ambiguous.*a_compute"):
            ClusterConfig(**duplicate_stage)

    def test_collective_requires_resolvable_fabric_and_lane(self):
        base = _explicit_decode_cluster_raw(a_tp=4, a_dp=2, f_tp=1, f_dp=1)
        base["af"]["execution_domains"][0]["communication_plan"] = [self._collective()]
        with self.assertRaisesRegex(ConfigurationError, "missing communication fabric"):
            ClusterConfig(**base)

        missing_lane = self._with_collective_fabric(deepcopy(base))
        del missing_lane["af"]["execution_domains"][0]["communication_plan"][0][
            "resource_lane"
        ]
        with self.assertRaisesRegex(
            ConfigurationError, "requires explicit resource_lane"
        ):
            ClusterConfig(**missing_lane)

        unknown_lane = self._with_collective_fabric(deepcopy(base))
        unknown_lane["af"]["execution_domains"][0]["communication_plan"][0][
            "resource_lane"
        ] = "missing"
        with self.assertRaisesRegex(ConfigurationError, "no resource lane 'missing'"):
            ClusterConfig(**unknown_lane)

    def test_collective_can_explicitly_share_the_half_duplex_af_lane(self):
        raw = _explicit_decode_cluster_raw(a_tp=2, f_tp=1, f_dp=1)
        raw["af"]["communication_fabrics"] = [
            {
                "resource_group": "af-fabric",
                "network": "net1",
                "duplex": "half",
                "lanes": [{"name": "shared", "capacity": 1}],
            }
        ]
        raw["af"]["execution_domains"][0]["communication_plan"] = [
            self._collective(
                operation="all_reduce",
                participants=[0, 1],
                resource_group="af-fabric",
                resource_lane="shared",
            )
        ]

        cluster = ClusterConfig(**raw)
        link = cluster.phase_to_af_link["decode"]
        collective = cluster.execution_domains["a-domain"].communication_plan[0]

        self.assertEqual(link.duplex, "half")
        self.assertEqual(
            (collective.resource_group, collective.resource_lane),
            link.communication_resources()[0].resource_key,
        )
