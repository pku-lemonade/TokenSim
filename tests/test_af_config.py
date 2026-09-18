from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import MagicMock

from TokenSim.config.cluster_config import (
    AFConfig,
    AFLinkConfig,
    ClusterConfig,
    WorkerGroupConfig,
)
from TokenSim.config.parallel_config import ParallelConfig
from TokenSim.config.constants import (
    AF_PHASES,
    LEGACY_ROLE_CAPABILITIES,
    WORKER_CAPABILITIES,
)
from TokenSim.errors import ConfigurationError

import benchmark


def _full_link(**overrides) -> AFLinkConfig:
    values = {
        "name": "af-link",
        "phases": ("prefill", "decode"),
        "attention_group": "attention",
        "ffn_group": "ffn",
        "network": "net1",
        "resource_group": "fabric",
        "duplex": "full",
        "a_to_f_capacity": 1,
        "f_to_a_capacity": 1,
    }
    values.update(overrides)
    return AFLinkConfig(**values)


def _af_group(name: str, capability: str, **overrides) -> WorkerGroupConfig:
    values = {
        "name": name,
        "capabilities": (capability,),
        "hardware": "device",
        "num_workers": 1,
        "network": "net1",
    }
    values.update(overrides)
    return WorkerGroupConfig(**values)


def _cluster(
    *,
    phases=("prefill", "decode"),
    links=None,
    extra_groups=None,
    networks=None,
) -> ClusterConfig:
    groups = [
        _af_group("attention", "af_attention"),
        _af_group("ffn", "af_ffn"),
    ]
    groups.extend(extra_groups or [])
    return ClusterConfig(
        num_workers=sum(group.num_workers for group in groups),
        networks=networks or {"net1": "test-network"},
        worker_groups=groups,
        af=AFConfig(
            enabled=True,
            phases=phases,
            links=[_full_link(phases=phases)] if links is None else links,
        ),
    )


class AFConfigModelTest(unittest.TestCase):
    def test_public_constant_sets(self):
        self.assertEqual(AF_PHASES, {"prefill", "decode"})
        self.assertEqual(
            WORKER_CAPABILITIES,
            {"prefill", "decode_full", "af_attention", "af_ffn"},
        )

    def test_af_defaults_are_disabled_and_inert(self):
        cluster = ClusterConfig(num_workers=0, networks={})

        self.assertFalse(cluster.af.enabled)
        self.assertEqual(cluster.af.phases, ("prefill", "decode"))
        self.assertEqual(cluster.af.af_block_size, 1)
        self.assertEqual(cluster.af.links, [])
        self.assertTrue(cluster.af.trace_enabled)
        self.assertEqual(cluster.phase_to_af_link, {})
        self.assertEqual(cluster.af_communication_resources, {})

    def test_phase_order_is_canonical(self):
        self.assertEqual(
            AFConfig(phases=("decode", "prefill")).phases, ("prefill", "decode")
        )
        self.assertEqual(AFConfig(phases=("prefill",)).phases, ("prefill",))
        self.assertEqual(AFConfig(phases=("decode",)).phases, ("decode",))

    def test_invalid_phases_are_rejected(self):
        for phases in ((), ("decode", "decode"), ("training",)):
            with self.subTest(phases=phases), self.assertRaises(ConfigurationError):
                AFConfig(phases=phases)

    def test_invalid_block_sizes_are_rejected(self):
        for value in (0, -1, True, 1.5, "2"):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                AFConfig(af_block_size=value)

    def test_boolean_fields_are_strict(self):
        with self.assertRaises(ConfigurationError):
            AFConfig(enabled=1)
        with self.assertRaises(ConfigurationError):
            AFConfig(trace_enabled="true")


class WorkerCapabilityTest(unittest.TestCase):
    def test_legacy_roles_are_normalized(self):
        for role, expected in LEGACY_ROLE_CAPABILITIES.items():
            with self.subTest(role=role):
                group = WorkerGroupConfig(role=role, hardware="device", num_workers=1)
                self.assertEqual(set(group.capabilities), set(expected))

        cluster = ClusterConfig(
            num_workers=3,
            networks={"net1": "test-network"},
            worker_groups=[
                WorkerGroupConfig(role="prefill", hardware="device", num_workers=1),
                WorkerGroupConfig(role="decode", hardware="device", num_workers=1),
                WorkerGroupConfig(role="hybrid", hardware="device", num_workers=1),
            ],
        )
        self.assertEqual(
            [group.name for group in cluster.worker_groups],
            ["legacy-prefill-0", "legacy-decode-0", "legacy-hybrid-0"],
        )

        positional = WorkerGroupConfig("hybrid", "device", 1, "net1")
        self.assertEqual(positional.role, "hybrid")
        self.assertEqual(positional.hardware, "device")
        self.assertEqual(positional.num_workers, 1)
        self.assertEqual(positional.network, "net1")

    def test_explicit_capabilities_reach_workers(self):
        group = WorkerGroupConfig(
            role="prefill",
            name="mixed",
            capabilities=("af_attention", "prefill"),
            hardware="device",
            num_workers=1,
        )
        worker = group.workers({"net1": "test-network"})[0]

        self.assertEqual(worker.name, "mixed")
        self.assertEqual(worker.capabilities, ("prefill", "af_attention"))
        self.assertEqual(worker.role, "prefill")

    def test_invalid_group_fields_are_rejected(self):
        cases = [
            {"role": "bad", "hardware": "device", "num_workers": 1},
            {"hardware": "device", "num_workers": 1, "capabilities": ("bad",)},
            {
                "hardware": "device",
                "num_workers": 1,
                "capabilities": ("af_ffn", "af_ffn"),
            },
            {"hardware": "device", "num_workers": 0, "capabilities": ("af_ffn",)},
            {"hardware": "device", "num_workers": True, "capabilities": ("af_ffn",)},
            {"hardware": "device", "num_workers": 1},
            {
                "role": "prefill",
                "hardware": "device",
                "num_workers": 1,
                "capabilities": ("decode_full",),
            },
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ConfigurationError):
                WorkerGroupConfig(**values)

    def test_duplicate_names_and_unknown_network_are_rejected(self):
        duplicate = [
            _af_group("same", "af_attention"),
            _af_group("same", "af_ffn"),
        ]
        with self.assertRaisesRegex(ConfigurationError, "duplicate.*same"):
            ClusterConfig(
                num_workers=2, networks={"net1": "network"}, worker_groups=duplicate
            )
        with self.assertRaisesRegex(ConfigurationError, "unknown network"):
            ClusterConfig(
                num_workers=1,
                networks={"net2": "network"},
                worker_groups=[_af_group("attention", "af_attention")],
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


class AFLoadingAndEntryPointTest(unittest.TestCase):
    def test_file_and_python_construction_normalize_equally(self):
        path = (
            Path(__file__).parents[1]
            / "data/clusters/af/prefill_decode_full_duplex.json"
        )
        raw = json.loads(path.read_text())
        direct = ClusterConfig(**raw)
        loaded = ClusterConfig.from_file(path)

        self.assertEqual(loaded.af.phases, direct.af.phases)
        self.assertEqual(loaded.worker_groups, direct.worker_groups)
        self.assertEqual(loaded.phase_to_af_link, direct.phase_to_af_link)
        self.assertEqual(
            loaded.af_communication_resources, direct.af_communication_resources
        )

    def test_repository_examples_parse(self):
        root = Path(__file__).parents[1]
        full = ClusterConfig.from_file(
            root / "data/clusters/af/prefill_decode_full_duplex.json"
        )
        decode = ClusterConfig.from_file(
            root / "data/clusters/af/decode_half_duplex.json"
        )
        combined = ClusterConfig.from_file(root / "data/clusters/af/tp2_dp2_ep2.json")

        self.assertEqual(full.af.phases, ("prefill", "decode"))
        self.assertEqual(
            set(full.af_communication_resources),
            {("af-fabric", "a_to_f"), ("af-fabric", "f_to_a")},
        )
        self.assertEqual(decode.af.phases, ("decode",))
        self.assertEqual(
            set(decode.af_communication_resources), {("shared-af-fabric", "shared")}
        )
        self.assertEqual(combined.execution_domains["a-domain"].required_rank_count, 4)
        self.assertEqual(combined.execution_domains["f-domain"].required_rank_count, 8)

    def test_explicit_domains_use_independent_one_to_one_worker_topologies(self):
        raw = {
            "num_workers": 4,
            "networks": {"net1": "network"},
            "worker_groups": [
                {
                    "name": "attention",
                    "capabilities": ["af_attention"],
                    "hardware": "a-device",
                    "num_workers": 2,
                },
                {
                    "name": "ffn",
                    "capabilities": ["af_ffn"],
                    "hardware": "f-device",
                    "num_workers": 2,
                },
            ],
            "af": {
                "enabled": True,
                "phases": ["prefill", "decode"],
                "links": [
                    {
                        **asdict(_full_link()),
                        "attention_domain": "decode-a",
                        "ffn_domain": "decode-f",
                    }
                ],
                "execution_domains": [
                    {
                        "name": "decode-a",
                        "capability": "af_attention",
                        "phases": ["prefill", "decode"],
                        "worker_group": "attention",
                        "parallel": {"tensor_parallel_size": 2},
                        "logical_instances": [
                            {
                                "name": "a0",
                                "rank_mapping": [
                                    {
                                        "domain_rank": rank,
                                        "tp_rank": rank,
                                        "pp_rank": 0,
                                        "dp_rank": 0,
                                        "physical_worker": {
                                            "group_name": "attention",
                                            "worker_index": rank,
                                        },
                                    }
                                    for rank in range(2)
                                ],
                            }
                        ],
                    },
                    {
                        "name": "decode-f",
                        "capability": "af_ffn",
                        "phases": ["prefill", "decode"],
                        "worker_group": "ffn",
                        "parallel": {"expert_parallel_size": 2},
                        "logical_instances": [
                            {
                                "name": "f0",
                                "rank_mapping": [
                                    {
                                        "domain_rank": rank,
                                        "tp_rank": 0,
                                        "pp_rank": 0,
                                        "dp_rank": 0,
                                        "ep_rank": rank,
                                        "physical_worker": {
                                            "group_name": "ffn",
                                            "worker_index": rank,
                                        },
                                    }
                                    for rank in range(2)
                                ],
                            }
                        ],
                    },
                ],
            },
        }

        cluster = ClusterConfig(**raw)
        workers = cluster.workers(ParallelConfig(tensor_parallel_size=8))

        self.assertEqual(len(workers), 4)
        self.assertEqual(cluster.execution_domains["decode-a"].parallel.world_size, 2)
        self.assertEqual(cluster.phase_to_af_domains["decode"][1].name, "decode-f")
        self.assertEqual(workers[0].physical_worker_ref.identity, "attention:0")

        invalid = deepcopy(raw)
        invalid["af"]["execution_domains"][0]["logical_instances"] = [
            {
                "name": "a0",
                "rank_mapping": raw["af"]["execution_domains"][0]["logical_instances"][
                    0
                ]["rank_mapping"][:1],
            }
        ]
        with self.assertRaisesRegex(ConfigurationError, "decode-a.*cover dp replica"):
            ClusterConfig(**invalid)

        duplicate = deepcopy(raw)
        duplicate_mapping = duplicate["af"]["execution_domains"][0][
            "logical_instances"
        ][0]["rank_mapping"]
        duplicate_mapping[1]["physical_worker"]["worker_index"] = 0
        with self.assertRaisesRegex(
            ConfigurationError,
            "decode-a.*a0.*rank 1.*attention:0.*a0.*rank 0.*one-to-one",
        ):
            ClusterConfig(**duplicate)

        unmapped = deepcopy(raw)
        unmapped["worker_groups"][0]["num_workers"] = 3
        unmapped["num_workers"] = 4
        with self.assertRaisesRegex(
            ConfigurationError, "decode-a.*unmapped physical workers.*attention:2"
        ):
            ClusterConfig(**unmapped)

        cross_group = deepcopy(raw)
        cross_group["af"]["execution_domains"][0]["logical_instances"][0][
            "rank_mapping"
        ][1]["physical_worker"]["group_name"] = "ffn"
        with self.assertRaisesRegex(
            ConfigurationError, "decode-a.*rank 1.*ffn.*attention"
        ):
            ClusterConfig(**cross_group)

    def test_domain_topology_matrix_keeps_a_f_world_sizes_independent(self):
        def replica(group, name, dp_rank, tp, ep=1, worker_offset=0):
            ranks_per_replica = tp * ep
            return {
                "name": name,
                "rank_mapping": [
                    {
                        "domain_rank": dp_rank * ranks_per_replica + local_rank,
                        "tp_rank": local_rank % tp,
                        "pp_rank": 0,
                        "dp_rank": dp_rank,
                        "ep_rank": local_rank // tp,
                        "physical_worker": {
                            "group_name": group,
                            "worker_index": worker_offset + local_rank,
                        },
                    }
                    for local_rank in range(ranks_per_replica)
                ],
            }

        raw = {
            "num_workers": 6,
            "networks": {"net1": "network"},
            "worker_groups": [
                {
                    "name": "attention",
                    "capabilities": ["af_attention"],
                    "hardware": "a-device",
                    "num_workers": 4,
                },
                {
                    "name": "ffn",
                    "capabilities": ["af_ffn"],
                    "hardware": "f-device",
                    "num_workers": 2,
                },
            ],
            "af": {
                "enabled": True,
                "links": [
                    {
                        **asdict(_full_link()),
                        "attention_domain": "decode-a",
                        "ffn_domain": "decode-f",
                    }
                ],
                "execution_domains": [
                    {
                        "name": "decode-a",
                        "capability": "af_attention",
                        "phases": ["prefill", "decode"],
                        "worker_group": "attention",
                        "parallel": {
                            "tensor_parallel_size": 2,
                            "pipeline_parallel_size": 1,
                            "data_parallel_size": 2,
                        },
                        "logical_instances": [
                            replica("attention", "a0", 0, 2, worker_offset=0),
                            replica("attention", "a1", 1, 2, worker_offset=2),
                        ],
                    },
                    {
                        "name": "decode-f",
                        "capability": "af_ffn",
                        "phases": ["prefill", "decode"],
                        "worker_group": "ffn",
                        "parallel": {
                            "tensor_parallel_size": 1,
                            "pipeline_parallel_size": 1,
                            "data_parallel_size": 1,
                            "expert_parallel_size": 2,
                        },
                        "logical_instances": [
                            replica("ffn", "f0", 0, 1, ep=2),
                        ],
                    },
                ],
            },
        }

        cluster = ClusterConfig(**raw)

        self.assertEqual(cluster.execution_domains["decode-a"].parallel.world_size, 4)
        self.assertEqual(cluster.execution_domains["decode-f"].parallel.world_size, 1)
        self.assertEqual(cluster.execution_domains["decode-f"].required_rank_count, 2)
        self.assertEqual(
            cluster.execution_domains["decode-f"].parallel.expert_parallel_size, 2
        )
        self.assertEqual(
            len(cluster.workers(ParallelConfig(tensor_parallel_size=8))), 6
        )

        unsupported_pp = deepcopy(raw)
        unsupported_pp["af"]["execution_domains"][0]["parallel"][
            "pipeline_parallel_size"
        ] = 2
        with self.assertRaisesRegex(
            ConfigurationError, "decode-a.*pipeline_parallel_size=2.*only supports"
        ):
            ClusterConfig(**unsupported_pp)

        invalid_a_ep = deepcopy(raw)
        invalid_a_ep["af"]["execution_domains"][0]["parallel"][
            "expert_parallel_size"
        ] = 2
        with self.assertRaisesRegex(
            ConfigurationError, "decode-a.*expert_parallel_size.*only valid for af_ffn"
        ):
            ClusterConfig(**invalid_a_ep)

        mixed_dp = deepcopy(raw)
        mixed_dp["af"]["execution_domains"][0]["logical_instances"][0]["rank_mapping"][
            1
        ]["dp_rank"] = 1
        with self.assertRaisesRegex(
            ConfigurationError, "decode-a.*a0.*exactly one dp_rank"
        ):
            ClusterConfig(**mixed_dp)

    def test_af_block_size_is_not_kv_block_size(self):
        cluster = _cluster()
        cluster.af.af_block_size = 7
        args = argparse.Namespace(block_size=16)

        self.assertEqual(cluster.af.af_block_size, 7)
        self.assertEqual(args.block_size, 16)

    def test_disabled_af_keeps_cli_block_size_on_kv_paths(self):
        cluster = ClusterConfig(num_workers=0, networks={})
        cluster.af.af_block_size = 7
        model = MagicMock(distribution="burst")
        model.from_args.return_value = model
        env = MagicMock(now=0)
        args = argparse.Namespace(
            cluster="cluster.json",
            latency_backend="operator_table",
            latency_fallback="table_first",
            operator_backend=None,
            decode_context_bucket=128,
            data_root="./data",
            kv_transfer_config=None,
            model_config_path="model.json",
            verbose="none",
            program_id=0,
            block_size=16,
            batching="dynamic",
            qps=1,
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_parallem_sum=1,
            max_occupy_ratio=1,
            random_seed=0,
            debug_print=False,
            sim_time=None,
        )
        with (
            patch.object(benchmark, "reset_g_time"),
            patch.object(benchmark.ClusterConfig, "from_file", return_value=cluster),
            patch.object(benchmark.PSLAConfig, "from_file", return_value=model),
            patch.object(benchmark.HardwareContext, "load", return_value=MagicMock()),
            patch.object(benchmark.simpy, "Environment", return_value=env),
            patch.object(benchmark, "build_parallel_config", return_value=MagicMock()),
            patch.object(benchmark, "validate_moe_parallel_config"),
            patch.object(benchmark, "TqdmManager", return_value=MagicMock()),
            patch.object(
                benchmark, "get_requests", return_value=([], [], [])
            ) as requests,
            patch.object(benchmark, "LLMEngine", return_value=MagicMock()) as engine,
            patch.object(benchmark, "LLMSource", return_value=object()),
            patch.object(benchmark, "check_results"),
        ):
            benchmark.main(args)

        self.assertEqual(requests.call_args.kwargs["block_size"], 16)
        self.assertEqual(engine.call_args.kwargs["block_size"], 16)
        self.assertNotEqual(cluster.af.af_block_size, 16)

    def test_benchmark_rejects_missing_af_providers_before_backends_or_simpy(self):
        args = argparse.Namespace(cluster="af.json")
        with (
            patch.object(benchmark, "reset_g_time"),
            patch.object(benchmark.ClusterConfig, "from_file", return_value=_cluster()),
            patch.object(benchmark, "get_latency_backend_type") as latency_backend,
            patch.object(benchmark.HardwareContext, "load") as hardware_load,
            patch.object(benchmark.simpy, "Environment") as environment,
            self.assertRaisesRegex(ConfigurationError, "compute latency provider"),
        ):
            benchmark.main(args)

        latency_backend.assert_not_called()
        hardware_load.assert_not_called()
        environment.assert_not_called()

    def test_benchmark_injects_complete_af_runtime_into_engine(self):
        cluster = _cluster()
        model = MagicMock(distribution="burst")
        model.from_args.return_value = model
        env = MagicMock(now=0)
        compute_provider = object()
        transfer_provider = object()
        args = argparse.Namespace(
            cluster="af.json",
            af_compute_provider=compute_provider,
            af_transfer_provider=transfer_provider,
            af_dtype="bf16",
            latency_backend="operator_table",
            latency_fallback="table_first",
            operator_backend=None,
            decode_context_bucket=128,
            data_root="./data",
            kv_transfer_config=None,
            model_config_path="model.json",
            verbose="none",
            program_id=0,
            block_size=16,
            batching="dynamic",
            qps=1,
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_parallem_sum=1,
            max_occupy_ratio=1,
            random_seed=0,
            debug_print=False,
            sim_time=None,
        )
        with (
            patch.object(benchmark, "reset_g_time"),
            patch.object(benchmark.ClusterConfig, "from_file", return_value=cluster),
            patch.object(benchmark.PSLAConfig, "from_file", return_value=model),
            patch.object(benchmark.HardwareContext, "load", return_value=MagicMock()),
            patch.object(benchmark.simpy, "Environment", return_value=env),
            patch.object(benchmark, "build_parallel_config", return_value=MagicMock()),
            patch.object(benchmark, "validate_moe_parallel_config"),
            patch.object(benchmark, "TqdmManager", return_value=MagicMock()),
            patch.object(benchmark, "get_requests", return_value=([], [], [])),
            patch.object(benchmark, "LLMEngine", return_value=MagicMock()) as engine,
            patch.object(benchmark, "LLMSource", return_value=object()),
            patch.object(benchmark, "check_results"),
        ):
            benchmark.main(args)

        self.assertIs(engine.call_args.kwargs["af_compute_provider"], compute_provider)
        self.assertIs(
            engine.call_args.kwargs["af_transfer_provider"], transfer_provider
        )
        self.assertEqual(engine.call_args.kwargs["af_dtype"], "bf16")


if __name__ == "__main__":
    unittest.main()
