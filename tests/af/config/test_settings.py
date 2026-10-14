"""Af config."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import unittest

from TokenSim.config.af.links import AFEffectiveDTypes, AFLinkConfig
from TokenSim.config.af.settings import AFConfig
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.cluster_config import WorkerGroupConfig
from TokenSim.config.constants import (
    AF_PHASES,
    LEGACY_ROLE_CAPABILITIES,
    WORKER_CAPABILITIES,
)
from TokenSim.errors import ConfigurationError
from tests.af.support.config_validation import (
    _af_group,
    _cluster,
    _full_link,
    _raw_link,
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

    def test_dtype_defaults_resolve_after_raw_presence_check(self):
        config = AFConfig()
        effective = config.resolve_dtypes(model_dtype="BFloat16")

        self.assertIsInstance(effective, AFEffectiveDTypes)
        self.assertEqual(
            (
                effective.weight_dtype,
                effective.activation_dtype,
                effective.kv_cache_dtype,
                effective.a_to_f_dtype,
                effective.f_to_a_dtype,
            ),
            ("bf16", "bf16", "bf16", "bf16", "bf16"),
        )
        self.assertFalse(effective.legacy_mapped)
        self.assertEqual(config.explicit_dtype_fields, frozenset())

    def test_five_dtype_roles_resolve_independently(self):
        link = _full_link(a_to_f_dtype="float8_e4m3fn", f_to_a_dtype="float32")
        config = AFConfig(
            weight_dtype="FP8",
            activation_dtype="BF16",
            kv_cache_dtype="half",
            links=[link],
        )

        effective = config.resolve_dtypes(model_dtype="fp32", link=link)

        self.assertEqual(effective.weight_dtype, "fp8")
        self.assertEqual(effective.activation_dtype, "bf16")
        self.assertEqual(effective.kv_cache_dtype, "fp16")
        self.assertEqual(effective.a_to_f_dtype, "fp8")
        self.assertEqual(effective.f_to_a_dtype, "fp32")
        self.assertEqual(
            effective.as_dict()["bytes_per_element"],
            {
                "a_to_f_dtype": 1,
                "activation_dtype": 2,
                "f_to_a_dtype": 4,
                "kv_cache_dtype": 2,
                "weight_dtype": 1,
            },
        )

    def test_legacy_dtype_maps_all_roles_and_reports_deprecation(self):
        config = AFConfig(af_dtype="BFloat16")
        effective = config.resolve_dtypes(model_dtype="fp32")

        self.assertEqual(
            {
                effective.weight_dtype,
                effective.activation_dtype,
                effective.kv_cache_dtype,
                effective.a_to_f_dtype,
                effective.f_to_a_dtype,
            },
            {"bf16"},
        )
        self.assertTrue(effective.legacy_mapped)
        self.assertEqual(effective.deprecated_fields, ("af_dtype",))
        self.assertEqual(set(effective.source_by_role.values()), {"legacy:af_dtype"})

    def test_legacy_and_new_dtype_fields_cannot_mix(self):
        with self.assertRaisesRegex(ConfigurationError, "af_dtype.*weight_dtype"):
            AFConfig(af_dtype="bf16", weight_dtype="bf16")
        with self.assertRaisesRegex(ConfigurationError, "af_dtype.*a_to_f_dtype"):
            AFConfig(
                af_dtype="bf16",
                links=[_full_link(a_to_f_dtype="fp8")],
            )
        with self.assertRaisesRegex(ConfigurationError, "activation_dtype"):
            AFConfig(activation_dtype=None)

    def test_model_kv_dtype_precedes_activation_fallback(self):
        effective = AFConfig(activation_dtype="bf16").resolve_dtypes(
            model_dtype="fp32", model_kv_cache_dtype="float16"
        )

        self.assertEqual(effective.weight_dtype, "fp32")
        self.assertEqual(effective.activation_dtype, "bf16")
        self.assertEqual(effective.kv_cache_dtype, "fp16")
        self.assertEqual(
            effective.source_by_role["kv_cache_dtype"], "model.kv_cache_dtype"
        )

    def test_dataclass_replace_preserves_raw_dtype_presence(self):
        default_link = _full_link()
        copied_link = replace(default_link, name="copied")
        typed_link = replace(default_link, a_to_f_dtype="fp8")

        self.assertEqual(copied_link.explicit_dtype_fields, frozenset())
        self.assertEqual(typed_link.explicit_dtype_fields, frozenset({"a_to_f_dtype"}))
        with self.assertRaisesRegex(ConfigurationError, "f_to_a_dtype"):
            AFLinkConfig(
                **{
                    **_raw_link(default_link),
                    "f_to_a_dtype": None,
                }
            )


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


class AFLoadingAndEntryPointTest(unittest.TestCase):
    def test_file_and_python_construction_normalize_equally(self):
        path = (
            Path(__file__).parents[3]
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
        root = Path(__file__).parents[3]
        example_dir = root / "data/clusters/af"
        examples = {
            path.name: ClusterConfig.from_file(path)
            for path in sorted(example_dir.glob("*.json"))
        }
        full = examples["prefill_decode_full_duplex.json"]
        decode = examples["decode_half_duplex.json"]
        expert_tp = examples["decode_tp2_dp2_ep1.json"]
        rank_reuse = examples["decode_tp2_dp2_ep4_rank_reuse.json"]

        self.assertEqual(full.af.phases, ("prefill", "decode"))
        self.assertEqual(
            set(full.af_communication_resources),
            {("af-fabric", "a_to_f"), ("af-fabric", "f_to_a")},
        )
        self.assertEqual(decode.af.phases, ("decode",))
        self.assertEqual(
            set(decode.af_communication_resources), {("shared-af-fabric", "shared")}
        )
        for cluster in (expert_tp, rank_reuse):
            self.assertEqual(
                cluster.execution_domains["decode-a"].required_rank_count, 4
            )
            self.assertEqual(
                cluster.execution_domains["decode-f"].required_rank_count, 4
            )
            self.assertEqual(len(cluster.workers()), 9)
            self.assertEqual(
                {
                    (plan.after_stage, plan.operation)
                    for plan in cluster.execution_domains["decode-a"].communication_plan
                },
                {
                    ("a_compute", "all_reduce"),
                    ("residual", "all_gather"),
                },
            )

        self.assertEqual(
            expert_tp.execution_domains["decode-f"].expert_parallel.mode,
            "expert_tp",
        )
        self.assertEqual(
            rank_reuse.execution_domains["decode-f"].expert_parallel.mode,
            "rank_reuse",
        )
        for cluster in examples.values():
            groups = {group.name: group for group in cluster.worker_groups}
            for domain in cluster.execution_domains.values():
                self.assertEqual(
                    domain.required_rank_count,
                    groups[domain.worker_group].num_workers,
                    msg=f"{domain.name} must not rely on EP-only workers",
                )

    def test_af_block_size_is_not_kv_block_size(self):
        cluster = _cluster()
        cluster.af.af_block_size = 7
        args = argparse.Namespace(block_size=16)

        self.assertEqual(cluster.af.af_block_size, 7)
        self.assertEqual(args.block_size, 16)
