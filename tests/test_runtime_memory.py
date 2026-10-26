"""Runtime KV observations replace estimates once and reject incompatible deployments."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from TokenSim.cli import build_parser
from TokenSim.cli.configuration import simulation_config
from TokenSim.config.cache_config import CacheConfig
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.runtime_memory import RuntimeMemoryProfile, runtime_memory_identity
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from tests.support.hardware import make_model

ROOT = Path(__file__).resolve().parents[1]


class RuntimeMemoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hardware = HardwareContext.load(ROOT / "data")
        cls.model = hardware.models.resolve(ROOT / "data/models/deepseek-v4-pro.yaml")
        cls.device = hardware.device("b200_sxm")
        cls.parallel = ClusterConfig.from_file(
            ROOT / "data/clusters/8_b200/dep8.json"
        ).effective_parallel_config()
        cls.settings = dict(
            block_size=256,
            utilization=0.9,
            max_num_batched_tokens=8192,
            max_num_seqs=49,
            speculative_tokens=3,
            max_concurrent_batches=2,
            attention_backend="module",
            moe_execution={"backend": "deep_gemm_amxf4_mega_moe"},
            execution=None,
        )

    def binding(self, **changes):
        return runtime_memory_identity(
            self.model, self.device, self.parallel, **(self.settings | changes)
        )

    def profile(self, **changes):
        values = dict(
            deployment_identity=self.binding(),
            kv_bytes_per_rank=[32 * (1 << 30)] * 8,
            uncertainty_bytes=1 << 20,
            provenance=dict(
                source="synthetic-runtime-memory-fixture",
                measurement_id="fixture-v1",
                framework="vllm",
                framework_version="test-version",
                graph_strategy="synthetic capture configuration",
            ),
        )
        return RuntimeMemoryProfile.from_value(values | changes)

    def cache(self, profile=None, **changes):
        values = dict(self.settings | changes)
        values["usable_memory_fraction"] = values.pop("utilization")
        values["runtime_memory_context"] = values.pop("execution")
        return CacheConfig(
            values.pop("block_size"),
            self.device,
            self.model,
            parallel_config=self.parallel,
            runtime_memory_profile=profile,
            **values,
        )

    def test_observed_bytes_size_real_group_slots_without_repeated_deductions(self):
        analytical = self.cache()
        observed = self.cache(self.profile())
        budget = observed.memory_budget
        self.assertEqual(
            budget["available_kv_bytes"], self.profile().kv_bytes_per_rank[0]
        )
        self.assertEqual(
            budget["analytical_available_kv_bytes"],
            analytical.memory_budget["available_kv_bytes"],
        )
        self.assertEqual(budget["deductions"], analytical.memory_budget["deductions"])
        self.assertGreater(observed.num_gpu_blocks, analytical.num_gpu_blocks)
        self.assertEqual(
            observed.num_gpu_blocks,
            budget["available_kv_bytes"] // observed.physical_slot_bytes,
        )
        observation = observed.to_dict()["memory_budget"]["runtime_observation"]
        self.assertFalse(observation["deductions_applied_to_observation"])
        self.assertEqual(observation["provenance"]["measurement_id"], "fixture-v1")
        self.assertEqual(len(observation["profile_sha256"]), 64)

    def test_incompatible_scheduler_and_geometry_fail_before_allocation(self):
        for changes in (
            {"max_num_seqs": 50},
            {"speculative_tokens": 0},
            {"max_num_batched_tokens": 4096},
            {"attention_backend": "generic"},
            {"utilization": 0.91},
            {"block_size": 128},
            {"execution": {"kv_connector": "different"}},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(ConfigurationError, "identity mismatch"),
            ):
                self.cache(self.profile(), **changes)
        changed_model = replace(
            self.model, max_position_embeddings=self.model.max_position_embeddings // 2
        )
        with self.assertRaisesRegex(ConfigurationError, "identity mismatch"):
            self.profile().apply(
                self.cache().memory_budget,
                runtime_memory_identity(
                    changed_model, self.device, self.parallel, **self.settings
                ),
                0,
                8,
            )

    def test_capacity_is_selected_by_global_rank_and_covers_every_rank(self):
        capacities = tuple((i + 1) * 1000000 for i in range(8))
        profile = self.profile(kv_bytes_per_rank=capacities, uncertainty_bytes=0)
        budget = self.cache().memory_budget
        for rank in (0, 3, 7):
            self.assertEqual(
                profile.apply(budget, self.binding(), rank, 8)["available_kv_bytes"],
                capacities[rank],
            )
        for rank, world in ((8, 8), (-1, 8), (0, 4)):
            with (
                self.subTest(rank=rank, world=world),
                self.assertRaises(ConfigurationError),
            ):
                profile.apply(budget, self.binding(), rank, world)

    def test_invalid_bytes_unknown_fields_and_provenance_are_rejected(self):
        for changes in (
            {"schema_version": True},
            {"schema_version": 2},
            {"unexpected": 1},
            {"kv_bytes_per_rank": []},
            {"kv_bytes_per_rank": [True]},
            {"kv_bytes_per_rank": [0]},
            {"kv_bytes_per_rank": [float("nan")]},
            {"provenance": {}},
            {"uncertainty_bytes": -1},
            {"uncertainty_bytes": 1 << 40},
        ):
            with self.subTest(changes=changes), self.assertRaises(ConfigurationError):
                self.profile(**changes)
        with self.assertRaisesRegex(ConfigurationError, "exceed"):
            self.cache(self.profile(kv_bytes_per_rank=[180 * (1 << 30)] * 8))
        with self.assertRaisesRegex(ConfigurationError, "one physical KV slot"):
            self.cache(self.profile(kv_bytes_per_rank=[1] * 8, uncertainty_bytes=0))

    def test_source_mappings_are_copied_and_file_cli_round_trip_is_exact(self):
        raw = asdict(self.profile())
        original = deepcopy(raw)
        profile = RuntimeMemoryProfile.from_value(raw)
        raw["deployment_identity"]["settings"]["block_size"] = 123
        self.assertEqual(asdict(profile), original)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text(json.dumps(original))
            args = build_parser().parse_args(["--runtime_memory_profile", str(path)])
            self.assertEqual(simulation_config(args).runtime_memory_profile, profile)
            path.write_text('"self-referencing-file-is-invalid"')
            with self.assertRaises(ConfigurationError):
                RuntimeMemoryProfile.from_file(path)

    def test_observation_can_replace_an_exhausted_analytical_envelope(self):
        settings = dict(utilization=0.75)
        with self.assertRaisesRegex(ConfigurationError, "exhaust"):
            self.cache(**settings)
        profile = self.profile(
            deployment_identity=self.binding(**settings),
            kv_bytes_per_rank=[1 << 30] * 8,
        )
        observed = self.cache(profile, **settings)
        self.assertLessEqual(observed.memory_budget["analytical_available_kv_bytes"], 0)
        self.assertGreater(observed.num_gpu_blocks, 0)

    def test_uniform_attention_does_not_silently_ignore_a_profile(self):
        with self.assertRaisesRegex(ConfigurationError, "compressed attention"):
            CacheConfig(
                16, self.device, make_model(), runtime_memory_profile=self.profile()
            )
