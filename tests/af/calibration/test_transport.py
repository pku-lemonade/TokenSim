import copy
from dataclasses import asdict
import unittest

from TokenSim.af.calibration.services import TransportServiceProfile
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.domains import CommunicationDomainConfig, MODEL_VERSION
from tests.af.support.profiles import NativeFixture


class TransportProfileTest(unittest.TestCase):
    def test_runtime_prices_each_message_once_and_cache_preserves_timeline(self):
        from TokenSim.af.factory import build_model_runtime
        from TokenSim.af.factory import simulate_decode

        NativeFixture.setUpClass()
        fixture = NativeFixture()
        domains = CommunicationDomainConfig(
            node_count=1, device_indices={"a-pool:0": 2, "f-pool:0": 3}
        )
        profile = self.document()
        profile["domains"] = asdict(domains)
        profile["device_indices"] = {"a-pool:0": 2, "f-pool:0": 3}
        profile["records"].append(
            {
                **copy.deepcopy(profile["records"][0]),
                "source_device": 3,
                "target_device": 2,
            }
        )
        for dtype, size in [("int64", 8), ("uint8", 179)]:
            profile["records"].append(
                dict(
                    source_device=2,
                    target_device=3,
                    backend="torch_distributed_nccl",
                    dtype=dtype,
                    points=[dict(bytes=size, service_us=11)],
                    interpolation="linear_within_training_range",
                )
            )
        times = []
        for cached in (False, True):
            doc = fixture.document(
                communication_timing=dict(enabled=True, native_control_enabled=True),
                communication_domains=asdict(domains),
                transfer_topology_mode="physical",
                calibration_environment={"gpu_pair": "A,F"},
                transport_service_profile=profile,
                service_estimate_cache=cached,
            )
            runtime = build_model_runtime(
                doc, hardware=fixture.base.hardware, model=fixture.model
            )
            executor, rows, _, _ = simulate_decode(
                runtime, 1, 129, steps=2, fallback="analytical_only"
            )
            wires = [
                e
                for e in executor.trace.events
                if e.event_type in ("transfer", "control_transfer")
            ]
            self.assertEqual(len(wires), 12)
            for e in wires:
                self.assertAlmostEqual(
                    e.service_duration * 1e6, 8 if e.event_type == "transfer" else 11
                )
            times.append([r["latency"] if "latency" in r else r for r in rows])
        self.assertEqual(times[0], times[1])

    def document(self):
        return dict(
            schema_version=1,
            unit="us",
            profile_id="synthetic-transport",
            timing_scope="transport_service",
            contains_wait=False,
            contains_cpu_submit=False,
            model_version=MODEL_VERSION,
            environment={"gpu_pair": "A,F"},
            domains=asdict(CommunicationDomainConfig()),
            device_indices={"a": 2, "f": 3},
            evidence=dict(
                kind="synthetic_test",
                source="unit test",
                window_count=3,
                sample_count=150,
                measurement_boundary="after_both_device_kernels_started_until_both_complete",
                peer_delay_audit_passed=True,
                limitations=["synthetic data, not a calibration"],
            ),
            records=[
                dict(
                    source_device=2,
                    target_device=3,
                    backend="pynccl",
                    dtype="bf16",
                    points=[
                        dict(bytes=512, service_us=8),
                        dict(bytes=4096, service_us=10),
                    ],
                    interpolation="linear_within_training_range",
                )
            ],
        )

    def load(self, d):
        return TransportServiceProfile(
            d,
            environment={"gpu_pair": "A,F"},
            domains=CommunicationDomainConfig(),
            device_indices={"a": 2, "f": 3},
        )

    def test_bounded_service_interpolation_excludes_wait_and_cpu(self):
        p = self.load(self.document())
        query = dict(source=2, target=3, backend="pynccl", dtype="bf16", size=2304)
        self.assertEqual(p.estimate_us(**query), 9)
        for patch in (
            {"size": 4097},
            {"backend": "torch_distributed_nccl"},
            {"source": 3, "target": 2},
            {"dtype": "fp32"},
        ):
            with self.assertRaises(ConfigurationError):
                p.estimate_us(**{**query, **patch})
        for patch in (
            {"contains_wait": True},
            {"contains_cpu_submit": True},
            {"environment": {"gpu_pair": "A,B"}},
            {"device_indices": {"a": 2, "f": 4}},
        ):
            with self.assertRaises(ConfigurationError):
                self.load({**self.document(), **patch})

    def test_mutation_and_invalid_samples_rejected(self):
        d = self.document()
        p = self.load(d)
        fingerprint = p.fingerprint
        d["records"][0]["points"][0]["service_us"] = 99
        self.assertEqual(
            p.estimate_us(source=2, target=3, backend="pynccl", dtype="bf16", size=512),
            8,
        )
        self.assertNotEqual(self.load(d).fingerprint, fingerprint)
        for value in (True, -1, float("nan"), float("inf")):
            bad = self.document()
            bad["records"][0]["points"][0]["service_us"] = value
            with self.assertRaises(ConfigurationError):
                self.load(bad)
        duplicate = self.document()
        duplicate["records"] *= 2
        with self.assertRaises(ConfigurationError):
            self.load(duplicate)


if __name__ == "__main__":
    unittest.main()
