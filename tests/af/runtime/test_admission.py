"""Window-aware cohort residence uses all physical pipeline KV owners."""

from dataclasses import replace
import unittest

from TokenSim.af.runtime.admission import ResidentKV, rank_kv_bytes
from TokenSim.af.runtime.compiler import compile_af_runtime_plan
from TokenSim.errors import ConfigurationError
from tests.af.support.model import _runtime_hardware
from tests.af.support.pipeline import pipeline_cluster


class ResidentKVTest(unittest.TestCase):
    def test_local_windows_are_bounded_and_full_layers_grow(self):
        hardware = _runtime_hardware()
        model = replace(hardware.model("TestModel"), attention_windows=(4, 0))
        cluster = pipeline_cluster()
        plan = compile_af_runtime_plan(
            cluster_config=cluster,
            worker_configs=cluster.workers(),
            hardware=hardware,
            model=model,
            block_size=4,
        )
        local, full = plan.binding("a-domain", 0), plan.binding("a-domain", 2)
        self.assertEqual(
            rank_kv_bytes(plan, local, 100), rank_kv_bytes(plan, local, 1000)
        )
        self.assertEqual(
            rank_kv_bytes(plan, full, 1000), 10 * rank_kv_bytes(plan, full, 100)
        )
        ledger = ResidentKV(plan)
        ledger.admit(1, "a-domain", "a0", maximum_context=1000, current_context=100)
        initial = ledger.summary()
        ledger.commit(1, 200)
        current = ledger.summary()
        self.assertEqual(
            initial["reserved_peak_kv_bytes"], current["reserved_peak_kv_bytes"]
        )
        self.assertGreater(
            sum(current["resident_current_kv_bytes"].values()),
            sum(initial["resident_current_kv_bytes"].values()),
        )
        with self.assertRaises(ConfigurationError):
            ledger.admit(2, "a-domain", "a0", maximum_context=10**12, current_context=0)
        self.assertEqual(len(ledger.leases), 1)
        ledger.release(1)
        self.assertEqual(sum(ledger.reserved.values()), 0)
