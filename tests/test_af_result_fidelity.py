import sys
import types
import unittest
from types import SimpleNamespace

try:
    import TransformerRoofline  # noqa: F401
except (ImportError, OSError):
    fake_roofline = types.ModuleType("TransformerRoofline")
    fake_roofline.TransformerRoofline = object
    sys.modules["TransformerRoofline"] = fake_roofline

from util.results import get_af_stats


class AFResultFidelityTest(unittest.TestCase):
    def _engine(self, phases, *, enabled=True):
        return SimpleNamespace(
            af_executor=(
                SimpleNamespace(trace=SimpleNamespace(enabled=False))
                if enabled
                else None
            ),
            af_iteration_summaries=[],
            af_runtime_summary=lambda: {},
            cluster_config=SimpleNamespace(af=SimpleNamespace(phases=tuple(phases))),
        )

    def test_decode_only_marks_prefill_aggregate_and_decode_rank_local(self):
        stats = get_af_stats(self._engine(("decode",)))

        self.assertEqual(
            stats["af_phase_fidelity"],
            {
                "prefill": "prefill_aggregate",
                "decode": "decode_af_rank_local",
            },
        )

    def test_prefill_af_is_labeled_as_compatibility_path(self):
        stats = get_af_stats(self._engine(("prefill",)))

        self.assertEqual(
            stats["af_phase_fidelity"],
            {
                "prefill": "af_prefill_compatibility",
                "decode": "decode_aggregate",
            },
        )

    def test_disabled_af_has_no_fidelity_claim(self):
        stats = get_af_stats(self._engine((), enabled=False))

        self.assertIsNone(stats["af_phase_fidelity"])


if __name__ == "__main__":
    unittest.main()
