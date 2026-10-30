import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from TokenSim.errors import WorkloadValidationError
from scripts import analyze_lpx_gpu_cc302 as analysis
from scripts import analyze_lpx_cc16k_peak as peak


class Cc302AnalysisTest(unittest.TestCase):
    def test_changed_native_source_requires_a_new_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.py"
            source.write_text("original")
            audit = Path(directory) / "audit.json"
            audit.write_text(
                json.dumps(
                    dict(
                        source_hashes={
                            str(source): hashlib.sha256(source.read_bytes()).hexdigest()
                        }
                    )
                )
            )
            self.assertEqual(len(analysis.verify_sources(audit)["sources"]), 1)
            source.write_text("changed")
            with self.assertRaises(WorkloadValidationError):
                analysis.verify_sources(audit)

    def test_earlier_subset_requires_exactly_40_unique_single_turn_ids(self):
        rows = [dict(row=dict(id=str(index), turn=0)) for index in range(302)]
        identities = {str(index) for index in range(40)}
        self.assertEqual(len(analysis.earlier_rows(rows, identities)), 40)
        for invalid in (rows[1:], [*rows, rows[0]]):
            with self.assertRaises(WorkloadValidationError):
                analysis.earlier_rows(invalid, identities)

    def test_partial_collection_cannot_start_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "gpu-cc302-collection-status.json").write_text(
                json.dumps(dict(status="running", completed=[]))
            )
            with (
                patch.object(analysis.study, "ARTIFACTS", work),
                patch.object(analysis, "lookup") as lookup,
                patch.object(analysis, "native") as native,
            ):
                with self.assertRaises(WorkloadValidationError):
                    analysis.main(
                        [
                            "--work-dir",
                            str(work),
                            "--requests",
                            str(work / "requests"),
                            "--phase",
                            "lookup",
                        ]
                    )
                lookup.assert_not_called()
                native.assert_not_called()

    def test_lookup_validation_failure_cannot_export_partial_outputs(self):
        with (
            patch.object(
                analysis.collector,
                "validate",
                side_effect=WorkloadValidationError("partial"),
            ),
            patch.object(analysis, "export") as export,
        ):
            with self.assertRaises(WorkloadValidationError):
                analysis.lookup(Path("/tmp/work"), Path("/tmp/requests"))
            export.assert_not_called()

    def test_peak_identity_mapping_preserves_ties_and_checks_price(self):
        step = lambda rows, context: 100.0 * rows
        per_row = [(10, [10]), (20, [20])]
        lookup = dict(
            results=[
                dict(
                    data=dataset,
                    minimum_ngram=1,
                    draft_tokens=1,
                    prompt="full",
                    per_row=per_row,
                )
                for dataset in (peak.COMPLETION, peak.EARLIER)
            ]
        )
        priced = peak.completion(per_row, step, 0)
        prices = dict(
            results=[
                dict(
                    minimum_ngram=1,
                    draft_tokens=1,
                    completion=[priced],
                    earlier_40=[priced],
                )
            ]
        )
        audits = {
            dataset: dict(identities=[dict(id="a", turn=0), dict(id="b", turn=0)])
            for dataset in (peak.COMPLETION, peak.EARLIER)
        }
        result = analysis.peak_identities(lookup, prices, audits, step)
        self.assertEqual(
            [row["id"] for row in result["results"][0]["fastest"]], ["a", "b"]
        )
        invalid = copy.deepcopy(prices)
        invalid["results"][0]["completion"][0]["maximum_tps"] += 1
        with self.assertRaises(WorkloadValidationError):
            analysis.peak_identities(lookup, invalid, audits, step)
