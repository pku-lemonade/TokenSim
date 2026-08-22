from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from TokenSim.config.psla_config import PSLAConfig
from TokenSim.errors import WorkloadValidationError
from TokenSim.workload.loaders import load_workload


class WorkloadValidationTest(unittest.TestCase):
    def _model_config(self) -> PSLAConfig:
        return PSLAConfig.from_file("./data/psla/llama-7b.json")

    def _args(self, dataset_path: str, request_count: int) -> argparse.Namespace:
        return argparse.Namespace(
            workload_type="qwen_jsonl",
            dataset_path=dataset_path,
            request_count=request_count,
            dataset_skip_count=0,
            random_seed=0,
        )

    def test_qwen_jsonl_repeated_rows_get_unique_request_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"request_id": 7, "input_length": 32, "output_length": 4, '
                        '"hash_ids": ["a", "b"]}',
                        '{"request_id": 8, "input_length": 32, "output_length": 4, '
                        '"hash_ids": ["a", "c"]}',
                    ]
                )
            )

            workload = load_workload(self._args(str(path), 4), self._model_config())

        self.assertEqual([request.request_id for request in workload], [7, 8, 2, 3])

    def test_missing_qwen_jsonl_dataset_path_raises_workload_error(self):
        args = self._args("", 1)

        with self.assertRaises(WorkloadValidationError):
            load_workload(args, self._model_config())


if __name__ == "__main__":
    unittest.main()
