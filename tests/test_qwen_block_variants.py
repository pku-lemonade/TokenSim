from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_qwen_block_variants import (
    CompositeIdMap,
    TraceConversionError,
    convert_hash_ids,
    convert_trace,
    validate_variant,
)


class QwenBlockVariantTest(unittest.TestCase):
    def test_128_block_uses_first_six_hashes(self):
        composite_ids = CompositeIdMap()
        first = convert_hash_ids(list(range(8)), 16, 128, 0.75, composite_ids)
        same_prefix = convert_hash_ids(
            [*range(6), 100, 101], 16, 128, 0.75, composite_ids
        )
        different_prefix = convert_hash_ids(
            [*range(5), 99, 100, 101], 16, 128, 0.75, composite_ids
        )

        self.assertEqual(first, same_prefix)
        self.assertNotEqual(first, different_prefix)

    def test_512_block_uses_first_twenty_four_hashes(self):
        composite_ids = CompositeIdMap()
        first = convert_hash_ids(list(range(32)), 16, 512, 0.75, composite_ids)
        same_prefix = convert_hash_ids(
            [*range(24), *range(100, 108)], 16, 512, 0.75, composite_ids
        )

        self.assertEqual(first, same_prefix)

    def test_partial_group_uses_all_available_hashes(self):
        composite_ids = CompositeIdMap()
        first = convert_hash_ids([1, 2, 3], 16, 128, 0.75, composite_ids)
        different_tail = convert_hash_ids([1, 2, 4], 16, 128, 0.75, composite_ids)

        self.assertNotEqual(first, different_tail)

    def test_conversion_preserves_fields_and_hash_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "trace_blksz_16.jsonl"
            output_path = Path(tmpdir) / "trace_blksz_128.jsonl"
            source_row = {
                "chat_id": 7,
                "timestamp": 1.25,
                "input_length": 129,
                "output_length": 4,
                "type": "text",
                "turn": 2,
                "hash_ids": list(range(9)),
            }
            source_path.write_text(json.dumps(source_row) + "\n", encoding="utf-8")

            stats = convert_trace(source_path, output_path)
            converted_row = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(stats.record_count, 1)
        self.assertEqual(len(converted_row["hash_ids"]), 2)
        converted_row.pop("hash_ids")
        source_row.pop("hash_ids")
        self.assertEqual(converted_row, source_row)

    def test_conversion_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "trace_blksz_16.jsonl"
            output_path = Path(tmpdir) / "trace_blksz_128.jsonl"
            source_path.write_text(
                json.dumps(
                    {
                        "input_length": 16,
                        "output_length": 1,
                        "hash_ids": [1],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_path.write_text("existing\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                convert_trace(source_path, output_path)

    def test_conversion_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "trace_blksz_16.jsonl"
            first_output = Path(tmpdir) / "first_blksz_128.jsonl"
            second_output = Path(tmpdir) / "second_blksz_128.jsonl"
            rows = [
                {
                    "input_length": 128,
                    "output_length": 1,
                    "hash_ids": list(range(8)),
                },
                {
                    "input_length": 144,
                    "output_length": 2,
                    "hash_ids": [*range(6), 100, 101, 102],
                },
            ]
            source_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            convert_trace(source_path, first_output)
            convert_trace(source_path, second_output)
            first_bytes = first_output.read_bytes()
            second_bytes = second_output.read_bytes()

        self.assertEqual(first_bytes, second_bytes)

    def test_validation_reproduces_six_of_eight_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "trace_blksz_16.jsonl"
            output_path = Path(tmpdir) / "trace_blksz_128.jsonl"
            rows = [
                {
                    "input_length": 128,
                    "output_length": 1,
                    "hash_ids": list(range(8)),
                },
                {
                    "input_length": 128,
                    "output_length": 1,
                    "hash_ids": [*range(6), 100, 101],
                },
            ]
            source_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            convert_trace(source_path, output_path)

            stats = validate_variant(source_path, output_path)

        self.assertEqual(stats.source_hit_tokens, 6 * 16)
        self.assertEqual(stats.variant_hit_tokens, 128)
        self.assertEqual(stats.ideal_threshold_hit_tokens, 128)
        self.assertEqual(stats.realization_gap_tokens, 0)

    def test_validation_reports_unrepresentable_partial_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "trace_blksz_16.jsonl"
            output_path = Path(tmpdir) / "trace_blksz_128.jsonl"
            rows = [
                {
                    "input_length": 96,
                    "output_length": 1,
                    "hash_ids": list(range(6)),
                },
                {
                    "input_length": 128,
                    "output_length": 1,
                    "hash_ids": [*range(6), 100, 101],
                },
            ]
            source_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            convert_trace(source_path, output_path)

            stats = validate_variant(source_path, output_path)

        self.assertEqual(stats.ideal_threshold_hit_tokens, 128)
        self.assertEqual(stats.variant_hit_tokens, 0)
        self.assertEqual(stats.realization_gap_tokens, -128)

    def test_rejects_incorrect_source_hash_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "trace_blksz_16.jsonl"
            output_path = Path(tmpdir) / "trace_blksz_128.jsonl"
            source_path.write_text(
                json.dumps(
                    {
                        "input_length": 17,
                        "output_length": 1,
                        "hash_ids": [1],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(TraceConversionError):
                convert_trace(source_path, output_path)


if __name__ == "__main__":
    unittest.main()
