"""Fragment reuse retains exporter contracts, mutation boundaries and collisions."""

from dataclasses import dataclass
import json
from unittest.mock import patch

from scripts.af_probe_fragments import FragmentEncoder, installed
from TokenSim.af.trace.export import _record_dict
from tests.af.trace import test_export


@dataclass(frozen=True)
class FrozenRecord:
    child: object


class FragmentExportTest(test_export.TraceExportTest):
    def run(self, result=None):
        with installed():
            return super().run(result)

    def test_fragments_reuse_records_and_keep_literal_marker_collisions(self):
        encoder = FragmentEncoder(default=_record_dict)
        record = FrozenRecord((1, 2))
        first = encoder.encode({"record": record})
        self.assertEqual(json.loads(first), {"record": {"child": [1, 2]}})
        with patch.object(
            encoder, "immutable", side_effect=AssertionError("cache miss")
        ):
            self.assertEqual(first, encoder.encode({"record": record}))
        literal = f"~tokensim-json:{id(record)}~"
        for item in (
            {"record": record, "literal": literal},
            {literal: "key", "record": record},
        ):
            self.assertEqual(
                json.loads(encoder.encode(item)),
                json.loads(json.dumps(item, default=_record_dict)),
            )
        nested = FrozenRecord(literal)
        item = {"nested": nested, "record": record}
        self.assertEqual(
            json.loads(encoder.encode(item)),
            json.loads(json.dumps(item, default=_record_dict)),
        )

    def test_mutable_descendants_are_revisited_and_cache_is_bounded(self):
        encoder = FragmentEncoder(default=_record_dict)
        encoder.capacity = 4
        mutable = FrozenRecord({"values": [1]})
        self.assertEqual(
            json.loads(encoder.encode(mutable)), {"child": {"values": [1]}}
        )
        mutable.child["values"].append(2)
        self.assertEqual(
            json.loads(encoder.encode(mutable)), {"child": {"values": [1, 2]}}
        )
        for index in range(100):
            encoder.encode(FrozenRecord(index))
            self.assertLessEqual(len(encoder.memo), 4)
