from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from TokenSim.config.config import ClusterConfig, KVTransferConfig, infer_kv_role
from TokenSim.errors import ConfigurationError
from TokenSim.kv_transfer import (
    ConnectorStats,
    KVConnectorFactory,
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
    MooncakeConnector,
    MooncakeStoreConnector,
    NoopConnector,
    P2PConnector,
)


class KVTransferConfigTest(unittest.TestCase):
    def test_default_config_uses_noop_connector(self):
        config = KVTransferConfig.default()

        self.assertEqual(config.kv_connector, "NoopConnector")
        self.assertIsNotNone(config.engine_id)

    def test_cluster_config_parses_embedded_kv_transfer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cluster.json"
            path.write_text(
                """
                {
                    "num_workers": 1,
                    "networks": {"netw1": "ethernet100Gb"},
                    "kv_transfer": {"kv_connector": "P2PConnector"},
                    "worker_groups": [
                        {
                            "role": "hybrid",
                            "hardware": "A100-40G",
                            "num_workers": 1,
                            "network": "netw1"
                        }
                    ]
                }
                """
            )

            cluster = ClusterConfig.from_file(path)

        self.assertEqual(cluster.effective_kv_transfer().kv_connector, "P2PConnector")

    def test_override_takes_precedence_over_cluster_config(self):
        cluster = ClusterConfig(
            num_workers=1,
            networks={"netw1": "ethernet100Gb"},
            kv_transfer=KVTransferConfig(kv_connector="P2PConnector"),
        )
        override = KVTransferConfig(kv_connector="NoopConnector")

        self.assertIs(cluster.effective_kv_transfer(override), override)

    def test_invalid_role_raises_configuration_error(self):
        with self.assertRaises(ConfigurationError):
            KVTransferConfig(kv_connector="NoopConnector", kv_role="bad-role")

    def test_worker_role_inference(self):
        self.assertEqual(infer_kv_role("prefill"), "kv_producer")
        self.assertEqual(infer_kv_role("decode"), "kv_consumer")
        self.assertEqual(infer_kv_role("hybrid"), "kv_both")


class KVConnectorFactoryTest(unittest.TestCase):
    def test_factory_creates_noop_and_p2p(self):
        noop = KVConnectorFactory.create_connector(KVTransferConfig.default())
        p2p = KVConnectorFactory.create_connector(
            KVTransferConfig(kv_connector="P2PConnector")
        )

        self.assertIsInstance(noop, NoopConnector)
        self.assertIsInstance(p2p, P2PConnector)

    def test_unknown_connector_raises_configuration_error(self):
        with self.assertRaises(ConfigurationError):
            KVConnectorFactory.create_connector(
                KVTransferConfig(kv_connector="MissingConnector")
            )

    def test_mooncake_connectors_are_implemented(self):
        mooncake = KVConnectorFactory.create_connector(
            KVTransferConfig(kv_connector="MooncakeConnector")
        )
        store = KVConnectorFactory.create_connector(
            KVTransferConfig(kv_connector="MooncakeStoreConnector")
        )

        self.assertIsInstance(mooncake, MooncakeConnector)
        self.assertIsInstance(store, MooncakeStoreConnector)

    def test_worker_metadata_aggregates_sets_and_events(self):
        left = KVConnectorWorkerMetadata(
            finished_sending={1},
            finished_recving={2},
            events=["a"],
        )
        right = KVConnectorWorkerMetadata(
            finished_sending={3},
            finished_recving={4},
            events=["b"],
        )

        combined = left.aggregate(right)

        self.assertEqual(combined.finished_sending, {1, 3})
        self.assertEqual(combined.finished_recving, {2, 4})
        self.assertEqual(combined.events, ["a", "b"])

    def test_noop_connector_has_zero_transfer_stats(self):
        connector = KVConnectorFactory.create_connector(KVTransferConfig.default())
        meta = connector.build_connector_meta(object())
        connector.bind_connector_metadata(meta)

        self.assertIsInstance(meta, KVConnectorMetadata)
        self.assertEqual(connector.start_load_kv(), 0)
        self.assertEqual(connector.wait_for_save(), 0)
        self.assertEqual(connector.stats.transfer_count, 0)

    def test_connector_stats_aggregate_and_dict(self):
        left = ConnectorStats(transfer_count=1, transfer_blocks=2)
        right = ConnectorStats(transfer_count=3, transfer_bytes=4)

        combined = left.aggregate(right)

        self.assertEqual(combined.transfer_count, 4)
        self.assertEqual(combined.transfer_blocks, 2)
        self.assertEqual(combined.transfer_bytes, 4)
        self.assertEqual(
            combined.as_dict()["connector_transfer_count"],
            4,
        )


if __name__ == "__main__":
    unittest.main()
