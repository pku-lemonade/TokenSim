"""Af hardware."""

from __future__ import annotations

from TokenSim.hardware.context import HardwareContext
from TokenSim.hardware.links import LinkClass
from tests.support.hardware import make_device, make_links, make_model, make_topology


AF_TEST_DEVICES = (
    "a-device",
    "f-device",
    "attention-device",
    "ffn-device",
    "prefill-device",
    "decode-device",
    "device",
)


AF_TEST_NETWORKS = (
    "profile",
    "af-network-profile",
    "test",
    "test-network",
    "af-net",
    "net1",
)


def af_test_hardware(
    *,
    layers=1,
    capacity_gib=0.01,
    moe=None,
    hidden_size=16,
    intermediate_size=64,
    num_attention_heads=8,
    devices=AF_TEST_DEVICES,
    networks=AF_TEST_NETWORKS,
):
    """In-memory catalog: tiny devices, the ``TestModel`` architecture and AF links."""
    device_specs = [make_device(name, memory_gib=capacity_gib) for name in devices]
    links = list(make_links()) + [
        LinkClass(
            name,
            f"test link {name}",
            100.0e9,
            1.0,
            source_id="test-fixture",
            grade="D",
            half_bandwidth_bytes=1 << 16,
        )
        for name in networks
        if name not in {"nvlink-test", "ethernet-test"}
    ]
    model = make_model(
        "TestModel",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=layers,
        num_attention_heads=num_attention_heads,
        # The legacy roofline stub had no vocabulary entry and the runtime plan
        # assumed 50000; the hand-calculated component bytes rely on it.
        vocab_size=50000,
        moe=moe,
    )
    return HardwareContext.in_memory(device_specs, [model], links, [make_topology()])


def cluster_test_hardware(cluster, **overrides):
    """Catalog whose device and link names come from ``cluster`` itself."""
    devices = tuple(dict.fromkeys(group.hardware for group in cluster.worker_groups))
    networks = tuple(dict.fromkeys(cluster.networks.values()))
    return af_test_hardware(devices=devices, networks=networks, **overrides)
