"""In-memory hardware/model catalogs shared by the test suite.

Tests used to fake the legacy roofline object with ad-hoc namespaces; the
new catalog is built from real dataclasses so that CacheConfig, the operator
backend and the communication model exercise the production code paths.
"""

from __future__ import annotations

from TokenSim.config.model_config import ModelSpec
from TokenSim.hardware.context import HardwareContext
from TokenSim.hardware.device import DeviceSpec
from TokenSim.hardware.links import LinkClass
from TokenSim.hardware.topology import TopologyLevel, TopologySpec
from TokenSim.moe.config import MoEModelConfig

GIB = 1 << 30


def test_links() -> list[LinkClass]:
    return [
        LinkClass("nvlink-test", "test NVLink", 100.0e9, 1.0, source_id="test-fixture", grade="D", half_bandwidth_bytes=1 << 16),
        LinkClass("ethernet-test", "test Ethernet", 10.0e9, 20.0, source_id="test-fixture", grade="D", half_bandwidth_bytes=1 << 16),
    ]


def test_topology(node_size: int = 8) -> TopologySpec:
    return TopologySpec(
        topology_id="test-topology",
        levels=(
            TopologyLevel("node", node_size, "nvlink-test", fabric="switched"),
            TopologyLevel("cluster", None, "ethernet-test", fabric="fat_tree"),
        ),
        source_id="test-fixture",
    )


def test_device(
    device_id: str = "TestGPU",
    *,
    memory_gib: float = 80.0,
    peak_flops: float = 100e12,
    bandwidth: float = 1.0e12,
    family: str = "nvidia_gpu",
    aliases: tuple[str, ...] = (),
) -> DeviceSpec:
    return DeviceSpec.simple(
        device_id,
        family=family,
        peak_flops=peak_flops,
        memory_capacity_bytes=memory_gib * GIB,
        memory_bandwidth_bytes_per_s=bandwidth,
        scale_up_link="nvlink-test",
        host_link="ethernet-test",
        aliases=aliases,
    )


def test_model(
    model_id: str = "TestModel",
    *,
    hidden_size: int = 128,
    intermediate_size: int = 256,
    num_layers: int = 4,
    num_attention_heads: int = 8,
    num_key_value_heads: int | None = None,
    vocab_size: int = 1024,
    moe: MoEModelConfig | None = None,
) -> ModelSpec:
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads
    return ModelSpec(
        model_id=model_id,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=hidden_size // num_attention_heads,
        vocab_size=vocab_size,
        activation="swiglu",
        dtype="fp16",
        kv_cache_dtype="fp16",
        moe=moe or MoEModelConfig(),
    )


def moe_test_model() -> ModelSpec:
    return test_model(
        "MoEModel",
        moe=MoEModelConfig(
            is_moe_model=True,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            num_shared_experts=1,
            num_moe_layers=2,
            first_k_dense_replace=1,
            moe_layer_freq=1,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=8,
            num_key_value_heads=8,
        ),
    )


def test_hardware(*devices: DeviceSpec, models: list[ModelSpec] | None = None) -> HardwareContext:
    devices = devices or (test_device(),)
    models = models or [test_model(), moe_test_model()]
    return HardwareContext.in_memory(devices, models, test_links(), [test_topology()])
