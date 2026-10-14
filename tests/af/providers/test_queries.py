from __future__ import annotations

import unittest

from TokenSim.af.providers.queries import (
    AFComputeQuery,
    AFEndpoint,
    AFTensorSlice,
    AFTransferQuery,
)
from TokenSim.dtypes import DTYPE_REGISTRY, canonicalize_dtype, dtype_size_bytes


class CanonicalDTypeRegistryTest(unittest.TestCase):
    def test_canonical_names_aliases_and_byte_widths(self):
        cases = {
            "float32": ("fp32", 4),
            "FLOAT16": ("fp16", 2),
            "bfloat16": ("bf16", 2),
            "float8_e4m3fn": ("fp8", 1),
            "float8_e5m2": ("fp8", 1),
            "int8": ("int8", 1),
        }
        for value, (canonical, size_bytes) in cases.items():
            with self.subTest(dtype=value):
                self.assertEqual(canonicalize_dtype(value), canonical)
                self.assertEqual(dtype_size_bytes(value), size_bytes)

        self.assertEqual(
            {name: spec.size_bytes for name, spec in DTYPE_REGISTRY.items()},
            {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1},
        )

    def test_unknown_empty_and_invalid_role_fail_without_a_default(self):
        for value in ("", "mystery"):
            with self.subTest(dtype=value):
                with self.assertRaisesRegex(ValueError, "activation_dtype"):
                    dtype_size_bytes(value, role="activation_dtype")
        with self.assertRaisesRegex(ValueError, "unknown dtype role"):
            canonicalize_dtype("bf16", role="optimizer_dtype")


class AFLatencyQueryContractTest(unittest.TestCase):
    @staticmethod
    def _endpoint(domain: str, rank: int, worker: str, device: str) -> AFEndpoint:
        return AFEndpoint(
            domain=domain,
            domain_rank=rank,
            physical_worker=worker,
            device_type=device,
            dp_rank=0,
            tp_rank=rank,
        )

    def test_legacy_query_construction_keeps_compatible_defaults(self):
        compute = AFComputeQuery(
            iteration_id=1,
            block_id=2,
            layer_id=3,
            phase="decode",
            stage="a",
            token_count=4,
            context_lengths=(8,),
            model_id="model",
            dtype="bf16",
            device_type="a-device",
            worker_id=0,
        )
        transfer = AFTransferQuery(
            iteration_id=1,
            block_id=2,
            layer_id=3,
            phase="decode",
            direction="a_to_f",
            token_count=4,
            hidden_size=16,
            dtype="bf16",
            source_device="a-device",
            target_device="f-device",
            network="af-net",
        )

        self.assertIsNone(compute.layer_type)
        self.assertIsNone(compute.activation_dtype)
        self.assertEqual(compute.tensor_slices, ())
        self.assertIsNone(transfer.communication_kind)
        self.assertIsNone(transfer.logical_stage_id)
        self.assertEqual(transfer.logical_input_bytes_per_rank, ())

    def test_compute_query_preserves_layer_work_owner_layout_and_dtypes(self):
        owner = self._endpoint("a-domain", 1, "a-pool:1", "a-device")
        tensor_slice = AFTensorSlice(
            tensor_item_id="request-7:token-3:routed-1",
            interval=(8, 16),
            layout="partial",
            owner_domain="a-domain",
            owner_domain_rank=1,
            owner_dp_rank=0,
            owner_tp_rank=1,
            owner_physical_worker="a-pool:1",
            partial_id="expert-5",
            element_count=8,
            payload_bytes=16,
        )
        query = AFComputeQuery(
            iteration_id=1,
            block_id=2,
            layer_id=3,
            phase="decode",
            stage="weighted_combine",
            token_count=2,
            context_lengths=(8, 9),
            model_id="model",
            dtype="bf16",
            device_type="a-device",
            worker_id=1,
            layer_type="moe",
            original_token_count=2,
            assignment_count=4,
            input_edge_count=6,
            output_token_count=2,
            weight_dtype="fp8",
            activation_dtype="bf16",
            kv_cache_dtype="bf16",
            a_to_f_dtype="fp8",
            f_to_a_dtype="bf16",
            logical_stage_id="iteration-1:block-2:layer-3:combine",
            tensor_slices=(tensor_slice,),
            owner_endpoint=owner,
        )

        self.assertEqual(query.layer_type, "moe")
        self.assertEqual(
            (
                query.original_token_count,
                query.assignment_count,
                query.input_edge_count,
                query.output_token_count,
            ),
            (2, 4, 6, 2),
        )
        self.assertEqual(query.tensor_slices[0].partial_id, "expert-5")
        self.assertEqual(query.owner_endpoint, owner)
        self.assertEqual(query.a_to_f_dtype, "fp8")

    def test_transfer_query_preserves_leg_endpoints_payload_and_resource(self):
        source = self._endpoint("f-domain", 2, "f-pool:2", "f-device")
        target = self._endpoint("a-domain", 1, "a-pool:1", "a-device")
        tensor_slice = AFTensorSlice(
            tensor_item_id="request-7:token-3:routed-1",
            interval=(8, 16),
            layout="partial",
            owner_domain="a-domain",
            owner_domain_rank=1,
            partial_id="expert-5",
            canonical_producer=None,
            element_count=8,
            payload_bytes=16,
        )
        query = AFTransferQuery(
            iteration_id=1,
            block_id=2,
            layer_id=3,
            phase="decode",
            direction="f_to_a",
            token_count=2,
            hidden_size=16,
            dtype="bf16",
            source_device="f-device",
            target_device="a-device",
            network="af-net",
            operation="af_f_to_a",
            payload_bytes=16,
            communication_kind="endpoint_leg",
            layer_type="moe",
            original_token_count=2,
            assignment_count=4,
            weight_dtype="fp8",
            activation_dtype="bf16",
            kv_cache_dtype="bf16",
            a_to_f_dtype="fp8",
            f_to_a_dtype="bf16",
            logical_stage_id="iteration-1:block-2:layer-3:f-to-a",
            leg_id="f2a:f2:a1:item-1:8-16",
            source_endpoint=source,
            target_endpoint=target,
            participant_endpoints=(source, target),
            tensor_slices=(tensor_slice,),
            source_layout="partial",
            target_layout="sharded",
            fabric="af-fabric",
            resource_group="af-fabric",
            resource_lane="f_to_a",
            logical_input_bytes_per_rank=((2, 16),),
            logical_output_bytes_per_rank=((1, 16),),
        )

        self.assertEqual(query.logical_stage_id, "iteration-1:block-2:layer-3:f-to-a")
        self.assertEqual(query.source_endpoint, source)
        self.assertEqual(query.target_endpoint, target)
        self.assertEqual(query.tensor_slices, (tensor_slice,))
        self.assertEqual(query.resource_lane, "f_to_a")
        self.assertEqual(query.logical_input_bytes_per_rank, ((2, 16),))

    def test_tensor_slice_rejects_an_invalid_global_interval(self):
        with self.assertRaisesRegex(ValueError, "half-open"):
            AFTensorSlice("tensor", (4, 4), "sharded")


if __name__ == "__main__":
    unittest.main()
