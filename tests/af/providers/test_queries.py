"""Compute/transfer query validation and precision contracts."""

from __future__ import annotations

from dataclasses import replace
import unittest

import simpy

from TokenSim.af.execution.executor import AFExecutor
from TokenSim.af.planning.batching import AFBlock
from TokenSim.af.planning.nodes import ExecutionNode, RankLocalWork
from TokenSim.af.providers.queries import (
    AFComputeQuery,
    AFEndpoint,
    AFServiceEstimate,
    AFTensorSlice,
    AFTransferQuery,
)
from TokenSim.dtypes import DTYPE_REGISTRY, canonicalize_dtype, dtype_size_bytes
from TokenSim.errors import ConfigurationError
from tests.af.support.config import _cluster
from tests.af.support.execution import af_test_resources
from tests.af.support.model import _moe, _runtime_plan


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


class RecordingComputeProvider:
    def __init__(self, *, rejection: str | None = None) -> None:
        self.rejection = rejection
        self.validated = []
        self.queries = []

    def validate_effective_dtypes(self, model_id, effective_dtypes) -> None:
        self.validated.append((model_id, effective_dtypes))
        if self.rejection is not None:
            raise ValueError(self.rejection)

    def estimate(self, query) -> AFServiceEstimate:
        self.queries.append(query)
        return AFServiceEstimate(0.0, "dtype-contract:compute")


class RecordingTransferProvider:
    def __init__(self) -> None:
        self.queries = []

    def estimate(self, query) -> AFServiceEstimate:
        self.queries.append(query)
        return AFServiceEstimate(
            0.0,
            "dtype-contract:transfer",
            payload_bytes=query.payload_bytes,
        )


def _typed_runtime():
    cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
    cluster.af = replace(
        cluster.af,
        weight_dtype="fp8",
        activation_dtype="bf16",
        kv_cache_dtype="fp32",
    )
    cluster.af.links = [
        replace(
            cluster.af.links[0],
            a_to_f_dtype="fp8",
            f_to_a_dtype="fp32",
        )
    ]
    moe_config = _moe(top_k=1, shared=1)
    plan = _runtime_plan(cluster, moe_config=moe_config, dtype="fp16")
    return cluster, moe_config, plan


def _executor(env, cluster, moe_config, plan, compute, transfer):
    return AFExecutor(
        env=env,
        cluster_config=cluster,
        worker_configs=cluster.workers(),
        compute_provider=compute,
        transfer_provider=transfer,
        model_id="TestModel",
        # This compatibility input intentionally differs from the runtime plan.
        dtype="fp16",
        total_layers=plan.total_layers,
        hidden_size=plan.hidden_size,
        moe_config=moe_config,
        runtime_plan=plan,
        resources=af_test_resources(env, cluster, cluster.workers()),
    )


class AFExecutorDTypeContractTest(unittest.TestCase):
    def test_effective_roles_cross_executor_provider_boundary(self):
        env = simpy.Environment()
        cluster, moe_config, plan = _typed_runtime()
        compute = RecordingComputeProvider()
        transfer = RecordingTransferProvider()
        executor = _executor(
            env, cluster, moe_config, SimpleRuntimePlanView(plan), compute, transfer
        )

        self.assertEqual(executor.dtype, "bf16")
        self.assertEqual(len(compute.validated), 1)
        model_id, validated = compute.validated[0]
        self.assertEqual(model_id, "TestModel")
        self.assertEqual(validated.weight_dtype, "fp8")
        self.assertEqual(validated.activation_dtype, "bf16")
        self.assertEqual(validated.kv_cache_dtype, "fp32")

        block = AFBlock(
            iteration_id=0,
            block_id=0,
            request_ids=(7,),
            request_count=1,
            token_count=1,
            context_lengths=(9,),
            source_a_worker_id=0,
            link_name="af-link",
        )
        a_binding = plan.binding("a-domain", 0)
        f_binding = plan.binding("f-domain", 0)
        # Runtime-node validity is covered by runtime tests. This focused test
        # drives only the executor-to-provider query boundary.
        compute_node = ExecutionNode(
            node_id="attention",
            operation="attention",
            kind="compute",
            predecessor_ids=(),
            stage="attention",
            direction=None,
            source_domain="a-domain",
            target_domain="a-domain",
            logical_participants=(a_binding.instance,),
            domain_ranks=(a_binding.domain_rank,),
            physical_workers=(a_binding.physical_identity,),
            dp_rank=0,
            tp_rank=0,
            local_work=RankLocalWork(
                token_count=1,
                original_token_count=1,
                output_token_count=1,
            ),
            layer_type="dense",
        )
        env.process(executor._run_planned_compute(block, 0, compute_node))
        for direction, source, target in (
            ("a_to_f", a_binding, f_binding),
            ("f_to_a", f_binding, a_binding),
        ):
            transfer_node = ExecutionNode(
                node_id=direction,
                operation=f"af_{direction}",
                kind="communication",
                predecessor_ids=(),
                stage=None,
                direction=direction,
                source_domain=source.domain,
                target_domain=target.domain,
                logical_participants=(source.instance, target.instance),
                domain_ranks=(source.domain_rank, target.domain_rank),
                physical_workers=(
                    source.physical_identity,
                    target.physical_identity,
                ),
                dp_rank=0,
                network="af-net",
                resource_group="af-fabric",
                resource_lane=direction,
                payload_bytes=16,
                layer_type="dense",
            )
            env.process(
                executor._run_planned_communication(
                    block,
                    0,
                    cluster.af.links[0],
                    transfer_node,
                )
            )
        env.run()

        self.assertTrue(compute.queries)
        for query in compute.queries:
            self.assertEqual(query.dtype, "bf16")
            self.assertEqual(query.weight_dtype, "fp8")
            self.assertEqual(query.activation_dtype, "bf16")
            self.assertEqual(query.kv_cache_dtype, "fp32")
            self.assertEqual(query.a_to_f_dtype, "fp8")
            self.assertEqual(query.f_to_a_dtype, "fp32")

        directional = {
            direction: [
                query for query in transfer.queries if query.direction == direction
            ]
            for direction in ("a_to_f", "f_to_a")
        }
        self.assertTrue(directional["a_to_f"])
        self.assertTrue(directional["f_to_a"])
        self.assertTrue(all(query.dtype == "fp8" for query in directional["a_to_f"]))
        self.assertTrue(all(query.dtype == "fp32" for query in directional["f_to_a"]))
        for query in transfer.queries:
            self.assertEqual(query.activation_dtype, "bf16")
            self.assertEqual(query.a_to_f_dtype, "fp8")
            self.assertEqual(query.f_to_a_dtype, "fp32")

    def test_manifest_mismatch_fails_before_event_loop_or_query(self):
        env = simpy.Environment()
        cluster, moe_config, plan = _typed_runtime()
        compute = RecordingComputeProvider(
            rejection=(
                "operator manifest dtype mismatch: model='TestModel', "
                "role='activation_dtype', effective='bf16', manifest='fp16'"
            )
        )
        transfer = RecordingTransferProvider()

        with self.assertRaisesRegex(
            ConfigurationError,
            "provider dtype contract validation failed.*activation_dtype",
        ):
            _executor(env, cluster, moe_config, plan, compute, transfer)

        self.assertEqual(env.now, 0)
        self.assertEqual(compute.queries, [])
        self.assertEqual(transfer.queries, [])


class SimpleRuntimePlanView:
    def __init__(self, plan) -> None:
        self.total_layers = plan.total_layers
        self.hidden_size = plan.hidden_size
        self.effective_dtypes = plan.effective_dtypes
        self.link_effective_dtypes = plan.link_effective_dtypes

    def dtypes_for_link(self, link_name):
        for name, effective in self.link_effective_dtypes:
            if name == link_name:
                return effective
        return self.effective_dtypes

    @staticmethod
    def validate_execution_node(_node) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
