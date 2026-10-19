"""Serving respects compiled KV and scheduled-query workspace budgets."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import simpy

from TokenSim.af.factory import build_model_runtime
from TokenSim.af.planning.batching import AFBlockBuilder
from TokenSim.af.runtime.admission import ResidentKV
from TokenSim.af.serving.service import AFService
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.context import HardwareContext
from tests.af.support.config import _cluster_dict
from tests.af.support.execution import af_test_hardware
from tests.support.hardware import make_device


def runtime(*, block_tokens=1, f_capacity=0.1, **fields):
    base = af_test_hardware(capacity_gib=0.1)
    model = replace(base.model("TestModel"), vocab_size=32)
    hardware = HardwareContext.in_memory(
        [
            make_device("a-device", memory_gib=0.1),
            make_device("f-device", memory_gib=f_capacity),
        ],
        [model],
        list(base.links),
        list(base.topologies),
    )
    document = _cluster_dict(af_block_size=block_tokens)
    document.update(fields)
    return build_model_runtime(document, hardware=hardware, model=model, block_size=4)


class ServingLimitsTest(unittest.TestCase):
    def test_both_explicit_kv_limits_are_enforced_and_release_restores_budget(self):
        for field in (
            "kv_cache_capacity_tokens_per_dp_rank",
            "kv_cache_capacity_tokens_total",
        ):
            with self.subTest(field=field):
                env = simpy.Environment()
                service = AFService(env, runtime(**{field: 32}))
                first = service.admit(0, 0, maximum_context=32, current_context=31)
                before = service.memory.summary()
                with self.assertRaisesRegex(
                    ConfigurationError, "compiled replica budget"
                ):
                    service.admit(1, 0, maximum_context=32, current_context=31)
                self.assertEqual(service.memory.summary(), before)
                self.assertNotIn(1, service.sessions)
                env.process(service.forward([first]))
                env.run()
                self.assertEqual(first.cached_context, 32)
                service.release(first)
                service.admit(1, 0, maximum_context=32, current_context=31)
                self.assertEqual(len(service.sessions), 1)

    def test_total_kv_limit_is_partitioned_across_a_owners(self):
        base = af_test_hardware(capacity_gib=0.1)
        document = _cluster_dict(attention_workers=2)
        document["kv_cache_capacity_tokens_total"] = 64
        compiled = build_model_runtime(
            document, hardware=base, model=base.model("TestModel"), block_size=4
        )
        service = AFService(simpy.Environment(), compiled)
        for owner in (0, 1):
            service.admit(owner, owner, maximum_context=32, current_context=0)
            with self.assertRaisesRegex(ConfigurationError, "compiled replica budget"):
                service.admit(owner + 10, owner, maximum_context=1, current_context=0)

    def test_windowed_kv_charges_resident_bytes_within_compiled_budget(self):
        base = af_test_hardware(capacity_gib=0.1)
        model = replace(base.model("TestModel"), attention_windows=(4,))
        document = _cluster_dict()
        document["kv_cache_capacity_tokens_per_dp_rank"] = 8
        compiled = build_model_runtime(
            document, hardware=base, model=model, block_size=4
        )
        ledger = ResidentKV(compiled.plan)
        owner = compiled.plan.request_owners[0]
        for request_id in (0, 1):
            ledger.admit(
                request_id,
                owner.domain,
                owner.instance,
                maximum_context=1000,
                current_context=100,
            )
        with self.assertRaisesRegex(ConfigurationError, "compiled replica budget"):
            ledger.admit(
                2,
                owner.domain,
                owner.instance,
                maximum_context=1000,
                current_context=100,
            )

    def test_oversized_request_is_rejected_before_execution_or_ownership_changes(self):
        compiled = runtime(f_capacity=6400 / (1 << 30))
        env = simpy.Environment()
        service = AFService(env, compiled)
        session = service.admit(0, 0, maximum_context=64, current_context=0)
        with self.assertRaisesRegex(ConfigurationError, "workspace token budget"):
            env.run(until=env.process(service.forward([session], query_tokens=64)))
        self.assertFalse(service.trace.events)
        self.assertEqual(service.executor._next_iteration_id, 0)
        self.assertEqual(session.cached_context, 0)
        request = SimpleNamespace(
            id=3, is_prefill=False, context_len=64, scheduled_query_tokens=64
        )
        with self.assertRaisesRegex(ConfigurationError, "workspace token budget"):
            service.executor.submit(0, [request])
        self.assertFalse(getattr(request, "af_a_owners", {}))

    def test_sufficient_compiled_workspace_allows_multi_query_forward(self):
        compiled = runtime(block_tokens=64)
        env = simpy.Environment()
        service = AFService(env, compiled)
        session = service.admit(0, 0, maximum_context=64, current_context=0)
        env.run(until=env.process(service.forward([session], query_tokens=64)))
        self.assertEqual(session.cached_context, 64)
        self.assertEqual(
            [
                event.local_work["token_count"]
                for event in service.trace.events
                if event.stage == "dense_ffn"
            ],
            [64],
        )

    def test_blocks_are_packed_by_scheduled_tokens(self):
        compiled = runtime(block_tokens=8)
        requests = [
            SimpleNamespace(
                id=i, is_prefill=False, context_len=16, scheduled_query_tokens=count
            )
            for i, count in enumerate((3, 5, 2))
        ]
        iteration = AFBlockBuilder().build(
            0, requests, 8, 0, compiled.cluster.af.links[0]
        )
        self.assertEqual([block.token_count for block in iteration.blocks], [8, 2])
        self.assertEqual(
            [block.request_ids for block in iteration.blocks], [(0, 1), (2,)]
        )
