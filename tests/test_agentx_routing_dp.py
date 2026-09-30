from __future__ import annotations

import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import simpy

from TokenSim.config.config import ClusterConfig, ParallelConfig, WorkerGroupConfig
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request, reset_g_time, RequestStatus
from TokenSim.placement.consistent_hash import ConsistentHashRouter, fbi_hash
from tests.test_parallel_simulation import _hardware, _psla


def engine(env, *, sync_interval=16, routing="none"):
    parallel = ParallelConfig(data_parallel_size=2)
    cluster = ClusterConfig(
        num_workers=2,
        networks={"node0": "ethernet-test"},
        worker_groups=[
            WorkerGroupConfig(
                role="hybrid", hardware="TestGPU", num_workers=2, network="node0"
            )
        ],
    )
    return LLMEngine(
        env=env,
        block_size=16,
        batching="paged-attn",
        kv_transfer_config=cluster.effective_kv_transfer(),
        psla_config=_psla(),
        cluster_config=cluster,
        parallel_config=parallel,
        hardware=_hardware(),
        prefill_worker_pool_type="round_robin",
        decode_worker_pool_type="round_robin",
        max_parallem_sum=8,
        max_num_batched_tokens=16,
        latency_backend_type="analytical",
        router_policy=routing,
        dp_lockstep=True,
        dp_sync_interval=sync_interval,
    )


def req(env, identifier, rank=0, output=1, tokens=16):
    r = Request(identifier, tokens, output, 16, chat_id=f"conversation-{identifier}")
    r.dp_rank = rank
    r.arrive(env)
    return r


class RouterTest(unittest.TestCase):
    def router(self, n=8):
        return ConsistentHashRouter(
            [SimpleNamespace(id=i, dp_rank=i) for i in range(n)]
        )

    def test_session_is_stable_and_subagent_has_its_own_key(self):
        router = self.router()
        parent = SimpleNamespace(chat_id="parent", parent_chat_id=None)
        target = router.select(parent)
        self.assertIs(router.select(parent), target)
        for n in range(100):
            child = SimpleNamespace(chat_id=f"child-{n}", parent_chat_id="parent")
            if router.select(child) is not target:
                break
        else:
            self.fail("independent child sessions never choose another rank")
        self.assertEqual(
            router.sessions[child.chat_id],
            router.rank_for_key(f"header:x-session-id:{child.chat_id}"),
        )

    def test_hashing_is_wrapped_utf8_and_worker_order_independent(self):
        router = self.router()
        reversed_router = ConsistentHashRouter(
            list(reversed(list(router.workers.values())))
        )
        for key in (
            "",
            "header:x-session-id:abc",
            "header:x-session-id:中文",
            "x" * 1000,
        ):
            self.assertEqual(
                router.rank_for_key(key), reversed_router.rank_for_key(key)
            )
            self.assertTrue(0 <= fbi_hash(key) < 2**64)

    def test_bulk_requests_are_routed_individually_instead_of_using_first_session(self):
        env = simpy.Environment()
        e = engine(env, routing="consistent_hash")
        requests = [req(env, i) for i in range(40)]
        e.add_requests_burst(requests)
        for r in requests:
            self.assertEqual(
                r.dp_rank, e.router.rank_for_key(f"header:x-session-id:{r.chat_id}")
            )
            self.assertEqual(r.owner_worker_ids, {r.routing_worker_id})
        self.assertEqual(set(r.dp_rank for r in requests), {0, 1})

    def test_reset_keeps_affinity_but_clears_profile_request_counts(self):
        router = self.router()
        r = SimpleNamespace(chat_id="a")
        rank = router.select(r).dp_rank
        router.reset_stats()
        self.assertFalse(router.request_counts)
        self.assertEqual(router.select(r).dp_rank, rank)

    def test_saved_observed_keys_cover_all_reference_worker_ranks(self):
        evidence = json.loads(
            Path("data/reference/agentx-router/validation.json").read_text()
        )
        router = self.router()
        self.assertEqual(
            {v["expected_rank"] for v in evidence["observed_vectors"]}, set(range(8))
        )
        for vector in evidence["observed_vectors"]:
            self.assertEqual(
                router.rank_for_key(vector["key"]), vector["expected_rank"]
            )


class DPStepTest(unittest.TestCase):
    def setUp(self):
        reset_g_time()

    def test_two_real_ranks_finish_at_maximum_latency_not_sum(self):
        env = simpy.Environment()
        e = engine(env)
        e.workers[0].dynamic_batch = lambda _: 1
        e.workers[1].dynamic_batch = lambda _: 2
        a, b = req(env, 0, 0), req(env, 1, 1)
        e.add_requests_burst([a])
        e.add_requests_burst([b])
        env.run(until=simpy.AllOf(env, [a.completion_event, b.completion_event]))
        e.raise_if_failed()
        self.assertEqual(a.completed_at, 2)
        self.assertEqual(b.completed_at, 2)
        self.assertEqual(e.dp_coordinator.stats.elapsed_step_seconds, 2)
        self.assertEqual(e.dp_coordinator.stats.synchronization_wait_seconds[0], 1)

    def test_empty_rank_runs_dummy_and_its_cost_participates(self):
        env = simpy.Environment()
        e = engine(env)
        e.workers[0].dynamic_batch = lambda _: 1
        e.workers[1].dynamic_batch = lambda _: 3
        a = req(env, 0)
        e.add_requests_burst([a])
        env.run(until=a.completion_event)
        e.raise_if_failed()
        self.assertEqual(a.completed_at, 3)
        self.assertEqual(e.dp_coordinator.stats.dummy_batches[1], 1)
        self.assertFalse(e.workers[1].scheduler.running)
        self.assertFalse(e.workers[1].scheduler.waiting)

    def test_finish_consensus_on_step_one_and_interval_resets_wave_counter(self):
        env = simpy.Environment()
        e = engine(env, sync_interval=4)
        for w in e.workers:
            w.dynamic_batch = lambda batch: 0.1 if batch[0].id < 0 else 1
        a = req(env, 0, output=3)
        e.add_requests_burst([a])
        env.run(until=3.11)
        e.raise_if_failed()
        c = e.dp_coordinator
        self.assertTrue(a.is_done)
        self.assertEqual(c.stats.steps, 4)
        self.assertEqual(c.stats.finish_consensus_checks, 2)
        self.assertEqual(c.stats.waves_finished, 1)
        self.assertEqual(c.step_counter, 0)
        self.assertEqual([w.dp_step_counter for w in e.workers], [0, 0])
        self.assertEqual(c.stats.dummy_batches[0], 1)

    def test_completion_callback_can_issue_next_request_before_next_step(self):
        env = simpy.Environment()
        e = engine(env)
        for w in e.workers:
            w.dynamic_batch = lambda batch: 0.1 if batch[0].id < 0 else 1
        a = req(env, 0)
        b = req(env, 1)
        a.completion_event.callbacks.append(lambda _: e.add_requests_burst([b]))
        e.add_requests_burst([a])
        env.run(until=b.completion_event)
        e.raise_if_failed()
        self.assertEqual(b.completed_at, 2)
        self.assertEqual(e.dp_coordinator.stats.real_batches[0], 2)

    def test_cancel_during_global_kernel_holds_blocks_until_boundary(self):
        env = simpy.Environment()
        e = engine(env)
        for w in e.workers:
            w.dynamic_batch = lambda _: 5
        a = req(env, 0, output=10, tokens=64)
        e.add_requests_burst([a])
        env.run(until=1)
        manager = e.workers[0].scheduler.block_manager
        self.assertTrue(manager.block_table.get_blocks(a.id))
        ack = e.cancel_requests([a])[0]
        env.run(until=2)
        self.assertFalse(ack.triggered)
        self.assertTrue(manager.block_table.get_blocks(a.id))
        env.run(until=ack)
        e.raise_if_failed()
        self.assertEqual(env.now, 5)
        self.assertEqual(a.generation_idx, 0)
        self.assertEqual(a.status, RequestStatus.FINISHED_CANCELLED)
        self.assertFalse(manager.block_table.get_blocks(a.id))

    def test_arrival_during_kernel_waits_until_next_global_step(self):
        env = simpy.Environment()
        e = engine(env)
        e.workers[0].dynamic_batch = lambda batch: 0.1 if batch[0].id < 0 else 2
        e.workers[1].dynamic_batch = lambda batch: 0.1 if batch[0].id < 0 else 0.5
        a = req(env, 0)
        e.add_requests_burst([a])
        env.run(until=0.5)
        b = req(env, 1, rank=1)
        e.add_requests_burst([b])
        env.run(until=b.completion_event)
        e.raise_if_failed()
        self.assertEqual(b.completed_at, 2.5)

    def test_resetting_stats_does_not_reset_running_wave_or_step_cadence(self):
        env = simpy.Environment()
        e = engine(env)
        for w in e.workers:
            w.dynamic_batch = lambda _: 1
        a = req(env, 0, output=3)
        e.add_requests_burst([a])
        env.run(until=1.1)
        counter = e.dp_coordinator.step_counter
        e.reset_profile_stats()
        self.assertEqual(e.dp_coordinator.step_counter, counter)
        self.assertTrue(e.dp_coordinator.wave_running)
        self.assertEqual(e.dp_coordinator.stats.steps, 0)


if __name__ == "__main__":
    unittest.main()
