import unittest
import simpy
from TokenSim.kv_transfer.native_storage import NativePageStore, QueuedKVLink


class ExternalStorageTest(unittest.TestCase):
    def test_incomplete_write_is_invisible_and_pinned_read_survives_eviction(self):
        store = NativePageStore(200)
        a = store.reserve_write(("a",), 100)
        self.assertIsNone(store.lookup(("a",)))
        store.finish_write(a)
        store.acquire(a)
        b = store.reserve_write(("b",), 100)
        store.finish_write(b)
        c = store.reserve_write(("c",), 100)
        self.assertIsNotNone(c)
        self.assertIs(store.lookup(("a",)), a)
        self.assertIsNone(store.lookup(("b",)))
        store.release(a)
        store.finish_write(c)
        store.validate()

    def test_simple_cpu_counts_packed_stride_while_objects_use_actual_group_bytes(self):
        cpu = NativePageStore(356125000000, allocation_unit=1228800, scope="worker0")
        self.assertEqual(cpu.capacity_bytes // cpu.allocation_unit, 289815)
        page = cpu.reserve_write(("state",), 32768)
        self.assertEqual(page.bytes, 1228800)
        shared = NativePageStore(100000)
        p = shared.reserve_write(("model", 0, "hash"), 32768)
        self.assertEqual(p.bytes, 32768)
        shared.finish_write(p)
        self.assertIs(shared.lookup(("model", 0, "hash")), p)

    def test_all_pinned_storage_refuses_new_write_without_exceeding_budget(self):
        store = NativePageStore(100)
        page = store.reserve_write(("a",), 100)
        self.assertIsNone(store.reserve_write(("b",), 100))
        self.assertFalse(page.ready)
        store.validate()

    def test_directional_link_serializes_competing_callers_and_counts_wait(self):
        env = simpy.Environment()
        link = QueuedKVLink(env, 100, utilization=0.5)
        a = env.process(link.transfer(100))
        b = env.process(link.transfer(100))
        env.run(until=simpy.AllOf(env, [a, b]))
        self.assertEqual(env.now, 4)
        self.assertEqual(link.queue_wait_seconds, 2)
        self.assertEqual(link.bytes, 200)

    def test_utilization_changes_service_without_multiplying_independent_bandwidth(
        self,
    ):
        for utilization, expected in ((0.5, 2), (0.75, 4 / 3), (1, 1)):
            env = simpy.Environment()
            link = QueuedKVLink(env, 100, utilization=utilization)
            env.run(until=env.process(link.transfer(100)))
            self.assertAlmostEqual(env.now, expected)
