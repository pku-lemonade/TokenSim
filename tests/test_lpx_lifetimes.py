"""Logical arena lifetimes follow completion events and include queueing delays."""

import unittest

from TokenSim.errors import ConfigurationError, SimulationStateError
from scripts.lpx.replay_engine import ReplayKernel, ReplayNode


class ReplayLifetimeTest(unittest.TestCase):
    def setUp(self):
        self.kernel = ReplayKernel()
        self.addCleanup(self.kernel.close)

    @staticmethod
    def program():
        return (
            ReplayNode((), ("arena",), "arena", 0, 0, 0, release_after=2),
            ReplayNode((0,), ("compute",), "compute", 0, 1, 1),
            ReplayNode((1,), (), None, 2, 0, 0),
        )

    def test_two_arena_users_serialize_through_the_release_node(self):
        program = self.kernel.add(self.program())
        self.kernel.submit(program, 0)
        self.kernel.submit(program, 0)
        self.kernel.window(0)
        self.assertEqual(self.kernel.advance().end, 3)
        self.assertEqual(self.kernel.advance().end, 6)
        self.assertEqual(self.kernel.stats()["arena"], (6, 3, 2))

    def test_external_contention_extends_the_actual_arena_lifetime(self):
        program = self.kernel.add(self.program())
        blocker = self.kernel.add((ReplayNode((), ("compute",), "compute", 0, 4, 4),))
        first = self.kernel.submit(program, 0)
        self.kernel.submit(blocker, 0)
        later = self.kernel.submit(program, 0)
        self.kernel.window(0)
        done = {}
        while self.kernel.active:
            row = self.kernel.advance()
            done[row.id] = row.end
        self.assertEqual(done[first], 7)
        self.assertEqual(done[later], 10)
        self.assertEqual(self.kernel.stats()["arena"][0], 10)

    def test_release_must_descend_from_acquisition_and_have_zero_service(self):
        for target in (0, 3):
            nodes = list(self.program())
            nodes[0] = ReplayNode(
                (), ("arena",), "arena", 0, 0, 0, release_after=target
            )
            with self.assertRaises(SimulationStateError):
                self.kernel.add(nodes)
        nodes = (
            ReplayNode((), ("arena",), "arena", 0, 0, 0, release_after=1),
            ReplayNode((), (), None, 0, 1, 0),
        )
        with self.assertRaisesRegex(SimulationStateError, "descend"):
            self.kernel.add(nodes)
        for duration, resources in ((1, ("arena",)), (0, ())):
            with self.assertRaises(ConfigurationError):
                ReplayNode((), resources, None, 0, duration, 0, release_after=2)


if __name__ == "__main__":
    unittest.main()
