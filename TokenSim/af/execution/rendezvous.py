"""Native multi-A transaction identities and shared failure propagation."""

from collections import Counter

from TokenSim.errors import ConfigurationError, SimulationStateError


class NativeRendezvous:
    def __init__(self, executor):
        config = executor.cluster_config
        self.peers = config.af_domains[0].parallel.data_parallel_size
        self.enabled = config.af.execution_protocol == "vllm_afd" and self.peers > 1
        self.failure = executor.env.event()
        self.sequences = Counter()
        self.iterations = {}
        self.rounds = {}
        self.env = executor.env
        self.changed = executor.env.event()
        self.completed_round = -1

    def check(self):
        if self.enabled and self.failure.triggered:
            raise self.failure.value

    def register(self, iteration):
        if not self.enabled:
            return
        self.check()
        owner = iteration.blocks[0].placement.a_dp_rank
        round_id = self.sequences[owner]
        phases = tuple(block.forward_phase for block in iteration.blocks)
        transaction = self.rounds.setdefault(
            round_id,
            {"phases": phases, "finished": set(), "stage": 0, "stage_peers": set()},
        )
        if transaction["phases"] != phases:
            error = ConfigurationError(
                "native AF peers require matching microbatch counts and phases "
                f"in submission round {round_id}"
            )
            self.abort(error)
            raise error
        self.iterations[iteration.iteration_id] = (round_id, owner)
        self.sequences[owner] += 1

    def transaction(self, block):
        self.check()
        return (self.iterations[block.iteration_id][0], block.block_id)

    def wait_turn(self, block, layer, stage):
        round_id, block_id = self.transaction(block)
        transaction = self.rounds[round_id]
        ordinal = layer * len(transaction["phases"]) + block_id
        while (
            round_id > self.completed_round + 1
            or stage == "router"
            and ordinal != transaction["stage"]
        ):
            yield self.changed
            self.check()

    def finish_stage(self, block, layer):
        if not self.enabled:
            return
        self.check()
        round_id, block_id = self.transaction(block)
        transaction = self.rounds[round_id]
        ordinal = layer * len(transaction["phases"]) + block_id
        if (
            ordinal != transaction["stage"]
            or block.placement.a_dp_rank in transaction["stage_peers"]
        ):
            raise SimulationStateError(
                "native AF stage completed out of transaction order"
            )
        transaction["stage_peers"].add(block.placement.a_dp_rank)
        if len(transaction["stage_peers"]) == self.peers:
            transaction["stage"] += 1
            transaction["stage_peers"].clear()
            self._wake()

    def _wake(self):
        self.changed.succeed()
        self.changed = self.env.event()

    def complete(self, iteration):
        entry = self.iterations.pop(iteration.iteration_id, None)
        if entry is None:
            return
        round_id, owner = entry
        transaction = self.rounds[round_id]
        transaction["finished"].add(owner)
        if len(transaction["finished"]) == self.peers:
            del self.rounds[round_id]
            self.completed_round = round_id
            self._wake()

    def abort(self, reason):
        if self.enabled and not self.failure.triggered:
            self.failure.succeed(
                SimulationStateError(
                    f"native AF peer transaction aborted: {reason}; "
                    "create a new executor before retrying"
                )
            )
            self.iterations.clear()
            self.rounds.clear()
