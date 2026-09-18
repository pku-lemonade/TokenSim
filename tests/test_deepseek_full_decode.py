import json
import unittest
from dataclasses import fields

from scripts.run_deepseek_mla import ROOT, configuration
from scripts.run_deepseek_v3_full import (
    RecordingLookup, UniformRequestRouting, boundary_shapes, simulate_full, validate_full_trace,
)
from TokenSim.operator_data import HardwareProfile, estimate_operator
from TokenSim.errors import ConfigurationError


HW = json.loads((ROOT/'data/operator_data/deepseek-v3-theoretical.json').read_text())
COMM = json.loads((ROOT/'data/operator_data/af-communication-separated.json').read_text())


class FullDecodeTest(unittest.TestCase):
    def test_multi_a_routes_weights_to_every_combine_owner_and_runs_parallel(self):
        for a_tp in (2, 4, 8):
            with self.subTest(a_tp=a_tp):
                config = configuration(False, a_tp, 2, 2, HW, router_location='f',
                                       communication_timing=COMM['host'])
                executor, iterations = simulate_full(config, RecordingLookup(HW, COMM),
                                                     batch=4, context=32, steps=2)
                for iteration in iterations:
                    for block in iteration.blocks:
                        events = [e for e in executor.trace.events if e.iteration_id == iteration.iteration_id
                                  and e.block_id == block.block_id and e.layer_id == 0]
                        attention = [e for e in events if e.stage == 'attention']
                        self.assertEqual(len(attention), a_tp)
                        self.assertEqual(len({e.start_at for e in attention}), 1)
                        self.assertGreater(min(e.end_at for e in attention), max(e.start_at for e in attention))
                        metadata = {e.target_endpoint.domain_rank:e for e in events
                                    if e.operation == 'af_router_metadata' and e.target_domain == 'a-domain'}
                        self.assertEqual(set(metadata), set(range(a_tp)))
                        for combine in (e for e in events if e.stage == 'weighted_combine'):
                            arrival = metadata[combine.participant_ranks[0]]
                            self.assertGreaterEqual(combine.start_at, arrival.end_at)
                            self.assertEqual(arrival.payload_bytes, block.token_count*8*8)
                        boundary = [e for e in executor.boundary_events if e['iteration_id'] == iteration.iteration_id
                                    and e['block_id'] == block.block_id]
                        self.assertEqual(len(boundary), 4*a_tp)
                        self.assertTrue(all(e['key']['n'] == 129280//a_tp for e in boundary if e['name']=='lm_head'))
                        comm = [e for e in executor.trace.events if e.iteration_id == iteration.iteration_id
                                and e.block_id == block.block_id and e.layer_id < 0 and e.event_type == 'collective']
                        self.assertEqual({e.operation for e in comm}, {'all_reduce','all_gather'})
                        gather = next(e for e in comm if e.operation=='all_gather')
                        self.assertEqual(dict(gather.details['logical_input_bytes_per_rank']),
                                         {r:block.token_count*(129280//a_tp)*2 for r in range(a_tp)})
                executor.boundary_events[-1]['end_at'] += 1
                with self.assertRaises(ValueError):
                    validate_full_trace(executor, iterations)

    def test_full_multi_a_capacity_retains_replicated_mla_cache(self):
        one = configuration(True,1,32,8,HW,router_location='f')[-1]
        multi = configuration(True,4,32,8,HW,router_location='f')[-1]
        a_states = [s for s in multi.physical_states if s.physical_identity.startswith('a-pool:')]
        baseline = next(s for s in one.physical_states if s.physical_identity=='a-pool:0')
        self.assertEqual(len(a_states),4)
        for state in a_states:
            self.assertEqual(state.kv_bytes_per_block,baseline.kv_bytes_per_block)
            self.assertGreater(state.model_bytes,baseline.model_bytes/4)
            self.assertLess(state.model_bytes,baseline.model_bytes)
        self.assertTrue(all(s.model_bytes < s.capacity_bytes for s in multi.physical_states))

    def test_routing_does_not_change_when_af_blocks_change(self):
        moe = configuration(False, 1, 2, 8, HW, router_location='f')[3]
        routing = UniformRequestRouting(moe)
        tokens = {i:(4095,) for i in range(32)}
        owners = {i:(0,0) for i in tokens}
        kwargs = dict(iteration_id=0, phase='decode', decode_step=0, layer_id=3)
        whole = routing._synthetic_assignments(tokens, owners, **kwargs)
        split = tuple(a for start in range(0,32,8)
            for a in routing._synthetic_assignments({i:tokens[i] for i in range(start,start+8)}, owners, **kwargs))
        self.assertEqual(whole, split)
        self.assertEqual(whole, UniformRequestRouting(moe)._synthetic_assignments(tokens, owners, **kwargs))
        self.assertGreater(len({a.expert_id//8 for a in whole}), 24)
        self.assertNotEqual(whole, routing._synthetic_assignments(tokens, owners, **dict(kwargs,layer_id=4)))

    def test_f_leader_metadata_and_model_boundaries_contend_and_order(self):
        config = configuration(False, 1, 2, 32, HW, router_location='f',
                               communication_timing=COMM['host'])
        executor, iterations = simulate_full(config, RecordingLookup(HW, COMM),
                                             batch=64, context=32, steps=2)
        self.assertEqual(len(executor.boundary_events), 16)
        self.assertGreaterEqual(iterations[1].start_at, iterations[0].end_at)
        for iteration in iterations:
            for block in iteration.blocks:
                events = [e for e in executor.trace.events
                          if e.iteration_id == iteration.iteration_id and e.block_id == block.block_id]
                router = next(e for e in events if e.stage == 'router')
                self.assertEqual(router.physical_endpoints, ('f-pool:0',))
                metadata = next(e for e in events if e.operation == 'af_router_metadata')
                self.assertEqual(metadata.payload_bytes, block.token_count * 8 * 8)
                self.assertEqual(metadata.direction, 'f_to_f')
                self.assertEqual(metadata.source_endpoint.physical_worker, 'f-pool:0')
                self.assertEqual(metadata.target_endpoint.physical_worker, 'f-pool:1')
                self.assertGreaterEqual(metadata.start_at, router.end_at)
                self.assertIsNone(metadata.dtype)  # mixed int32/FP32 protocol
                arrivals = [e for e in events if e.operation == 'af_a_to_f']
                self.assertEqual(len(arrivals), 2)
                self.assertTrue(all(e.payload_bytes == block.token_count*7168*2 for e in arrivals))
                f_input = next(e for e in arrivals if e.target_endpoint.physical_worker == 'f-pool:0')
                self.assertGreaterEqual(router.start_at, f_input.end_at)
                remote_experts = [e for e in events if e.stage == 'routed_expert'
                                  and e.physical_endpoints == ('f-pool:1',)]
                self.assertTrue(remote_experts)
                self.assertTrue(all(e.start_at >= metadata.end_at for e in remote_experts))
                a_metadata = next(e for e in events if e.operation == 'af_router_metadata'
                                  and e.target_domain == 'a-domain')
                combine = next(e for e in events if e.stage == 'weighted_combine')
                self.assertEqual(a_metadata.payload_bytes, block.token_count * 8 * 8)
                self.assertGreaterEqual(combine.start_at, a_metadata.end_at)
        executor.boundary_events[-1]['end_at'] += 1
        with self.assertRaisesRegex(ValueError, 'Token marked complete'):
            validate_full_trace(executor, iterations)

    def test_actual_full_model_capacity_and_layer_types(self):
        config = configuration(True, 1, 32, 8, HW, router_location='f',
                               communication_timing=COMM['host'])
        plan = config[-1]
        self.assertEqual([x.layer_type for x in plan.layers], ['dense']*3 + ['moe']*58)
        self.assertTrue(all(p.model_bytes < p.capacity_bytes for p in plan.physical_states))
        with self.assertRaisesRegex(ValueError, 'cache exceeds'):
            simulate_full(config, RecordingLookup(HW, COMM), 128, 16384)
        with self.assertRaises(ConfigurationError):
            configuration(True, 1, 1, 8, HW, router_location='f')

    def test_embedding_and_greedy_selection_traffic(self):
        hardware = HardwareProfile(**{f.name: HW[f.name] for f in fields(HardwareProfile) if f.name in HW})
        shapes = boundary_shapes(HW['device'], 4, 7168, 129280)
        embedding = estimate_operator(shapes[0][1], shapes[0][2], hardware)
        argmax = estimate_operator(shapes[3][1], shapes[3][2], hardware)
        self.assertEqual(embedding.activation_bytes, 4*7168*4 + 4*4)
        self.assertEqual(argmax.activation_bytes, 4*129280*2 + 4*4)
        self.assertEqual(argmax.flops, 4*129280)

    def test_hypothetical_f_memory_preserves_a_capacity_and_latency_keys(self):
        config = configuration(True, 1, 1, 8, HW, router_location='f',
            hypothetical_f_memory_bytes=1536 * 2**30)
        states = {s.physical_identity: s for s in config[-1].physical_states}
        self.assertEqual(states['a-pool:0'].capacity_bytes, HW['memory_capacity_bytes'])
        self.assertEqual(states['f-pool:0'].capacity_bytes, 1536 * 2**30)
        self.assertGreater(states['f-pool:0'].model_bytes, HW['memory_capacity_bytes'])
        with self.assertRaisesRegex(ValueError, 'cache exceeds'):
            simulate_full(config, RecordingLookup(HW, COMM), 128, 16384)
        with self.assertRaises(ConfigurationError):
            configuration(True, 1, 1, 8, HW, router_location='f',
                hypothetical_f_memory_bytes=HW['memory_capacity_bytes'])
        lookup = RecordingLookup(HW, COMM)
        key = dict(device=states['f-pool:0'].hardware, m=1, n=2048, k=7168)
        lookup.lookup('gemm', key)
        self.assertEqual(next(iter(lookup.shapes.values()))['key']['device'], HW['device'])
        self.assertEqual(key['device'], states['f-pool:0'].hardware)


if __name__ == '__main__':
    unittest.main()
