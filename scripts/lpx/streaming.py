"""Refine captured native LPX decode DAGs into output-vector dependencies.

The native operator planner remains unchanged and supplies all compute costs.
Only analytical LPX collectives use catalog topology pricing in this experiment.
"""

from dataclasses import asdict, replace
from functools import lru_cache
import math

from TokenSim.comm.collective_types import CollectiveQuery
from TokenSim.comm.collectives import CollectiveModel
from TokenSim.comm.streaming import StreamingConfig, output_vectors
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.operator_data.analytical import analytical_model_for
from scripts.lpx.replay_engine import ReplayKernel, ReplayNode


def duration(nodes):
    kernel = ReplayKernel()
    try:
        kernel.submit(kernel.add(nodes), 0)
        result = kernel.advance()
        if result is None:
            raise SimulationStateError("streaming profile has no completion")
        return result.end
    finally:
        kernel.close()


class StreamingBuilder:
    def __init__(self, runtime, events, profile, config):
        self.runtime, self.events, self.profile = runtime, events, profile
        self.config = config
        self.nodes = list(profile.nodes)
        self.ids = {event.node_id: i for i, event in enumerate(events)}
        self.worker_indices = {
            worker.physical_worker_ref.identity: worker.physical_worker_ref.worker_index
            for worker in runtime.workers
        }
        self.workers = {
            worker.physical_worker_ref.identity: worker for worker in runtime.workers
        }
        from scripts.lpx.fabric import FabricRefinement, LP30Fabric

        raw = runtime.hardware.device("groq_lp30").analytical.get("lpx_fabric")
        self.fabric = (
            FabricRefinement(self, LP30Fabric(**raw)) if raw is not None else None
        )
        if self.fabric:
            for i, event in enumerate(events):
                if event.operation in {
                    "attention",
                    "attention_output",
                    "dense_ffn",
                    "lm_head",
                }:
                    resources = self.fabric.vxm_resources(event.physical_endpoints)
                    self.nodes[i] = replace(
                        self.nodes[i],
                        resources=tuple(
                            dict.fromkeys((*self.nodes[i].resources, *resources))
                        ),
                    )
        self.used_producers = set()
        # A class-level method cache would retain every builder and native DAG.
        self.collective = lru_cache(maxsize=None)(self.collective)
        self.counters = dict(
            catalog_collectives=0,
            streaming_collectives=0,
            output_vectors=0,
            producer_nodes=0,
            producer_service_s=0.0,
            startup_service_s=0.0,
            streamable_work_s=0.0,
            injected_bytes_per_endpoint_sum=0.0,
        )

    def append(self, parents, resources=(), *, delay=0, duration=0, rank=0):
        index = len(self.nodes)
        self.nodes.append(
            ReplayNode(
                tuple(parents),
                resources,
                resources[0] if resources else None,
                delay,
                duration,
                duration,
                rank,
            )
        )
        return index

    def barrier(self, index, parents):
        original = self.nodes[index]
        self.nodes[index] = ReplayNode(tuple(parents), (), None, 0, 0, 0, original.rank)

    def collective(self, endpoints, operation, payload, dtype):
        device = self.runtime.hardware.device(self.workers[endpoints[0]].hardware)
        model = CollectiveModel(
            self.runtime.hardware.topology("groq_lpx256"),
            self.runtime.hardware.links,
            family=device.family,
            launch_us=(
                self.config.collective_launch_us
                if self.config.collective_launch_us is not None
                else device.analytical.get("collective_launch_us")
            ),
            fallback="analytical_only",
        )
        group = model.topology.group_layout(
            [self.worker_indices[endpoint] for endpoint in endpoints]
        )
        payload = payload * len(endpoints) if operation == "all_gather" else payload
        return model.estimate(CollectiveQuery(operation, payload, group, dtype))

    def producer_work(self, index, payload, element_bytes):
        event = self.events[index]
        name = {
            "attention": "output_projection",
            "attention_output": "output_projection",
            "dense_ffn": "down",
        }.get(event.operation)
        if name is None or event.layer_type != "dense":
            return None
        components = event.details.get("components", ())
        candidates = [part for part in components if part["name"] == name]
        if len(candidates) != 1:
            raise ConfigurationError("streaming producer needs one terminal GEMM")
        part = candidates[0]
        if (
            part["table"] != "gemm"
            or any(item["match_type"] != "analytical" for item in components)
            or part["key"]["m"] != 1
            or part["key"]["n"] * element_bytes != payload
        ):
            raise ConfigurationError(
                "streaming supports analytical, single-token, full-output GEMMs"
            )
        device = self.runtime.hardware.device(event.details["device_id"])
        estimate = analytical_model_for(device.family).estimate(
            part["table"], part["key"], device
        )
        total = sum(item["latency_us"] for item in components) * 1e-6
        if not math.isclose(total, self.nodes[index].duration, rel_tol=1e-8):
            raise ConfigurationError(
                "streaming producer service differs from components"
            )
        if not math.isclose(estimate.latency_us, part["latency_us"], rel_tol=1e-8):
            raise ConfigurationError("streaming GEMM differs from analytical estimate")
        work = max(estimate.compute_time_us, estimate.memory_time_us) * 1e-6
        return work, estimate.overhead_us * 1e-6

    def split_producer(self, index, vectors, work, startup):
        if index in self.used_producers:
            raise ConfigurationError(
                "a streaming producer has multiple collective users"
            )
        self.used_producers.add(index)
        original = self.nodes[index]
        if not math.isclose(original.occupancy, original.duration, rel_tol=1e-8):
            raise ConfigurationError("streaming compute needs full resource occupancy")
        prefix = original.duration - work
        previous = self.append(
            original.predecessors,
            original.resources,
            delay=original.delay,
            duration=prefix,
            rank=original.rank,
        )
        ready = []
        for vector in vectors:
            previous = self.append(
                (previous,),
                original.resources,
                duration=work * vector.production_fraction,
                rank=original.rank,
            )
            ready.append(previous)
        self.barrier(index, (previous,))
        self.counters["producer_nodes"] += 1
        self.counters["producer_service_s"] += original.duration
        self.counters["startup_service_s"] += startup
        self.counters["streamable_work_s"] += work
        return ready

    def stream(self, index, estimate, dtype):
        event, original = self.events[index], self.nodes[index]
        element_bytes = dtype_size_bytes(dtype, role="activation_dtype")
        producers = [self.ids[name] for name in event.predecessor_ids]
        work = [
            self.producer_work(parent, event.payload_bytes, element_bytes)
            for parent in producers
        ]
        if not work or any(item is None for item in work):
            return
        if len(producers) != len(event.physical_endpoints) or {
            self.events[p].physical_endpoints[0] for p in producers
        } != set(event.physical_endpoints):
            raise ConfigurationError("streaming reduction needs every rank's producer")
        if len(estimate.levels) != 1 or estimate.levels[0].algorithm != "direct":
            raise ConfigurationError(
                "streaming TP group must fit one direct topology level"
            )
        link = self.runtime.hardware.link(estimate.levels[0].link)
        vectors = output_vectors(
            event.payload_bytes,
            element_bytes,
            self.config.vector_bytes,
            estimate,
            link.latency_us,
        )
        if len(vectors) == 1:
            return
        ranks = (
            [
                self.split_producer(parent, vectors, *cost)
                for parent, cost in zip(producers, work)
            ]
            if self.config.early_output
            else [[parent] * len(vectors) for parent in producers]
        )
        launched = self.append(
            [rank[0] for rank in ranks],
            delay=original.delay,
            duration=estimate.launch_us * 1e-6,
            rank=original.rank,
        )
        completed = []
        for k, vector in enumerate(vectors):
            ready = self.append([launched, *(rank[k] for rank in ranks)])
            if self.fabric:
                completed.append(
                    self.fabric.vector(
                        ready,
                        tuple(event.physical_endpoints),
                        vector.size_bytes,
                        element_bytes,
                        original.rank,
                    )
                )
                continue
            # Reduction and dissemination share the same endpoint TX/RX budget.
            for _ in range(2):
                wire = self.append(
                    (ready,),
                    original.resources,
                    duration=vector.wire_seconds_per_phase,
                    rank=original.rank,
                )
                ready = self.append((wire,), delay=link.latency_us * 1e-6)
            completed.append(ready)
        self.barrier(index, completed)
        self.counters["streaming_collectives"] += 1
        self.counters["output_vectors"] += len(vectors)
        self.counters["injected_bytes_per_endpoint_sum"] += sum(
            2 * vector.wire_bytes_per_phase for vector in vectors
        )
        return True

    def build(self):
        for i, event in enumerate(self.events):
            if self.fabric:
                self.fabric.point_to_point(i, event)
            if event.operation not in {
                "all_reduce",
                "all_gather",
                "reduce_scatter",
                "all_to_all",
            }:
                continue
            endpoints = tuple(event.physical_endpoints)
            if not endpoints or any(
                self.workers[e].hardware != "groq_lp30" for e in endpoints
            ):
                continue
            if event.match_type != "analytical":
                raise ConfigurationError(
                    "streaming cannot decompose measured collectives"
                )
            dtype = event.details["wire_dtype"]
            estimate = self.collective(
                endpoints, event.operation, event.payload_bytes, dtype
            )
            original = self.nodes[i]
            self.nodes[i] = replace(
                original,
                duration=estimate.latency_s,
                occupancy=estimate.latency_s,
            )
            self.counters["catalog_collectives"] += 1
            streamed = False
            if self.config.overlap and event.operation == "all_reduce":
                streamed = self.stream(i, estimate, dtype)
            if self.fabric and not streamed:
                self.fabric.whole_collective(i, event, estimate.launch_us * 1e-6)
        metadata = dict(
            config=asdict(self.config),
            **self.counters,
            nodes=len(self.nodes),
            assumptions=[
                "Uniform final-GEMM output production; operator startup is unchanged.",
                "One collective launch begins after every rank produces its leading vector.",
                "Direct reduce-scatter and all-gather share endpoint injection resources.",
                "Flight latency releases injection; whole-buffer consumers wait for all vectors.",
                "Compute arbitration may occur at vector boundaries between concurrent jobs.",
                "Vector boundaries and per-rank output schedules are analytical assumptions.",
                "No overlap across earlier attention/FFN operators or across model layers.",
            ],
        )
        if self.fabric:
            metadata["fabric"] = self.fabric.report()
        return tuple(self.nodes), metadata


def transform(runtime, events, profile, settings):
    config = StreamingConfig(**settings)
    if profile.metadata["phase"] != "decode":
        return replace(
            profile, metadata={**profile.metadata, "streaming_skipped": "prefill"}
        )
    if profile.metadata["batch"] != 1 or profile.metadata["queries"] != 1:
        raise ConfigurationError(
            "LPX output streaming currently requires single-token decode"
        )
    builder = StreamingBuilder(runtime, events, profile, config)
    nodes, metadata = builder.build()
    return replace(
        profile,
        nodes=nodes,
        metadata={
            **profile.metadata,
            "streaming": metadata,
            "modeled_duration_s": duration(nodes),
        },
    )
