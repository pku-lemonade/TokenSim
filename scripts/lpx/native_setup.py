"""Build native serving pools and their shared endpoint budgets."""

import simpy

from TokenSim.af.communication.endpoints import EndpointResources
from TokenSim.af.serving.metrics import ResourceMetrics
from TokenSim.af.serving.service import AFService


def create_serving(
    hardware,
    runtimes,
    case,
    workload,
    measurement,
    *,
    trace,
    stream_metrics,
    endpoint_transfer_occupancy,
):
    env = simpy.Environment()
    networks = {
        key: value
        for runtime in runtimes.values()
        for key, value in runtime.cluster.networks.items()
    }
    networks.update(
        handoff="ethernet1600Gb",
        replication=hardware.device(case.decode.a_device).scale_up_link,
    )
    identities = [
        worker.physical_worker_ref.identity
        for runtime in runtimes.values()
        for worker in runtime.workers
    ]
    communication = EndpointResources(
        env, networks, identities, transfer_occupancy=endpoint_transfer_occupancy
    )
    windows = (
        {name: ResourceMetrics(measurement, namespace=name) for name in runtimes}
        if stream_metrics and not trace
        else None
    )
    services = {
        name: AFService(
            env,
            runtime,
            communication=communication,
            seed=workload.seed,
            trace=True,
            retain_trace=trace,
            metric_sink=windows[name].observe if windows is not None else None,
        )
        for name, runtime in runtimes.items()
    }
    return env, communication, services, windows
