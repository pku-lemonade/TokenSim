"""Build shared serving pools, endpoint budgets and optional execution drivers."""

import simpy

from TokenSim.af.communication.endpoints import EndpointResources
from TokenSim.af.serving.metrics import ResourceMetrics
from TokenSim.af.serving.service import AFService
from TokenSim.errors import ConfigurationError


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
    execution_factory=None,
):
    if execution_factory is not None and trace:
        raise ConfigurationError("sampled replay cannot retain full execution traces")
    driver = execution_factory(runtimes, measurement) if execution_factory else None
    env = driver.env if driver else simpy.Environment()
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
    communication = (
        driver.communication(networks, identities, endpoint_transfer_occupancy)
        if driver
        else EndpointResources(
            env, networks, identities, transfer_occupancy=endpoint_transfer_occupancy
        )
    )
    windows = (
        {name: ResourceMetrics(measurement, namespace=name) for name in runtimes}
        if driver is None and stream_metrics and not trace
        else None
    )
    services = {
        name: AFService(
            env,
            runtime,
            communication=communication,
            seed=workload.seed,
            trace=driver is None,
            retain_trace=trace,
            metric_sink=windows[name].observe if windows is not None else None,
            execution_driver=driver,
        )
        for name, runtime in runtimes.items()
    }
    if driver:
        driver.bind(services, communication)
    return env, communication, services, windows, driver


def allocated_power(hardware, runtimes, overrides, facility_watts):
    from TokenSim.config.validation import number

    inventory = {}
    for runtime in runtimes.values():
        for worker in runtime.workers:
            inventory[worker.hardware] = inventory.get(worker.hardware, 0) + 1
    power = facility_watts
    for device, count in inventory.items():
        watts = (overrides or {}).get(device, hardware.device(device).tdp_w)
        number(watts, f"{device} power", positive=True)
        power += count * watts
    return inventory, power
