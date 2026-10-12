"""Create a collective model using the operator backend fallback policy."""

from __future__ import annotations

from TokenSim.comm.collectives import CollectiveModel
from TokenSim.hardware.device import DeviceSpec
from TokenSim.operator_data.coverage import MissingShapeReport
from TokenSim.operator_data.lookup import OperatorLookup
from TokenSim.operator_data.package import OperatorDataPackage


def collective_model_for(
    device: DeviceSpec,
    placement,
    links,
    package: OperatorDataPackage | None,
    *,
    fallback: str = "table_first",
    missing_report: MissingShapeReport | None = None,
) -> CollectiveModel:
    """Build the collective model a worker on ``device`` should use.

    ``fallback`` and ``missing_report`` are the worker's operator-table policy
    and checklist, so communication misses are reported (or rejected under
    ``table_only``) exactly like compute misses.
    """
    has_comm_tables = package is not None and bool(
        {"collective", "ep_all2all"} & set(package.tables)
    )
    measured = OperatorLookup(package) if has_comm_tables else None
    launch_us = device.analytical.get("collective_launch_us")
    return CollectiveModel(
        placement.topology,
        links,
        family=device.family,
        measured_lookup=measured,
        launch_us=float(launch_us) if launch_us is not None else None,
        fallback=fallback,
        missing_report=missing_report,
    )
