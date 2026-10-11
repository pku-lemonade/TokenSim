"""Normalize measured mHC (hyper-connection residual mixing) pre/post modules.

Each row prices one operation (pre or post) for the recorded number of sites
per layer; consumers must not multiply by the site count again.
"""

import math

from TokenSim.operator_data.package import OperatorDataValidationError


def normalize(rows, *, source_id, source_hardware, source_version):
    result = []
    for row in rows:
        if (
            min(
                int(row.get(name, 0))
                for name in ("hidden_size", "hc_mult", "num_sites", "sinkhorn_iters")
            )
            <= 0
        ):
            raise OperatorDataValidationError("unsupported mHC measurement scope")
        if row.get("op_name") not in ("pre", "post"):
            raise OperatorDataValidationError("unknown mHC operation")
        latency = float(row["latency"])
        if not math.isfinite(latency) or latency <= 0 or row["num_tokens"] < 1:
            raise OperatorDataValidationError("invalid mHC measurement")
        result.append(
            {
                "op_name": row["op_name"],
                "architecture": row["architecture"],
                "kernel_source": row["kernel_source"],
                "hidden_size": int(row["hidden_size"]),
                "hc_mult": int(row["hc_mult"]),
                "num_sites": int(row["num_sites"]),
                "sinkhorn_iters": int(row["sinkhorn_iters"]),
                "num_tokens": int(row["num_tokens"]),
                "latency_us": latency * 1000.0,
                "source_id": source_id,
                "kernel": row["kernel_source"],
                "grade": "B",
                "source_hardware": source_hardware,
                "source_version": source_version,
                "source_framework": row["framework"],
            }
        )
    if not result:
        raise OperatorDataValidationError("no mHC rows in source")
    return result
