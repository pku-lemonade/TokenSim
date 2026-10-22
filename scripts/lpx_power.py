"""Processor power scopes shared by study selection and result export."""

import math


POWER_SCOPES = ("allocated_processors", "processor_racks")


def processor_power_rows(rows, scope, *, rubin_watts=1800, lp30_watts=450):
    if scope not in POWER_SCOPES:
        raise ValueError(f"unknown processor power scope: {scope}")
    result = []
    for row in rows:
        if row["status"] != "ok":
            continue
        inventory = row["inventory"]
        gpu, lpu = inventory.get("rubin", 0), inventory.get("groq_lp30", 0)
        watts = row["power_watts"]
        if scope == "processor_racks":
            expected = gpu * rubin_watts + lpu * lp30_watts
            if not math.isclose(expected, watts, rel_tol=1e-9):
                raise ValueError(
                    "rack sensitivity needs matching per-processor budgets"
                )
            watts = (
                math.ceil(gpu / 72) * 72 * rubin_watts
                + math.ceil(lpu / 256) * 256 * lp30_watts
            )
        result.append(
            dict(
                row,
                power_scope=scope,
                power_watts=watts,
                allocated_power_watts=row["power_watts"],
                gpu_count=gpu,
                lpu_count=lpu,
                minimum_nvl72_racks=math.ceil(gpu / 72),
                minimum_lpx_racks=math.ceil(lpu / 256),
                tokens_per_s_per_mw=row["output_tokens_per_s"] * 1e6 / watts,
            )
        )
    return result
