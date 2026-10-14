"""Compatibility adapters for AF result consumers."""

import json
from pathlib import Path
from TokenSim.reporting import *
from TokenSim.reporting import (
    export_result as _export_result,
    print_all_stats as _print_all_stats,
)


def export_result(
    args,
    g_time,
    engine,
    model_config,
    cluster,
    request_count,
    prefill_lens,
    decode_lens,
    requests,
    notdone,
    duration,
    simulator_wall_time=0,
):
    path = (
        Path(args.results_path)
        if args.results_path
        else Path("results")
        / model_config.model
        / Path(args.cluster).parent.name
        / Path(args.cluster).stem
    )
    run = RunReport(path, args.qps, args.batching)
    return _export_result(
        run,
        g_time,
        engine,
        cluster,
        request_count,
        prefill_lens,
        decode_lens,
        requests,
        notdone,
        duration,
        simulator_wall_time,
    )


def print_all_stats(
    g_time, requests, engine, model_config, duration, request_count, prefill_lens
):
    return _print_all_stats(
        g_time, requests, engine, duration, request_count, prefill_lens
    )
