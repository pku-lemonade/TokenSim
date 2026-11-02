"""Replay greedy Gemma assistant proposals plus native target verification."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError
from scripts.lpx.assistant_pool import (
    AssistantSpec,
    assistant_memory,
    draft_step,
    element,
    remote_workspace,
)
from scripts.lpx.fidelity import load
from scripts.lpx.io_pool import pooled_program, transformer_profile
from scripts.lpx.mfu_metrics import throughput_bounds, useful_matrix_flops
from scripts.lpx.mfu_replay import replay_cohorts
from scripts.lpx.program import LP30Program
from scripts.lpx.replay_profiles import save_profile
from scripts.lpx.speculative_fit import committed_tokens
from scripts.run_lpx_article_sweep import article_config
from scripts.run_lpx_context_parallel import environment
from scripts.search_lpx_mfu import build_cache
from scripts.search_lpx_native import write_json


def compose(
    hardware,
    target,
    transformer,
    *,
    spec,
    layout,
    draft_tp,
    draft_dtype,
    tokens,
    slots,
    resident,
):
    tp, pp, kvp, head_count = (
        layout[name] for name in ("tp", "pp", "kvp", "pool_chips")
    )
    target_count = tp * pp
    heads = tuple(range(target_count, target_count + head_count))
    draft_chips = tuple(
        range(target_count + head_count, target_count + head_count + draft_tp)
    )
    memory = assistant_memory(
        spec,
        draft_tp,
        draft_dtype,
        slots=slots,
        resident=resident,
        verification_queries=tokens + 1,
    )
    verifier, head_admission = pooled_program(
        hardware,
        target,
        transformer,
        tp=tp,
        pp=pp,
        pool_chips=head_count,
        queries=tokens + 1,
    )
    total = target_count + head_count + draft_tp
    programs = []
    for slot in range(slots):
        program = LP30Program(hardware, total)
        arena_name = f"assistant/slot:{slot}"
        acquired = program.acquire((), arena_name)
        ready = acquired
        for _ in range(tokens):
            ready = draft_step(
                program,
                (ready,),
                spec=spec,
                dtype=draft_dtype,
                draft_chips=draft_chips,
                head_chips=heads,
                target_chips=tuple(range((pp - 1) * tp, pp * tp)),
                kvp=kvp,
                launch_resource="mfu-search/host:0",
                launch_seconds=4e-6,
            )
        program.release_at(acquired, ready)
        verified = program.include(verifier, (ready,))
        feedback = program.acquire((verified,), arena_name)
        arrived = program.transfer(
            (feedback,),
            [
                ((pp - 1) * tp, chip, (tokens + 1) * target.hidden_size * 2)
                for chip in draft_chips
            ],
        )
        selected = [
            element(
                program,
                (arrived,),
                chip,
                "other_residual",
                tokens + 1,
                target.hidden_size,
            )
            for chip in draft_chips
        ]
        completed = program.append(tuple(selected))
        program.release_at(feedback, completed)
        programs.append(program.profile())
    return programs, dict(
        assistant_memory=memory,
        target_head=head_admission,
        shared_kv_scope="Three local assistant layers use target layer 58; the global assistant layer uses target layer 59. All cache reads and attention execute at existing native cache owners.",
        remote_scratch_per_transformer_chip=slots
        * remote_workspace(
            spec,
            tp=tp,
            kvp=kvp,
            tile_keys=hardware.device("groq_lp30")
            .analytical["lpx_pipeline"]
            .get("attention_tile_keys", 256),
        ),
        feedback="All verifier hidden rows are copied to the drafter, one accepted row is retained per request, and greedy draft/target token comparison is priced as vector selection.",
        precision="Matrix weights use the selected draft dtype; norms, activations and shared target KV retain BF16/BF16/FP8 respectively.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assistant-config",
        type=Path,
        default=ROOT / "experiments/lpx/gemma4-31b-assistant-config.json",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/fidelity-study.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--kvp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=10)
    parser.add_argument("--pool-chips", type=int, default=3)
    parser.add_argument("--draft-tp", type=int, default=2)
    parser.add_argument("--draft-dtype", choices=("fp8", "bf16"), default="bf16")
    parser.add_argument("--draft-tokens", type=int, default=3)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--resident-requests", type=int)
    parser.add_argument("--issue-us", type=float)
    parser.add_argument("--warmup-rounds", type=int, default=4)
    parser.add_argument("--measured-rounds", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        for name in (
            "tp",
            "kvp",
            "pp",
            "pool_chips",
            "draft_tp",
            "draft_tokens",
            "slots",
            "warmup_rounds",
        ):
            integer(getattr(args, name), name, minimum=1)
        integer(args.measured_rounds, "measured rounds", minimum=2)
        build_cache()
        config = article_config(load(args.config))
        config["compute"]["attention_mfu"] = 0.5
        hardware, target, _, _ = environment(config)
        raw = json.loads(args.assistant_config.read_text())
        spec = AssistantSpec.from_config(raw, target)
        # Independent remote-attention arenas share no native verification scratch.
        config["memory"]["reserve_bytes"] += args.slots * remote_workspace(
            spec,
            tp=args.tp,
            kvp=args.kvp,
            tile_keys=hardware.device("groq_lp30")
            .analytical["lpx_pipeline"]
            .get("attention_tile_keys", 256),
        )
        hardware, target, transformer, admission = transformer_profile(
            config,
            tp=args.tp,
            kvp=args.kvp,
            pp=args.pp,
            queries=args.draft_tokens + 1,
            cache=args.output / "profiles",
            resident=args.resident_requests,
        )
        if (
            spec.windows != (1024, 1024, 1024, 0)
            or args.pp != 10
            or admission["layer_counts"] != (6,) * 10
        ):
            raise ConfigurationError(
                "assistant cache mapping currently requires six target layers per stage and the official 31B assistant"
            )
        layout = dict(tp=args.tp, kvp=args.kvp, pp=args.pp, pool_chips=args.pool_chips)
        resident = admission["resident_requests"]
        profiles, memory = compose(
            hardware,
            target,
            transformer,
            spec=spec,
            layout=layout,
            draft_tp=args.draft_tp,
            draft_dtype=args.draft_dtype,
            tokens=args.draft_tokens,
            slots=args.slots,
            resident=resident,
        )
        profiles = [
            save_profile(
                args.output / f"program-slot-{i}.json.gz",
                profile,
                dict(
                    transformer=admission["native_profile"]["input_sha256"],
                    program_sha256=profile.metadata["program_sha256"],
                ),
            )
            for i, profile in enumerate(profiles)
        ]
        for profile in profiles:
            profile.verify()
        issue = (
            args.issue_us * 1e-6
            if args.issue_us is not None
            else profiles[0].metadata["modeled_duration_s"] / resident * 1.01
        )
        number(issue, "assistant issue interval", positive=True)
        print(
            json.dumps(
                dict(
                    stage="captured",
                    resident=resident,
                    isolated_us=profiles[0].metadata["modeled_duration_s"] * 1e6,
                    issue_us=issue * 1e6,
                )
            ),
            flush=True,
        )
        replay = replay_cohorts(
            profiles[0],
            resident_requests=resident,
            additional_profiles=profiles[1:],
            warmup_rounds=args.warmup_rounds,
            measured_rounds=args.measured_rounds,
            initial_spacing_s=issue,
            issue_interval_s=issue,
        )
        work = useful_matrix_flops(target, 101000)
        chips = args.tp * args.pp + args.pool_chips + args.draft_tp
        peak = chips * hardware.device("groq_lp30").peak_compute_for("fp8").value
        # Replay counts completed verification rounds. Useful credit is accepted tokens only.
        rounds = replay["aggregate_tps"]
        scenarios = [
            dict(
                alpha=alpha,
                expected_committed=committed_tokens(alpha, args.draft_tokens),
                aggregate_tps=rounds * committed_tokens(alpha, args.draft_tokens),
                per_request_tps=rounds
                / resident
                * committed_tokens(alpha, args.draft_tokens),
                mfu_percent=100
                * rounds
                * committed_tokens(alpha, args.draft_tokens)
                * work["total"]
                / peak,
            )
            for alpha in (0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1)
        ]
        result = dict(
            source_cpp_sha256=hashlib.sha256(
                (ROOT / "scripts/lpx/replay_engine.cpp").read_bytes()
            ).hexdigest(),
            config=config,
            layout=layout,
            chips=chips,
            draft_tp=args.draft_tp,
            draft_dtype=args.draft_dtype,
            draft_tokens=args.draft_tokens,
            slots=args.slots,
            native_transformer=admission,
            assistant_spec=asdict(spec),
            parameters=spec.parameters(),
            assistant_config_sha256=hashlib.sha256(
                args.assistant_config.read_bytes()
            ).hexdigest(),
            memory=memory,
            programs=[p.metadata for p in profiles],
            round_replay=replay,
            scenarios=scenarios,
            useful_work_per_committed_token=work,
            single_program_bounds=throughput_bounds(
                profiles[0].nodes, resident_requests=resident, batch=1
            ),
            scope="Conditional greedy draft/verify replay with explicit assistant weights, scratch, remote KV-owner attention and feedback copies. IID acceptance scenarios are not 100K workload measurements. Physical bank placement, native target-stage lifetimes and compiler execution remain unverified.",
        )
        write_json(args.output / "result.json", result)
        print(
            json.dumps(
                dict(
                    chips=chips,
                    resident=resident,
                    rounds_per_s=rounds,
                    scenarios=scenarios,
                )
            ),
            flush=True,
        )
        return 0
    except (ConfigurationError, OSError, ValueError) as exc:
        print(f"LPX assistant: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
