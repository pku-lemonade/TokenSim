import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from TokenSim.errors import WorkloadValidationError
from scripts.generate_lpx_gpu_outputs import complete, generate, stream_events
from scripts.generate_lpx_gemma_outputs import run_request, export
from scripts.lpx.gpu_metrics import (
    acceptance,
    counters,
    counter_delta,
    decode_timing,
    output_quality,
)


def sample_metrics(drafts=2, proposed=6, accepted=3):
    return "\n".join(
        [
            f'vllm:spec_decode_num_drafts_total{{model_name="gemma",engine="0"}} {drafts}',
            f'vllm:spec_decode_num_draft_tokens_total{{model_name="gemma"}} {proposed}',
            f'vllm:spec_decode_num_accepted_tokens_total{{model_name="gemma"}} {accepted}',
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="gemma",position="0"} 2',
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="gemma",position="1"} 1',
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="gemma",position="2"} 0',
            'vllm:spec_decode_num_drafts_total{model_name="other"} 999',
            'vllm:spec_decode_num_drafts_created{model_name="gemma"} 123456',
        ]
    )


class GpuMetricsTest(unittest.TestCase):
    def test_optional_engine_counters_preserve_missing_old_records(self):
        before = counters(
            'vllm:iteration_tokens_total_count{model_name="gemma"} 3\n'
            'vllm:iteration_tokens_total_count{model_name="other"} 90\n'
            'vllm:num_preemptions_total{model_name="gemma"} 0\n',
            "gemma",
        )
        self.assertEqual(before, dict(emitted_steps=3, preemptions=0))
        after = dict(emitted_steps=8, preemptions=0)
        result = counter_delta(before, after, False)
        self.assertEqual(result["emitted_steps"], 5)
        self.assertEqual(result["preemptions"], 0)
        self.assertNotIn("emitted_steps", counter_delta({}, {}, False))
        with self.assertRaises(WorkloadValidationError):
            counter_delta({}, after, False)
        with self.assertRaises(WorkloadValidationError):
            counter_delta(after, before, False)

    def test_repetition_diagnostic_does_not_infer_missing_text_or_tokens(self):
        self.assertIsNone(output_quality({})["repetition_flag"])
        measured = output_quality(
            dict(token_ids=[1] + [2] * 128, content="", finish_reason="length")
        )
        self.assertEqual(measured["longest_identical_token_run"], 128)
        self.assertTrue(measured["repetition_flag"])
        self.assertTrue(measured["empty_answer"])
        self.assertAlmostEqual(measured["dominant_token_share"], 128 / 129)
        self.assertFalse(
            output_quality(dict(token_ids=list(range(200))))["repetition_flag"]
        )

    def test_periodic_repetition_detects_two_token_loop_without_single_token_run(self):
        measured = output_quality(dict(token_ids=[9] + [1, 2] * 80 + [8]))
        self.assertEqual(measured["longest_identical_token_run"], 1)
        self.assertEqual(measured["longest_periodic_token_span"], 160)
        self.assertEqual(measured["periodic_token_period"], 2)
        self.assertTrue(measured["repetition_flag"])
        self.assertFalse(
            output_quality(dict(token_ids=list(range(9)) * 20))["repetition_flag"]
        )

    def test_model_filter_and_position_counters(self):
        measured = counters(sample_metrics(), "gemma")
        self.assertEqual(measured["drafts"], 2)
        self.assertEqual(measured["accepted_per_position"], {"0": 2, "1": 1, "2": 0})
        self.assertEqual(counters(sample_metrics(), "absent"), {})

    def test_engine_series_sum_and_old_names(self):
        text = (
            sample_metrics()
            + '\nvllm:spec_decode_num_drafts{model_name="gemma",engine="1"} 3'
        )
        self.assertEqual(counters(text, "gemma")["drafts"], 5)

    def test_invalid_metric_values(self):
        for value in ("NaN", "+Inf", "-1", "1.5"):
            with self.subTest(value=value), self.assertRaises(WorkloadValidationError):
                counters(
                    f'vllm:spec_decode_num_drafts_total{{model_name="gemma"}} {value}',
                    "gemma",
                )
        with self.assertRaises(WorkloadValidationError):
            counters(
                'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="gemma"} 1',
                "gemma",
            )

    def test_counter_differences(self):
        before = counters(sample_metrics(), "gemma")
        after = counters(sample_metrics(4, 12, 3), "gemma")
        delta = counter_delta(before, after, True)
        self.assertEqual(delta["drafts"], 2)
        self.assertEqual(delta["draft_tokens"], 6)
        self.assertEqual(delta["accepted_tokens"], 0)
        self.assertEqual(counter_delta({}, {}, False)["drafts"], 0)
        with self.assertRaises(WorkloadValidationError):
            counter_delta({}, after, True)
        with self.assertRaises(WorkloadValidationError):
            counter_delta(after, before, True)

    def test_decode_speed_and_single_chunk(self):
        result = decode_timing(11, [2.0, 2.5, 4.0])
        self.assertEqual(result, dict(ttft_s=2.0, decode_s=2.0, output_tps=5.0))
        self.assertIsNone(decode_timing(2, [1.0])["output_tps"])
        for tokens, times in (
            (0, [1]),
            (3, []),
            (3, [2, 1]),
            (3, [float("nan")]),
            (3, [-1]),
        ):
            with (
                self.subTest(tokens=tokens, times=times),
                self.assertRaises(WorkloadValidationError),
            ):
                decode_timing(tokens, times)

    def test_acceptance_identity_and_verification_rows(self):
        turn = dict(completion_tokens=6, **counters(sample_metrics(), "gemma"))
        result = acceptance(turn, 3)
        self.assertEqual(result["tau"], 2.5)
        self.assertEqual(result["counter_identity_residual"], 0)
        self.assertEqual(result["per_row"], [5, [0, 0, 0, 2]])
        self.assertEqual(
            result["per_position_acceptance"], {"0": 1.0, "1": 0.5, "2": 0}
        )
        self.assertNotIn("per_row", acceptance(turn, 3, False))
        for change in (
            dict(drafts=0),
            dict(completion_tokens=20),
            dict(accepted_tokens=4),
            dict(draft_tokens=7),
        ):
            with (
                self.subTest(change=change),
                self.assertRaises(WorkloadValidationError),
            ):
                acceptance(dict(turn, **change), 3)


class GpuClientTest(unittest.TestCase):
    def test_sse_comments_and_multiline_data(self):
        stream = io.BytesIO(
            b': comment\n\ndata: {"choices": [],\ndata: "usage": {"completion_tokens": 2}}\n\ndata: [DONE]\n\n'
        )
        self.assertEqual(
            list(stream_events(stream))[0]["usage"]["completion_tokens"], 2
        )
        with self.assertRaises(WorkloadValidationError):
            list(stream_events(io.BytesIO(b'data: {"choices": []}\n\n')))
        with self.assertRaises(WorkloadValidationError):
            list(stream_events(io.BytesIO(b'data: {"error": "failed"}\n\n')))

    def test_complete_payload_usage_and_chunk_times(self):
        events = [
            dict(choices=[dict(delta=dict(role="assistant"))]),
            dict(choices=[dict(delta=dict(reasoning="thought"), token_ids=[1, 2])]),
            dict(choices=[dict(delta=dict(content="answer"), token_ids=[3, 4, 5])]),
            dict(choices=[dict(delta={}, finish_reason="stop")]),
            dict(choices=[], usage=dict(prompt_tokens=8, completion_tokens=5)),
        ]
        stream = io.BytesIO(
            (
                "".join("data: " + json.dumps(event) + "\n\n" for event in events)
                + "data: [DONE]\n\n"
            ).encode()
        )
        args = SimpleNamespace(
            model="gemma",
            api_base="http://localhost:8000/v1",
            method="plain",
            metrics_settle_seconds=0,
        )
        with (
            patch("scripts.generate_lpx_gpu_outputs.metrics", return_value={}),
            patch(
                "scripts.generate_lpx_gpu_outputs.urllib.request.urlopen",
                return_value=stream,
            ) as opened,
            patch(
                "scripts.generate_lpx_gpu_outputs.time.perf_counter",
                side_effect=[10, 11, 12, 13, 14, 15, 16],
            ),
        ):
            turn = complete(
                args,
                [dict(role="user", content="prompt")],
                dict(reasoning=True, max_tokens=10),
            )
        body = json.loads(opened.call_args.args[0].data)
        self.assertEqual(body["chat_template_kwargs"], dict(enable_thinking=True))
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["seed"], 1)
        self.assertTrue(body["stream_options"]["include_usage"])
        self.assertTrue(body["return_token_ids"])
        for field in ("provider", "reasoning", "usage"):
            self.assertNotIn(field, body)
        self.assertEqual(turn["reasoning"], "thought")
        self.assertEqual(turn["content"], "answer")
        self.assertEqual(turn["chunk_arrivals_s"], [1, 2, 3, 4, 5])
        self.assertEqual(turn["ttft_s"], 2)
        self.assertEqual(turn["decode_s"], 1)
        self.assertEqual(turn["output_tps"], 4)
        self.assertEqual(turn["token_ids"], [1, 2, 3, 4, 5])
        self.assertNotIn("prompt_token_ids", turn)

    def test_resume_multiturn_without_writing_prompts(self):
        request = dict(
            id="test",
            set="speed-coding",
            meta={},
            turns=["private question", "private follow-up"],
            reasoning=False,
            max_tokens=10,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "requests.jsonl").write_text(json.dumps(request) + "\n")
            args = SimpleNamespace(
                requests=path / "requests.jsonl",
                outputs=path / "outputs.jsonl",
                method="plain",
                draft_tokens=0,
                weight_precision="bf16",
                kv_precision="bf16",
                drafter_precision="bf16",
                model="gemma",
                run_label="test",
                max_tokens=None,
                metrics_settle_seconds=0,
                environment=None,
                set=None,
                thinking="both",
                context=None,
                limit=None,
            )
            turn = dict(
                content="answer",
                reasoning="",
                completion_tokens=4,
                output_tps=2,
                drafts=0,
            )
            with patch(
                "scripts.generate_lpx_gpu_outputs.complete",
                side_effect=lambda *args: dict(turn),
            ) as completed:
                generate(args)
                self.assertEqual(completed.call_count, 2)
                self.assertEqual(
                    completed.call_args.args[1][1],
                    dict(role="assistant", content="answer"),
                )
                generate(args)
                self.assertEqual(completed.call_count, 2)
            text = args.outputs.read_text()
            self.assertNotIn("private", text)
            self.assertEqual(len(json.loads(text)["turns"]), 2)
            args.max_tokens = 5
            with self.assertRaises(WorkloadValidationError):
                generate(args)

    def test_remote_client_preserves_turns_and_export_format(self):
        request = dict(
            id="test",
            set="speed-coding",
            reasoning=False,
            meta={},
            turns=["question", "follow-up"],
        )
        response = dict(
            choices=[
                dict(
                    message=dict(content="answer", reasoning="thought"),
                    finish_reason="stop",
                )
            ],
            provider="provider",
            model="gemma",
            usage=dict(completion_tokens=5),
            elapsed_s=2,
        )
        with patch(
            "scripts.generate_lpx_gemma_outputs.complete", return_value=response
        ):
            result = run_request("key", "provider", request)
        self.assertEqual(len(result["turns"]), 2)
        self.assertEqual(result["turns"][0]["reasoning"], "thought")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            requests, outputs = path / "requests.jsonl", path / "outputs.jsonl"
            requests.write_text(json.dumps(request) + "\n")
            outputs.write_text(json.dumps(result) + "\n")
            export(
                SimpleNamespace(
                    requests=requests, outputs=outputs, export=path / "rows"
                )
            )
            rows = json.loads((path / "rows/speed-coding-plain.json").read_text())[
                "rows"
            ]
            self.assertEqual(
                rows[1]["row"]["messages"][0]["content"],
                "question\n\nanswer\n\nfollow-up",
            )
            self.assertEqual(rows[0]["row"]["thinking"], "thought")


if __name__ == "__main__":
    unittest.main()
