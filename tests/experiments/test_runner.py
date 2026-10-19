"""Committed points survive failures and never reuse changed experiment inputs."""

import json
from multiprocessing import get_context
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from TokenSim.errors import ConfigurationError
from TokenSim.experiments import ExperimentRunner


def _shared_point(output, queue):
    runner = ExperimentRunner(output)

    def execute(directory):
        with (Path(output) / "executions").open("a") as stream:
            stream.write("executed\n")
        time.sleep(0.1)
        (directory / "trace").write_bytes(b"complete")
        return {"value": 1}

    result = runner.run({"case": "shared"}, execute)
    queue.put(result.reused)


def _killable_point(output, queue, pause):
    def execute(directory):
        (directory / "trace").write_bytes(b"partial")
        queue.put("staged")
        if pause:
            time.sleep(60)
        (directory / "trace").write_bytes(b"complete")
        return {"value": 1}

    point = ExperimentRunner(output).run({"case": "killed"}, execute)
    queue.put(point.reused)


class ExperimentRunnerTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.calls = 0

    def execute(self, directory):
        self.calls += 1
        (directory / "trace").write_bytes(b"complete")
        return {"value": 3, "tuple": (1, 2)}

    def test_fresh_and_new_runner_reuse_have_identical_results(self):
        first = ExperimentRunner(self.root).run({"seed": 42}, self.execute)
        saved = ExperimentRunner(self.root).run({"seed": 42}, self.execute)
        self.assertEqual(self.calls, 1)
        self.assertFalse(first.reused)
        self.assertTrue(saved.reused)
        self.assertEqual(first.result, saved.result)
        self.assertEqual(saved.result["tuple"], [1, 2])
        repeated = ExperimentRunner(self.root).run(
            {"seed": 42}, self.execute, resume=False
        )
        self.assertFalse(repeated.reused)
        self.assertEqual(self.calls, 2)

    def test_parameters_dirty_code_and_added_deleted_data_invalidate(self):
        data, code = self.root / "inputs", self.root / "source.py"
        data.mkdir()
        item = data / "table"
        item.write_text("one")
        code.write_text("version = 1")
        runner = ExperimentRunner(self.root / "runs", inputs=[data], code_paths=[code])
        points = [runner.run({"seed": 1}, self.execute)]
        points.append(runner.run({"seed": 2}, self.execute))
        item.write_text("two")
        points.append(runner.run({"seed": 1}, self.execute))
        added = data / "new-table"
        added.write_text("new")
        points.append(runner.run({"seed": 1}, self.execute))
        item.unlink()
        points.append(runner.run({"seed": 1}, self.execute))
        code.write_text("version = 2")
        points.append(runner.run({"seed": 1}, self.execute))
        self.assertEqual(len({point.key for point in points}), 6)
        self.assertTrue(all(not point.reused for point in points))

    def test_different_callbacks_in_the_same_source_file_are_distinct(self):
        runner = ExperimentRunner(self.root)

        def other(directory):
            return {"value": 4}

        first = runner.run({}, self.execute)
        second = runner.run({}, other)
        self.assertNotEqual(first.key, second.key)
        self.assertEqual(second.result, {"value": 4})
        self.assertTrue(runner.run({}, other).reused)

    def test_failed_point_retries_without_losing_other_committed_points(self):
        runner = ExperimentRunner(self.root)
        completed = runner.run({"case": 1}, self.execute)

        def interrupted(directory):
            (directory / "partial").write_text("partial")
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            runner.run({"case": 2}, interrupted)
        self.assertTrue(runner.run({"case": 1}, self.execute).reused)
        self.assertTrue((completed.directory / "manifest.json").is_file())
        self.assertFalse(list(self.root.glob(".*-*")))
        self.assertFalse(runner.run({"case": 2}, self.execute).reused)

    def test_missing_changed_and_corrupt_outputs_rerun(self):
        runner = ExperimentRunner(self.root)
        for damage in ("missing", "changed", "result", "manifest", "status"):
            point = runner.run({"damage": damage}, self.execute)
            if damage == "missing":
                (point.directory / "trace").unlink()
            elif damage == "changed":
                (point.directory / "trace").write_bytes(b"bad")
            elif damage == "status":
                path = point.directory / "manifest.json"
                manifest = json.loads(path.read_text())
                manifest["status"] = "running"
                path.write_text(json.dumps(manifest))
            else:
                (point.directory / f"{damage}.json").write_text("broken")
            with self.subTest(damage=damage):
                self.assertFalse(runner.run({"damage": damage}, self.execute).reused)

    def test_full_hash_detects_corruption_with_preserved_size_and_mtime(self):
        runner = ExperimentRunner(self.root, verify="sha256")
        point = runner.run({}, self.execute)
        path = point.directory / "trace"
        stat = path.stat()
        path.write_bytes(b"modified")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertFalse(runner.run({}, self.execute).reused)
        self.assertTrue(runner.run({}, self.execute).reused)

    def test_stronger_verification_rebuilds_a_stat_only_checkpoint(self):
        ExperimentRunner(self.root).run({}, self.execute)
        runner = ExperimentRunner(self.root, verify="sha256")
        self.assertFalse(runner.run({}, self.execute).reused)
        self.assertTrue(runner.run({}, self.execute).reused)

    def test_input_change_during_execution_is_not_committed(self):
        source = self.root / "data"
        source.write_text("initial")
        runner = ExperimentRunner(self.root / "runs", inputs=[source])

        def execute(directory):
            source.write_text("changed")
            return {}

        with self.assertRaisesRegex(ConfigurationError, "changed during execution"):
            runner.run({}, execute)
        self.assertFalse(list((self.root / "runs").glob("*/manifest.json")))

    def test_rejects_invalid_inputs_reserved_names_and_symlinks(self):
        for values in (dict(resume=1), dict(verify="unknown")):
            with self.assertRaises(ConfigurationError):
                ExperimentRunner(self.root, **values)
        runner = ExperimentRunner(self.root)
        for parameters in ([], {"bad": float("nan")}, {"bad": object()}):
            with self.assertRaises(ConfigurationError):
                runner.run(parameters, self.execute)
        for name in ("result.json", "manifest.json", "link"):

            def execute(directory):
                if name == "link":
                    (directory / name).symlink_to(self.root / "outside")
                else:
                    (directory / name).write_text("reserved")
                return {}

            with self.subTest(name=name), self.assertRaises(ConfigurationError):
                runner.run({"name": name}, execute)

    def test_same_point_in_two_processes_executes_once(self):
        context = get_context("spawn")
        queue = context.Queue()
        workers = [
            context.Process(target=_shared_point, args=(str(self.root), queue))
            for _ in range(2)
        ]
        for worker in workers:
            worker.start()
        try:
            observations = sorted(queue.get(timeout=20) for _ in workers)
            for worker in workers:
                worker.join(20)
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual(observations, [False, True])
            self.assertEqual((self.root / "executions").read_text(), "executed\n")
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
            queue.close()

    def test_killed_process_releases_lock_and_stale_staging_is_replaced(self):
        context = get_context("spawn")
        for pause in (True, False):
            queue = context.Queue()
            worker = context.Process(
                target=_killable_point, args=(str(self.root), queue, pause)
            )
            worker.start()
            try:
                self.assertEqual(queue.get(timeout=20), "staged")
                if pause:
                    worker.kill()
                else:
                    self.assertFalse(queue.get(timeout=20))
                worker.join(20)
                self.assertEqual(worker.exitcode, -9 if pause else 0)
                self.assertEqual(bool(list(self.root.glob(".*-*"))), pause)
            finally:
                if worker.is_alive():
                    worker.kill()
                    worker.join()
                queue.close()
        completed = list(self.root.glob("*/manifest.json"))
        self.assertEqual(len(completed), 1)
        self.assertEqual((completed[0].parent / "trace").read_bytes(), b"complete")
