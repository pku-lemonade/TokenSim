"""Run independent experiment points and reuse only committed matching outputs."""

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import inspect
import json
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter

from TokenSim.errors import ConfigurationError
from .fingerprints import (
    InputFingerprints,
    environment_identity,
    file_hash,
    identity,
    json_bytes,
)


@dataclass(frozen=True)
class ExperimentResult:
    key: str
    directory: Path
    result: dict
    reused: bool


class ExperimentRunner:
    """Commit one callback result and its files per complete parameter set.

    The callback receives a private staging directory and returns a JSON dict.
    Declare external inputs with ``inputs`` and include all case settings in
    ``parameters``. Failed points are retried; completed points are reusable.
    """

    def __init__(
        self, output, *, inputs=(), code_paths=None, resume=True, verify="stat"
    ):
        if type(resume) is not bool or verify not in ("stat", "sha256"):
            raise ConfigurationError(
                "resume must be boolean and verify must be stat or sha256"
            )
        self.output = Path(output).resolve()
        self.inputs = tuple(Path(path) for path in inputs)
        self.code_paths = (
            tuple(code_paths)
            if code_paths is not None
            else (Path(__file__).resolve().parents[1],)
        )
        self.resume = resume
        self.verify = verify
        self._fingerprints = InputFingerprints()
        self._environment = environment_identity()

    def run(self, parameters, execute, *, resume=None):
        """Execute or reuse a point; the lock is released even after interruption."""
        use_saved = self.resume if resume is None else resume
        if type(use_saved) is not bool or not isinstance(parameters, dict):
            raise ConfigurationError(
                "parameters must be a dict and resume must be boolean"
            )
        if not callable(execute):
            raise ConfigurationError("experiment execute must be callable")
        parameters = json.loads(json_bytes(parameters))
        try:
            callback_file = inspect.getsourcefile(execute)
        except TypeError:
            try:
                callback_file = inspect.getsourcefile(type(execute))
            except TypeError:
                callback_file = None
        code_paths = self.code_paths + ((callback_file,) if callback_file else ())
        signature = dict(
            schema=1,
            parameters=parameters,
            callback=dict(
                module=getattr(execute, "__module__", type(execute).__module__),
                name=getattr(execute, "__qualname__", type(execute).__qualname__),
            ),
            code=self._fingerprints.paths(code_paths, code=True),
            inputs=self._fingerprints.paths(self.inputs),
            environment=self._environment,
        )
        key = identity(signature)
        directory = self.output / key
        with self._locked(key):
            saved = self._load(directory, key) if use_saved else None
            if saved is not None:
                return ExperimentResult(key, directory, saved, True)
            # A killed process can leave staging files; this key's lock proves
            # no current writer owns them. Other points are left alone.
            for stale in self.output.glob(f".{key}-*"):
                if stale.is_dir():
                    shutil.rmtree(stale)
            with TemporaryDirectory(prefix=f".{key}-", dir=self.output) as staging:
                work = Path(staging)
                started = perf_counter()
                result = execute(work)
                if not isinstance(result, dict):
                    raise ConfigurationError("experiment callback must return a dict")
                if any(
                    (work / name).exists() for name in ("result.json", "manifest.json")
                ):
                    raise ConfigurationError(
                        "result.json and manifest.json are reserved runner files"
                    )
                if (
                    self._fingerprints.paths(code_paths, code=True) != signature["code"]
                    or self._fingerprints.paths(self.inputs) != signature["inputs"]
                ):
                    raise ConfigurationError(
                        "experiment code or input files changed during execution; rerun the point"
                    )
                # Normalize fresh and resumed values identically, including tuples.
                encoded = json_bytes(result, sort_keys=False)
                (work / "result.json").write_bytes(encoded)
                artifacts = self._artifacts(work)
                manifest = dict(
                    schema=1,
                    key=key,
                    status="complete",
                    signature=signature,
                    elapsed_seconds=perf_counter() - started,
                    artifacts=artifacts,
                )
                (work / "manifest.json").write_bytes(json_bytes(manifest))
                if directory.exists():
                    shutil.rmtree(directory)
                work.replace(directory)
                return ExperimentResult(key, directory, json.loads(encoded), False)

    @contextmanager
    def _locked(self, key):
        locks = self.output / ".locks"
        locks.mkdir(parents=True, exist_ok=True)
        with (locks / key).open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _artifacts(self, directory):
        records = {}
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ConfigurationError(
                    "experiment artifacts must not be symbolic links"
                )
            if not path.is_file():
                continue
            stat = path.stat()
            record = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            # Always protect the small callback result. Full trace hashing is
            # explicit because it adds another complete read of large files.
            if path.name == "result.json" or self.verify == "sha256":
                record["sha256"] = file_hash(path)
            records[path.relative_to(directory).as_posix()] = record
        return records

    def _load(self, directory, key):
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
            if (
                manifest.get("schema") != 1
                or manifest.get("key") != key
                or manifest.get("status") != "complete"
                or identity(manifest.get("signature")) != key
            ):
                return None
            artifacts = manifest["artifacts"]
            if not isinstance(artifacts, dict) or "result.json" not in artifacts:
                return None
            for name, expected in artifacts.items():
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts:
                    return None
                path = directory / name
                if path.is_symlink() or not path.is_file():
                    return None
                if not path.resolve().is_relative_to(directory.resolve()):
                    return None
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != (
                    expected["size"],
                    expected["mtime_ns"],
                ):
                    return None
                if self.verify == "sha256" and "sha256" not in expected:
                    return None
                if "sha256" in expected and file_hash(path) != expected["sha256"]:
                    return None
            value = json.loads((directory / "result.json").read_text())
            return value if isinstance(value, dict) else None
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return None
