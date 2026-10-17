"""Content identities for experiment inputs and runtime dependencies."""

from dataclasses import asdict, is_dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys

from TokenSim.errors import ConfigurationError


def _json_default(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"experiment identity cannot encode {type(value).__name__}")


def json_bytes(value, *, sort_keys=True):
    try:
        return json.dumps(
            value,
            sort_keys=sort_keys,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=False,
            default=_json_default,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"experiment inputs must be finite JSON values: {exc}"
        ) from exc


def identity(value):
    return sha256(json_bytes(value)).hexdigest()


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class InputFingerprints:
    """Reuse file digests while size, inode, modification and change times agree."""

    def __init__(self):
        self._files = {}

    def paths(self, paths, *, code=False):
        records = []
        for source in sorted({Path(path).resolve() for path in paths}):
            if not source.exists():
                records.append((str(source), "missing"))
                continue
            files = (
                sorted(source.rglob("*.py" if code else "*"))
                if source.is_dir()
                else [source]
            )
            for path in files:
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                stat = path.stat()
                stamp = (
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
                previous = self._files.get(path)
                if previous is None or previous[0] != stamp:
                    previous = self._files[path] = stamp, file_hash(path)
                records.append((str(path), previous[1]))
        return identity(records)


def environment_identity():
    packages = {}
    for name in ("simpy", "numpy", "pandas", "pydantic", "pyarrow", "PyYAML"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=packages)
