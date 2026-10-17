"""Optional Rust experiment; production execution does not import this module."""

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import functools
import hashlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]
SOURCES = sorted(
    [SOURCE / name for name in ("Cargo.toml", "Cargo.lock", "build.rs")]
    + list((SOURCE / "src").glob("*.rs"))
)
IDENTITY = hashlib.sha256(b"".join(path.read_bytes() for path in SOURCES)).hexdigest()
TARGET = ROOT / ".codex" / "af-rust" / sys.implementation.cache_tag / IDENTITY
NAME = "tokensim_rust_probe"
SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"
BINARY = TARGET / "release" / f"lib{NAME}{SUFFIX}"


def build(*, offline=False):
    environment = os.environ | {"PYO3_PYTHON": sys.executable}
    command = [
        "cargo",
        "build",
        "--release",
        "--locked",
        "--manifest-path",
        str(SOURCE / "Cargo.toml"),
        "--target-dir",
        str(TARGET),
    ]
    if offline:
        command.append("--offline")
    subprocess.run(command, env=environment, check=True)


@functools.lru_cache(maxsize=1)
def library():
    if not BINARY.exists():
        raise RuntimeError(
            "Rust probe is not built; run python scripts/profile_af_rust.py --build"
        )
    loader = importlib.machinery.ExtensionFileLoader(NAME, str(BINARY))
    spec = importlib.util.spec_from_file_location(NAME, BINARY, loader=loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _layout(kind):
    if not is_dataclass(kind):
        return None
    return (
        tuple(field.name for field in fields(kind)),
        kind.__dataclass_params__.frozen or getattr(kind, "_af_readonly", False),
    )


@contextmanager
def installed(kernels=("keys", "templates")):
    """Temporarily substitute private kernels in this diagnostic process."""
    import simpy
    from TokenSim.af.caching import estimates, templates
    from TokenSim.af.communication import resources

    if not set(kernels) <= {"keys", "templates", "events", "domains"}:
        raise ValueError("unknown Rust probe kernel")
    if "events" in kernels and simpy.__version__ != "4.1.2":
        raise RuntimeError("Rust event probe requires the tested SimPy 4.1.2 semantics")
    native = library()
    original_memo = estimates.FingerprintMemo
    original_map = vars(estimates).get("map")
    original_bind = templates.DAGTemplate.bind
    original_run = simpy.Environment.run
    original_step = simpy.Environment.step
    original_scheduler = resources.DomainScheduler
    reference_loop = (
        original_run.__module__ == "simpy.core"
        and original_step.__module__ == "simpy.core"
    )

    def native_map(function, values):
        if isinstance(function, native.Fingerprints):
            return function.normalize_many(values)
        return map(function, values)

    def bind(self, block, layer_id, tokens, edges):
        compiled = vars(self).get("_rust_template")
        if compiled is None:
            compiled = native.Template(self.nodes, estimates.FrozenDict)
            object.__setattr__(self, "_rust_template", compiled)
        nodes = compiled.bind(
            f"i{block.iteration_id}:b{block.block_id}:l{layer_id}", tokens, edges
        )
        return templates.BoundDAG(
            nodes, self.predecessor_indices, self.terminal_indices
        )

    event_loop = native.EventLoop() if "events" in kernels else None

    def run(self, until=None):
        if (
            reference_loop
            and type(self) is simpy.Environment
            and getattr(self.step, "__func__", None) is original_step
        ):
            return event_loop.run(self, until)
        return original_run(self, until)

    try:
        if "keys" in kernels:
            estimates.FingerprintMemo = functools.partial(
                native.Fingerprints, _layout, Mapping, estimates.FrozenDict
            )
            estimates.map = native_map
        if "templates" in kernels:
            templates.DAGTemplate.bind = bind
        if "events" in kernels:
            simpy.Environment.run = run
        if "domains" in kernels:
            from scripts.af_rust.domains import scheduler_type

            resources.DomainScheduler = scheduler_type(native)
        yield native
    finally:
        estimates.FingerprintMemo = original_memo
        if original_map is None:
            vars(estimates).pop("map", None)
        else:
            estimates.map = original_map
        templates.DAGTemplate.bind = original_bind
        simpy.Environment.run = original_run
        resources.DomainScheduler = original_scheduler
