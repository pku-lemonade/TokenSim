"""Checked C ABI for the optional compiled native-DAG replay scheduler."""

import ctypes as ct
from dataclasses import dataclass
import fcntl
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.config.validation import integer, number


class NodeSpec(ct.Structure):
    _fields_ = [
        (name, ct.c_int)
        for name in (
            "dep_begin",
            "dep_count",
            "res_begin",
            "res_count",
            "rank",
            "primary",
        )
    ] + [(name, ct.c_double) for name in ("delay", "duration", "occupancy")]


class Completion(ct.Structure):
    _fields_ = [("id", ct.c_longlong)] + [
        (name, ct.c_double) for name in ("ready", "start", "end", "released")
    ]


@lru_cache(maxsize=1)
def library():
    source = Path(__file__).with_suffix(".cpp")
    identity = hashlib.sha256(source.read_bytes()).hexdigest()
    directory = source.parents[2] / ".codex" / "lpx-replay" / identity
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "scheduler.so"
    with (directory / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.exists():
            compiler = shutil.which("g++") or shutil.which("clang++")
            if compiler is None:
                raise ConfigurationError("sampled replay requires a C++17 compiler")
            with tempfile.NamedTemporaryFile(
                dir=directory, suffix=".so", delete=False
            ) as stream:
                temporary = Path(stream.name)
            try:
                result = subprocess.run(
                    [
                        compiler,
                        "-O3",
                        "-std=c++17",
                        "-fPIC",
                        "-shared",
                        str(source),
                        "-o",
                        str(temporary),
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode:
                    raise ConfigurationError(f"replay compiler failed: {result.stderr}")
                os.replace(temporary, binary)
            finally:
                temporary.unlink(missing_ok=True)
    result = ct.CDLL(str(binary))
    pointer = ct.c_void_p
    signatures = {
        "ts_create": ([], pointer),
        "ts_destroy": ([pointer], None),
        "ts_resource": ([pointer], ct.c_int),
        "ts_node_size": ([], ct.c_int),
        "ts_error": ([pointer], ct.c_char_p),
        "ts_window": ([pointer, ct.c_double, ct.c_double], None),
        "ts_add": (
            [
                pointer,
                ct.POINTER(NodeSpec),
                ct.c_int,
                ct.POINTER(ct.c_int),
                ct.c_int,
                ct.POINTER(ct.c_int),
                ct.c_int,
            ],
            ct.c_int,
        ),
        "ts_submit": ([pointer, ct.c_int, ct.c_double], ct.c_longlong),
        "ts_advance": (
            [pointer, ct.c_double, ct.c_int, ct.POINTER(Completion)],
            ct.c_int,
        ),
        "ts_active": ([pointer], ct.c_int),
        "ts_stats": (
            [
                pointer,
                ct.c_int,
                ct.POINTER(ct.c_double),
                ct.POINTER(ct.c_double),
                ct.POINTER(ct.c_longlong),
            ],
            ct.c_int,
        ),
    }
    for name, (inputs, output) in signatures.items():
        function = getattr(result, name)
        function.argtypes, function.restype = inputs, output
    if result.ts_node_size() != ct.sizeof(NodeSpec):
        raise ConfigurationError("replay scheduler ABI size mismatch")
    return result


@dataclass(frozen=True)
class ReplayNode:
    predecessors: tuple
    resources: tuple
    primary: str | None
    delay: float
    duration: float
    occupancy: float
    rank: int = 0

    def __post_init__(self):
        for value in self.predecessors:
            integer(value, "replay predecessor")
        integer(self.rank, "replay tie rank")
        number(self.delay, "replay readiness delay")
        number(self.duration, "replay duration")
        number(self.occupancy, "replay occupancy", maximum=self.duration)
        if any(not isinstance(key, str) or not key for key in self.resources):
            raise ConfigurationError("replay resources require nonempty names")


class ReplayKernel:
    def __init__(self):
        self.handle = None
        self.api = library()
        self.handle = self.api.ts_create()
        self.keys = {}

    def _check(self, value):
        if value < 0:
            raise SimulationStateError(self.api.ts_error(self.handle).decode())
        return value

    def resource(self, name):
        if name not in self.keys:
            self.keys[name] = self.api.ts_resource(self.handle)
        return self.keys[name]

    def add(self, nodes):
        dependencies, keys, specs = [], [], []
        for node in nodes:
            if len(set(node.resources)) != len(node.resources):
                raise ConfigurationError("duplicate replay resource")
            specs.append(
                NodeSpec(
                    len(dependencies),
                    len(node.predecessors),
                    len(keys),
                    len(node.resources),
                    node.rank,
                    -1 if node.primary is None else self.resource(node.primary),
                    node.delay,
                    node.duration,
                    node.occupancy,
                )
            )
            dependencies.extend(node.predecessors)
            keys.extend(self.resource(name) for name in node.resources)
        spec_array = (NodeSpec * len(specs))(*specs)
        dep_array = (ct.c_int * len(dependencies))(*dependencies)
        key_array = (ct.c_int * len(keys))(*keys)
        return self._check(
            self.api.ts_add(
                self.handle,
                spec_array,
                len(specs),
                dep_array,
                len(dependencies),
                key_array,
                len(keys),
            )
        )

    def submit(self, profile, ready):
        return self._check(self.api.ts_submit(self.handle, profile, ready))

    def advance(self, limit=math.inf, priority=4):
        completion = Completion()
        status = self._check(
            self.api.ts_advance(self.handle, limit, priority, ct.byref(completion))
        )
        return completion if status else None

    def window(self, start, end=math.inf):
        self.api.ts_window(self.handle, start, end)

    @property
    def active(self):
        return self.api.ts_active(self.handle)

    def stats(self):
        result = {}
        for name, index in self.keys.items():
            busy, waited, count = ct.c_double(), ct.c_double(), ct.c_longlong()
            self._check(
                self.api.ts_stats(
                    self.handle,
                    index,
                    ct.byref(busy),
                    ct.byref(waited),
                    ct.byref(count),
                )
            )
            result[name] = (busy.value, waited.value, count.value)
        return result

    def close(self):
        if self.handle:
            self.api.ts_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
