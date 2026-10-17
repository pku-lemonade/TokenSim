"""Configurable fixed-batch AF experiments over the existing model runtime."""

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from TokenSim.af import factory
from TokenSim.af.trace.config import AFTraceConfig
from TokenSim.af.trace.export import write_chrome_trace
from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError
from TokenSim.experiments import ExperimentRunner
from TokenSim.experiments.fingerprints import identity


def _trace_config(value):
    if isinstance(value, AFTraceConfig):
        return value
    return AFTraceConfig.from_mapping(value)


def _write_profile(path, executor, metadata=None):
    return write_chrome_trace(
        path,
        executor.trace.events,
        metadata,
        trace_config=getattr(executor.trace, "trace_config", None),
    )


class AFExperimentRunner:
    """Run any AFModelRuntime with per-point trace choices and durable reuse.

    Completed independent cases are checkpoints. An interrupted case restarts
    from its initial seed; a live SimPy environment is never serialized.
    Runtime caches, collection policy and prefix export belong to the shared
    factory and are used without additional optimization switches.
    """

    def __init__(
        self,
        runtime,
        output,
        *,
        trace_config=None,
        fallback="table_first",
        seed=42,
        resume=True,
        verify="stat",
        inputs=(),
        code_paths=None,
        profile_metadata=None,
        simulate=None,
        write_profile=None,
    ):
        self.runtime = runtime
        self.trace_config = (
            AFTraceConfig() if trace_config is None else _trace_config(trace_config)
        )
        self.fallback = fallback
        self.seed = integer(seed, "experiment seed", minimum=0)
        self.profile_metadata = dict(profile_metadata or {})
        self._simulate = simulate
        self._write_profile = write_profile or _write_profile
        self._inputs = tuple(inputs)
        self.runner = ExperimentRunner(
            output, resume=resume, verify=verify, code_paths=code_paths
        )

    def _runtime_identity(self, fallback):
        runtime, hardware = self.runtime, self.runtime.hardware
        devices = sorted({worker.hardware for worker in runtime.workers})
        packages = []
        if fallback != "analytical_only":
            # Loaded on-disk packages are immutable snapshots too. Include
            # actual table values so manually registered or modified packages
            # cannot be confused with a cold disk load.
            for device in devices:
                backends = {
                    worker.operator_backend
                    for worker in runtime.workers
                    if worker.hardware == device
                }
                for backend in sorted(backends, key=str):
                    package = hardware.operator_package(device, backend)
                    if package is not None:
                        packages.append(
                            dict(
                                device=device,
                                backend=package.backend,
                                meta=package.meta.to_dict(),
                                tables=package.tables,
                                analysis=package.analysis,
                            )
                        )
        return identity(
            dict(
                model=runtime.model,
                doc=runtime.doc,
                cluster=runtime.cluster,
                workers=runtime.workers,
                plan=runtime.plan,
                placement=runtime.placement,
                devices=[hardware.device(device).to_dict() for device in devices],
                links=list(hardware.links),
                packages=packages,
            )
        )

    def run(
        self,
        batch_size,
        context_len,
        *,
        steps=1,
        trace_config=None,
        fallback=None,
        seed=None,
        resume=None,
    ):
        """Return an ExperimentResult with summary and per-iteration timing rows."""
        batch = integer(batch_size, "experiment batch_size", minimum=1)
        context = integer(context_len, "experiment context_len", minimum=1)
        steps = integer(steps, "experiment steps", minimum=1)
        seed = (
            self.seed if seed is None else integer(seed, "experiment seed", minimum=0)
        )
        fallback = self.fallback if fallback is None else fallback
        if fallback not in ("table_first", "table_only", "analytical_only"):
            raise ConfigurationError("invalid experiment fallback policy")
        options = (
            self.trace_config if trace_config is None else _trace_config(trace_config)
        )
        inputs = list(self._inputs)
        data = self.runtime.hardware.operator_data_root
        if data is not None and fallback != "analytical_only":
            inputs.extend(data / worker.hardware for worker in self.runtime.workers)
        self.runner.inputs = tuple(Path(path) for path in inputs)
        parameters = dict(
            runtime=self._runtime_identity(fallback),
            batch_size=batch,
            context_len=context,
            steps=steps,
            seed=seed,
            fallback=fallback,
            trace_config=asdict(options),
            profile_metadata=self.profile_metadata,
        )

        def execute(directory):
            trace_path = (
                directory
                / f"decode-layers{self.runtime.plan.total_layers}-b{batch}.trace.json"
            )
            kwargs = {"trace_config": options} if options != AFTraceConfig() else {}
            # The original seed default keeps diagnostic wrappers compatible.
            if seed != 42:
                kwargs["seed"] = seed
            executor, measurements, compute, transfer = (
                self._simulate or factory.simulate_decode
            )(
                self.runtime,
                batch,
                context,
                steps=steps,
                trace=True,
                fallback=fallback,
                trace_output=trace_path,
                **kwargs,
            )
            try:
                step_us = sum(row["makespan_us"] for row in measurements) / steps
                profile = (
                    self._write_profile(
                        trace_path,
                        executor,
                        metadata=self.profile_metadata,
                    )
                    if options.mode == "full"
                    else None
                )
                summary = dict(
                    batch_size=batch,
                    context_len=context,
                    steps=steps,
                    avg_step_us=step_us,
                    tokens_per_s=batch / (step_us / 1e6),
                    tokens_per_s_per_user=1e6 / step_us,
                    profile=profile.name if profile is not None else None,
                    operator_stats=compute.stats_dict(),
                    transfer=transfer.describe(),
                )
                return dict(
                    summary=summary,
                    iterations=measurements,
                    trace_config=options.describe(),
                )
            finally:
                executor.trace.close()

        return self.runner.run(parameters, execute, resume=resume)

    def run_many(self, cases):
        """Evaluate a sequence of run() keyword mappings in the supplied order."""
        results = []
        for case in cases:
            if not isinstance(case, Mapping):
                raise ConfigurationError("experiment cases must be keyword mappings")
            results.append(self.run(**case))
        return results
