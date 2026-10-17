"""Diagnostic dependency wait preserving the two original scheduling turns."""

from contextlib import contextmanager
import inspect
import textwrap

from TokenSim.af.execution.iteration import ExecutorIterationMixin


def dependency_barrier(env, predecessors, failure):
    complete, ready = env.event(), env.event()
    remaining = len(predecessors)

    def forward(event):
        if ready.triggered:
            return
        if event.ok:
            ready.succeed()
        else:
            event.defused = True
            ready.fail(event.value)

    def check(event):
        nonlocal remaining
        if complete.triggered:
            return
        if not event.ok:
            event.defused = True
            complete.fail(event.value)
        else:
            remaining -= 1
            if remaining == 0:
                complete.succeed()

    def cleanup(event):
        if failure.callbacks is not None and forward in failure.callbacks:
            failure.callbacks.remove(forward)
        if complete.callbacks is not None and forward in complete.callbacks:
            complete.callbacks.remove(forward)
        for predecessor in predecessors:
            if predecessor.callbacks is not None and check in predecessor.callbacks:
                predecessor.callbacks.remove(check)

    complete.callbacks.append(forward)
    ready.callbacks.append(cleanup)
    for predecessor in predecessors:
        if predecessor.callbacks is None:
            check(predecessor)
        else:
            predecessor.callbacks.append(check)
    if failure.callbacks is None:
        forward(failure)
    else:
        failure.callbacks.append(forward)
    return ready


@contextmanager
def installed(output=None):
    original = ExecutorIterationMixin._run_dag_node_guarded
    source = textwrap.dedent(inspect.getsource(original))
    old = (
        "predecessor_done = self.env.all_of(predecessors)\n"
        "            yield self.env.any_of((predecessor_done, failure_event))"
    )
    replacement = "yield dependency_barrier(self.env, predecessors, failure_event)"
    if old not in source:
        raise RuntimeError("dependency-wait implementation changed")
    namespace = dict(original.__globals__, dependency_barrier=dependency_barrier)
    exec(
        compile(source.replace(old, replacement), "<dependency-probe>", "exec"),
        namespace,
    )
    ExecutorIterationMixin._run_dag_node_guarded = namespace[original.__name__]
    try:
        yield
    finally:
        ExecutorIterationMixin._run_dag_node_guarded = original
