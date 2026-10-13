"""Bounded CPU-only native scheduler/output intervals; no whole-step residual."""
import functools
import hashlib
import inspect
import time

active = False
events = []
sources = {}


def wrap(owner, name, label):
    original = getattr(owner, name)
    source = inspect.unwrap(original)
    sources[label] = dict(file=inspect.getsourcefile(source),
        line=inspect.getsourcelines(source)[1],
        sha256=hashlib.sha256(inspect.getsource(source).encode()).hexdigest())

    @functools.wraps(original)
    def measured(*args, **kwargs):
        if not active:
            return original(*args, **kwargs)
        start = time.perf_counter_ns()
        result = original(*args, **kwargs)
        finish = time.perf_counter_ns()
        events.append(dict(name=label,start_ns=start,end_ns=finish))
        return result
    setattr(owner, name, measured)


def install():
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.output_processor import OutputProcessor
    wrap(Scheduler,'schedule','schedule')
    wrap(Scheduler,'update_from_output','update_from_output')
    wrap(OutputProcessor,'process_outputs','process_outputs')


def finish_step():
    if sorted(e['name'] for e in events) != ['process_outputs','schedule','update_from_output']:
        raise ValueError('expected exactly one CPU invocation of each audited function')
    ordered = sorted(events,key=lambda e:e['start_ns'])
    if any(left['end_ns'] > right['start_ns'] for left,right in zip(ordered,ordered[1:])):
        raise ValueError('CPU intervals overlap')
    return {e['name']:(e['end_ns']-e['start_ns'])/1000 for e in events}
