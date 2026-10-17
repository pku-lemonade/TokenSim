"""Bind records with one copied field dictionary and one dictionary attachment."""

from contextlib import contextmanager
import inspect
import textwrap

from TokenSim.af.caching.templates import DAGTemplate


@contextmanager
def installed(output=None):
    original = DAGTemplate.bind
    source = textwrap.dedent(inspect.getsource(original))
    source = source.replace(
        "result.__dict__.update(static)", "state = static.copy()"
    ).replace("result.__dict__.update(\n", "state.update(\n")
    old = "        memo[key] = result\n"
    new = "            object.__setattr__(result, '__dict__', state)\n" + old
    if old not in source:
        raise RuntimeError("template binding implementation changed")
    namespace = dict(original.__globals__)
    exec(compile(source.replace(old, new), "<binding-copy-probe>", "exec"), namespace)
    DAGTemplate.bind = namespace[original.__name__]
    try:
        yield
    finally:
        DAGTemplate.bind = original
