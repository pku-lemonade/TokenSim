"""Diagnostic admission reuse, excluded because it changes the DAG type.

Source-node reuse also loses static sharing between admission and cache hits.
This prototype is intentionally absent from production and contract tests.
"""

from contextlib import contextmanager
import inspect
import textwrap

from TokenSim.af.caching.templates import TemplateCache


@contextmanager
def installed(output=None):
    original = TemplateCache.build
    source = textwrap.dedent(inspect.getsource(original))
    old = "            self.templates.popitem(last=False)\n    else:"
    new = (
        "            self.templates.popitem(last=False)\n        return dag\n    else:"
    )
    if old not in source:
        raise RuntimeError("template compilation implementation changed")
    namespace = dict(original.__globals__)
    exec(compile(source.replace(old, new), "<cold-binding-probe>", "exec"), namespace)
    TemplateCache.build = namespace[original.__name__]
    try:
        yield
    finally:
        TemplateCache.build = original
