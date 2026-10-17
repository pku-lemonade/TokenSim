"""Diagnostic bulk initialization for the wide frozen trace record."""

from contextlib import contextmanager
from dataclasses import MISSING, fields
from functools import update_wrapper
from inspect import Parameter, signature

from TokenSim.af.trace.metrics import AFTraceEvent


def initializer(record_type):
    original = record_type.__init__
    namespace = {}
    parameters = []
    for index, parameter in enumerate(signature(original).parameters.values()):
        declaration = parameter.name
        if parameter.default is not Parameter.empty:
            symbol = f"_default_{index}"
            namespace[symbol] = parameter.default
            declaration += "=" + symbol
        parameters.append(declaration)
    assignments = []
    for index, field in enumerate(fields(record_type), 1):
        expression = field.name
        if field.default_factory is not MISSING:
            symbol = f"_factory_{index}"
            namespace[symbol] = field.default_factory
            expression = (
                f"{symbol}() if {field.name} is _default_{index} else {field.name}"
            )
        assignments.append(f"{field.name!r}: {expression}")
    source = (
        f"def __init__({', '.join(parameters)}):\n"
        f"    self.__dict__.update({{{', '.join(assignments)}}})\n"
    )
    exec(compile(source, "<trace-bulk-initializer>", "exec"), namespace)
    return update_wrapper(namespace["__init__"], original)


@contextmanager
def installed(output=None):
    original = AFTraceEvent.__init__
    AFTraceEvent.__init__ = initializer(AFTraceEvent)
    try:
        yield
    finally:
        AFTraceEvent.__init__ = original
