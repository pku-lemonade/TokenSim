"""Diagnostic straight-line Python binding compiled from existing recipes."""

from contextlib import contextmanager

from TokenSim.af.caching.estimates import FrozenDict
from TokenSim.af.caching.templates import BoundDAG, DAGTemplate


def compile_binding(template):
    constants, expressions, lines = [], {}, []

    def constant(value):
        index = len(constants)
        constants.append(value)
        return f"_constants[{index}]"

    def visit(recipe):
        key = id(recipe)
        if key in expressions:
            return expressions[key]
        if recipe.kind == "static":
            expression = constant(recipe.value)
        else:
            name = f"v{len(expressions)}"
            if recipe.kind == "prefix":
                expression = f"prefix + {constant(recipe.value)}"
            elif recipe.kind == "token":
                index, suffix = recipe.value
                expression = (
                    f"f'request:{{tokens[{index}][0]}}:token:{{tokens[{index}][1]}}'"
                    f" + {constant(suffix)}"
                )
            elif recipe.kind == "edge":
                expression = f"edges[{recipe.value}]"
            elif recipe.kind == "tuple":
                children = [visit(child) for child in recipe.children]
                expression = "(" + ",".join(children) + ("," if children else "") + ")"
            else:
                children = [
                    f"{field!r}: {visit(child)}" for field, child in recipe.children
                ]
                mapping = "{" + ",".join(children) + "}"
                if recipe.kind == "mapping":
                    expression = f"_mapping({mapping})"
                else:
                    record_type, static = recipe.value
                    # Children have already received variable names.
                    name = f"v{len(expressions)}"
                    lines.extend(
                        [
                            f"{name} = _new({constant(record_type)})",
                            f"{name}.__dict__.update({constant(static)})",
                            f"{name}.__dict__.update({mapping})",
                        ]
                    )
                    expression = name
            if expression != name:
                name = f"v{len(expressions)}"
                lines.append(f"{name} = {expression}")
                expression = name
        expressions[key] = expression
        return expression

    try:
        outputs = [visit(node) for node in template.nodes]
    finally:
        del visit
    lines.append("return (" + ",".join(outputs) + ("," if outputs else "") + ")")
    source = "def bind(prefix, tokens, edges):\n    " + "\n    ".join(lines)
    namespace = {
        "_constants": constants,
        "_new": object.__new__,
        "_mapping": FrozenDict,
    }
    exec(compile(source, "<compiled-dag-binding>", "exec"), namespace)
    return namespace["bind"]


@contextmanager
def installed(output=None):
    original = DAGTemplate.bind

    def bind(self, block, layer_id, tokens, edges):
        method = vars(self).get("_probe_compiled_bind")
        if method is None:
            method = compile_binding(self)
            object.__setattr__(self, "_probe_compiled_bind", method)
        return BoundDAG(
            method(
                f"i{block.iteration_id}:b{block.block_id}:l{layer_id}", tokens, edges
            ),
            self.predecessor_indices,
            self.terminal_indices,
        )

    DAGTemplate.bind = bind
    try:
        yield
    finally:
        DAGTemplate.bind = original
