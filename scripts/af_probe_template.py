"""Opt-in postorder template binder without recursive per-bind memo lookup."""

from contextlib import contextmanager

from TokenSim.af.caching.estimates import FrozenDict
from TokenSim.af.caching.templates import BoundDAG, DAGTemplate


class BindingProgram:
    def __init__(self, template):
        self.seeds = []
        self.instructions = []
        positions = {}

        def visit(recipe):
            key = id(recipe)
            if key in positions:
                return positions[key]
            kind, value = recipe.kind, recipe.value
            if kind in ("record", "mapping"):
                children = tuple(
                    (name, visit(child)) for name, child in recipe.children
                )
            elif kind == "tuple":
                children = tuple(visit(child) for child in recipe.children)
            else:
                children = ()
            index = len(self.seeds)
            positions[key] = index
            self.seeds.append(value if kind == "static" else None)
            if kind != "static":
                self.instructions.append((index, kind, value, children))
            return index

        try:
            self.outputs = tuple(visit(recipe) for recipe in template.nodes)
        finally:
            del visit

    def bind(self, prefix, tokens, edges):
        values = self.seeds.copy()
        for index, kind, value, children in self.instructions:
            if kind == "prefix":
                result = prefix + value
            elif kind == "token":
                token_index, suffix = value
                request, token = tokens[token_index]
                result = f"request:{request}:token:{token}{suffix}"
            elif kind == "edge":
                result = edges[value]
            elif kind == "tuple":
                result = tuple(values[child] for child in children)
            elif kind == "mapping":
                result = FrozenDict({name: values[child] for name, child in children})
            elif kind == "record":
                record_type, static = value
                result = object.__new__(record_type)
                result.__dict__.update(static)
                result.__dict__.update(
                    (name, values[child]) for name, child in children
                )
            else:
                raise ValueError(f"unknown template recipe: {kind}")
            values[index] = result
        return tuple(values[index] for index in self.outputs)


@contextmanager
def installed(output=None):
    original = DAGTemplate.bind

    def bind(self, block, layer_id, tokens, edges):
        program = vars(self).get("_probe_binding_program")
        if program is None:
            program = BindingProgram(self)
            object.__setattr__(self, "_probe_binding_program", program)
        prefix = f"i{block.iteration_id}:b{block.block_id}:l{layer_id}"
        return BoundDAG(
            program.bind(prefix, tokens, edges),
            self.predecessor_indices,
            self.terminal_indices,
        )

    DAGTemplate.bind = bind
    try:
        yield
    finally:
        DAGTemplate.bind = original
