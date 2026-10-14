"""Validated structural DAG templates with per-execution identity bindings.

The cache is local to one planner/runtime plan. Exact token counts and routing
incidence are shape classes: a changed count or branch compiles another template.
Context affects provider queries, not static planning. No SimPy state lives here.
"""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass, replace
import re

from TokenSim.af.caching.estimates import FrozenDict, identity
from TokenSim.af.providers.queries import AFRoutingEdge


@dataclass(frozen=True)
class Recipe:
    kind: str
    value: object
    children: tuple = ()


@dataclass(frozen=True)
class BoundDAG:
    nodes: tuple
    predecessor_indices: tuple
    terminal_indices: tuple

    @property
    def terminal_node_ids(self):
        return tuple(self.nodes[index].node_id for index in self.terminal_indices)


@dataclass(frozen=True)
class DAGTemplate:
    nodes: tuple
    predecessor_indices: tuple
    terminal_indices: tuple

    @classmethod
    def compile(cls, dag, block, layer_id, tokens, edges):
        prefix = f"i{block.iteration_id}:b{block.block_id}:l{layer_id}"
        token_indices = {token: index for index, token in enumerate(tokens)}
        edge_indices = {edge: index for index, edge in enumerate(edges)}
        memo = {}

        def compile_value(value):
            cached = memo.get(id(value))
            if cached is not None:
                return cached
            if isinstance(value, AFRoutingEdge):
                recipe = Recipe("edge", edge_indices[value])
            elif isinstance(value, str) and value.startswith(prefix + ":"):
                recipe = Recipe("prefix", value[len(prefix) :])
            elif (
                isinstance(value, str)
                and value.startswith("request:")
                and (
                    match := re.fullmatch(r"request:(-?\d+):token:(-?\d+)(:.*)", value)
                )
            ):
                recipe = Recipe(
                    "token", (token_indices[(int(match[1]), int(match[2]))], match[3])
                )
            elif is_dataclass(value):
                children = tuple(
                    (name, compile_value(item)) for name, item in vars(value).items()
                )
                if all(child.kind == "static" for _, child in children):
                    recipe = Recipe("static", value)
                else:
                    static = FrozenDict(
                        {
                            name: child.value
                            for name, child in children
                            if child.kind == "static"
                        }
                    )
                    dynamic = tuple(
                        (name, child)
                        for name, child in children
                        if child.kind != "static"
                    )
                    recipe = Recipe("record", (type(value), static), dynamic)
            elif isinstance(value, Mapping):
                children = tuple(
                    (key, compile_value(item)) for key, item in value.items()
                )
                recipe = (
                    Recipe("static", FrozenDict({k: v.value for k, v in children}))
                    if all(v.kind == "static" for _, v in children)
                    else Recipe("mapping", None, children)
                )
            elif isinstance(value, tuple):
                children = tuple(compile_value(item) for item in value)
                recipe = (
                    Recipe("static", tuple(child.value for child in children))
                    if all(child.kind == "static" for child in children)
                    else Recipe("tuple", None, children)
                )
            else:
                recipe = Recipe("static", value)
            memo[id(value)] = recipe
            return recipe

        indices = {node.node_id: i for i, node in enumerate(dag.nodes)}
        return cls(
            tuple(compile_value(node) for node in dag.nodes),
            tuple(
                tuple(indices[p] for p in node.predecessor_ids) for node in dag.nodes
            ),
            tuple(indices[node_id] for node_id in dag.terminal_node_ids),
        )

    def bind(self, block, layer_id, tokens, edges):
        prefix = f"i{block.iteration_id}:b{block.block_id}:l{layer_id}"
        memo = {}

        def bind_value(recipe):
            if recipe.kind == "static":
                return recipe.value
            key = id(recipe)
            if key in memo:
                return memo[key]
            if recipe.kind == "prefix":
                result = prefix + recipe.value
            elif recipe.kind == "token":
                index, suffix = recipe.value
                request, token = tokens[index]
                result = f"request:{request}:token:{token}{suffix}"
            elif recipe.kind == "edge":
                result = edges[recipe.value]
            elif recipe.kind == "tuple":
                result = tuple(bind_value(child) for child in recipe.children)
            elif recipe.kind == "mapping":
                result = FrozenDict(
                    {name: bind_value(child) for name, child in recipe.children}
                )
            else:
                # Geometry, ownership and dependency validation ran at compile.
                # The key guarantees identical structure; only validated routing
                # identities and token strings are substituted at bind time.
                record_type, static = recipe.value
                result = object.__new__(record_type)
                result.__dict__.update(static)
                result.__dict__.update(
                    (name, bind_value(child)) for name, child in recipe.children
                )
            memo[key] = result
            return result

        return BoundDAG(
            tuple(bind_value(node) for node in self.nodes),
            self.predecessor_indices,
            self.terminal_indices,
        )


class TemplateCache:
    def __init__(self, limit=128):
        self.limit = limit
        self.templates = OrderedDict()
        self.requests = 0
        self.hits = 0
        self.static_fingerprints = {}
        self._configuration_fingerprint = None
        self._layer_fingerprints = {}
        self._link_fingerprints = {}
        self._static_classes = {}

    def build(self, planner, block, layer_id, link):
        self.requests += 1
        layer = planner.runtime_plan.layer_plan(layer_id)
        # The planner pins a read-only runtime configuration. Fingerprints are
        # scoped to this plan, and compiled only for layers/links actually used.
        static_key = (id(planner.runtime_plan), layer_id, link.name)
        static = self.static_fingerprints.get(static_key)
        if static is None:
            if self._configuration_fingerprint is None:
                self._configuration_fingerprint = identity(planner.cluster_config.af)
            if layer_id not in self._layer_fingerprints:
                self._layer_fingerprints[layer_id] = identity(
                    replace(layer, layer_id=0)
                )
            if link.name not in self._link_fingerprints:
                self._link_fingerprints[link.name] = identity(link)
            static = (
                self._layer_fingerprints[layer_id],
                self._configuration_fingerprint,
                self._link_fingerprints[link.name],
            )
            # Intern with full equality once. Hot keys hash only this small
            # integer; unlike a bare hash, interning cannot alias collisions.
            static = self._static_classes.setdefault(static, len(self._static_classes))
            self.static_fingerprints[static_key] = static
        snapshot = block.routing_snapshot_for_layer(layer_id)
        # Let the reference planner diagnose invalid layer/snapshot contracts.
        if (layer.layer_type == "moe") != (snapshot is not None):
            return planner._build_runtime_layer(block, layer_id, link)
        tokens = (
            tuple(snapshot.original_tokens)
            if snapshot
            else planner._block_tokens(block)
        )
        token_indices = {token: index for index, token in enumerate(tokens)}
        edges = (
            tuple(planner._routing_edge(a) for a in snapshot.assignments)
            if snapshot
            else ()
        )
        route_key = tuple(
            (
                token_indices[(e.request_id, e.token_id)],
                e.top_k_index,
                e.expert_id,
                e.routing_weight,
                e.a_dp_rank,
                e.a_tp_rank,
            )
            for e in edges
        )
        # Lexical token ordering affects stable endpoint-copy ordering too.
        ordering = tuple(
            sorted(
                range(len(tokens)),
                key=lambda i: f"request:{tokens[i][0]}:token:{tokens[i][1]}:",
            )
        )
        key = (
            id(planner.runtime_plan),
            block.phase,
            static,
            block.placement,
            block.token_count,
            block.request_count,
            ordering,
            route_key,
        )
        template = self.templates.get(key)
        if template is None:
            dag = planner._build_runtime_layer(block, layer_id, link)
            template = DAGTemplate.compile(dag, block, layer_id, tokens, edges)
            self.templates[key] = template
            if len(self.templates) > self.limit:
                self.templates.popitem(last=False)
        else:
            self.hits += 1
            self.templates.move_to_end(key)
        return template.bind(block, layer_id, tokens, edges)
