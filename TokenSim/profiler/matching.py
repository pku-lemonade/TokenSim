"""Order-preserving matching with explicit ambiguity and semantic group rules."""

from collections import defaultdict
from dataclasses import replace

from TokenSim.errors import ConfigurationError
from TokenSim.profiler.records import canonical, union_duration, intervals_for

EXCLUDED_OPERATIONS = {
    "step_overhead",
    "rank_synchronize",
    "af_completion",
    "kv_transfer",
    "kv_save",
    "kv_dma",
}


def compatible(left, right):
    if left.operation != right.operation:
        return False
    if right.details.get("simulated_members") and not left.members:
        return False
    if left.phase is not None and right.phase is not None and left.phase != right.phase:
        return False
    for key in ("m", "n", "k", "num_tokens", "forward_tokens"):
        if (
            key in left.shape
            and key in right.shape
            and left.shape[key] != right.shape[key]
        ):
            return False
    return True


def _scores(left, right, forbidden=None):
    scores = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) - 1, -1, -1):
        for j in range(len(right) - 1, -1, -1):
            score = max(scores[i + 1][j], scores[i][j + 1])
            if (i, j) != forbidden and compatible(left[i], right[j]):
                score = max(score, 1 + scores[i + 1][j + 1])
            scores[i][j] = score
    return scores


def _align(left, right):
    scores = _scores(left, right)
    optimum = scores[0][0]
    prefix = _scores(list(reversed(left)), list(reversed(right)))
    matches = []
    ambiguous_left, ambiguous_right = set(), set()
    for i, item in enumerate(left):
        for j, other in enumerate(right):
            if not compatible(item, other):
                continue
            before = prefix[len(left) - i][len(right) - j]
            if before + 1 + scores[i + 1][j + 1] != optimum:
                continue
            if _scores(left, right, forbidden=(i, j))[0][0] < optimum:
                matches.append((item, other))
            else:
                ambiguous_left.add(item.id)
                ambiguous_right.add(other.id)
    return matches, ambiguous_left, ambiguous_right


def _merge(members, operation, rule, *, owner=None):
    intervals = intervals_for(members)
    return replace(
        owner or members[0],
        id="group:" + "+".join(item.id for item in members),
        operation=operation,
        start_us=min(start for start, _ in intervals),
        end_us=max(end for _, end in intervals),
        members=tuple(
            identity for item in members for identity in (item.members or (item.id,))
        ),
        kernel_sum_us=sum(item.kernel_sum_us for item in members),
        busy_us=union_duration(intervals),
        details=dict(
            (owner or members[0]).details,
            grouping_rule=rule,
            service_intervals=intervals,
            compute_intervals=intervals_for(
                members, {"compute", "kernel", "gpu_kernel"}
            ),
            communication_intervals=intervals_for(
                members, {"collective", "transfer", "gpu_memcpy"}
            ),
        ),
    )


def _observed_groups(simulated, observed, rank_map):
    targets = {}
    for item in observed:
        members = item.details.get("simulated_members")
        if members:
            key = (
                str(rank_map.get(item.rank, item.rank)),
                item.forward,
                item.layer,
                item.operation,
            )
            targets[key] = members
    windows = defaultdict(list)
    for item in simulated:
        windows[item.rank, item.forward].append(item)
    result = []
    for window in windows.values():
        window.sort(key=lambda item: (item.start_us, item.id))
        index = 0
        while index < len(window):
            pair = window[index : index + 2]
            if len(pair) == 2:
                operations = [item.operation for item in pair]
                for owner in pair:
                    key = owner.rank, owner.forward, owner.layer, owner.operation
                    if targets.get(key) == operations:
                        result.append(
                            _merge(
                                pair,
                                owner.operation,
                                {
                                    "origin": "observed_kernel_fusion",
                                    "simulated": operations,
                                },
                                owner=owner,
                            )
                        )
                        index += 2
                        break
                else:
                    result.append(window[index])
                    index += 1
            else:
                result.append(window[index])
                index += 1
    return result


def _groups(spans, rules):
    """Combine only declared contiguous semantic members within one scope."""
    result = []
    position = 0
    while position < len(spans):
        for rule in rules:
            if not isinstance(rule, dict):
                raise ConfigurationError("group rules must be objects")
            names = rule.get("simulated")
            if (
                not isinstance(names, list)
                or not names
                or not all(isinstance(name, str) for name in names)
                or not isinstance(rule.get("profiled"), str)
            ):
                raise ConfigurationError(
                    "group rules require simulated names and a profiled operation"
                )
            members = spans[position : position + len(names)]
            if [item.operation for item in members] == [
                canonical(name) for name in names
            ]:
                result.append(_merge(members, canonical(rule["profiled"]), rule))
                position += len(members)
                break
        else:
            result.append(spans[position])
            position += 1
    return result


def match_spans(simulated, observed, mapping=None):
    if mapping is None:
        mapping = {}
    if not isinstance(mapping, dict):
        raise ConfigurationError("comparison mapping must be an object")
    if (
        set(mapping) - {"schema_version", "aliases", "rank_map", "groups"}
        or mapping.get("schema_version", 1) != 1
        or type(mapping.get("schema_version", 1)) is not int
    ):
        raise ConfigurationError("unsupported mapping schema or fields")
    rank_map = mapping.get("rank_map", {})
    rules = mapping.get("groups", [])
    if not isinstance(rank_map, dict) or not isinstance(rules, list):
        raise ConfigurationError("rank_map must be an object and groups must be a list")
    simulated = _observed_groups(simulated, observed, rank_map)
    groups = [defaultdict(list), defaultdict(list)]
    excluded = []
    for side, spans in enumerate((simulated, observed)):
        for span in spans:
            if (
                span.operation in EXCLUDED_OPERATIONS
                or span.details.get("run_phase") == "warmup"
                or span.details.get("incomplete")
                or span.category in ("host_submit", "host_prepare", "synchronize")
            ):
                excluded.append(
                    dict(
                        side="simulated" if side == 0 else "profiled", **span.as_dict()
                    )
                )
                continue
            rank = str(rank_map.get(span.rank, span.rank)) if side == 1 else span.rank
            # Missing attribution remains visible and cannot borrow a layer from timing.
            key = rank, span.forward, span.layer
            groups[side][key].append(span)
    matched = []
    ambiguous = [set(), set()]
    normalized = [[], []]
    for key in sorted(set(groups[0]) | set(groups[1]), key=str):
        left = sorted(groups[0].get(key, ()), key=lambda span: (span.start_us, span.id))
        right = sorted(
            groups[1].get(key, ()), key=lambda span: (span.start_us, span.id)
        )
        left = _groups(left, rules)
        normalized[0].extend(left)
        normalized[1].extend(right)
        if key[1] is None or key[2] is None:
            continue
        pairs, a, b = _align(left, right)
        matched.extend(pairs)
        ambiguous[0].update(a)
        ambiguous[1].update(b)
    used = [{pair[side].id for pair in matched} for side in (0, 1)]
    unmatched = []
    for side, spans in enumerate(normalized):
        for span in spans:
            if span.id not in used[side]:
                unmatched.append(
                    dict(
                        side="simulated" if side == 0 else "profiled",
                        status=(
                            "ambiguous" if span.id in ambiguous[side] else "unmatched"
                        ),
                        **span.as_dict(),
                    )
                )
    return matched, unmatched, excluded
