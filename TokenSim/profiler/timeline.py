"""Select comparison intervals without changing durations or clock relationships."""

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import math

from TokenSim.errors import ConfigurationError
from TokenSim.profiler.annotations import kernel_operation


@dataclass(frozen=True)
class TimelineBar:
    side: str
    rank: str
    clock: str
    track: str
    start_us: float
    end_us: float
    name: str
    operation: str
    category: str
    span_id: str
    status: str
    intervals: tuple
    waits: tuple
    translation_us: float = 0


def _number(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ConfigurationError("timeline timestamps must be finite numbers")
    return value


def _intervals(values):
    if not isinstance(values, (list, tuple)):
        raise ConfigurationError("timeline intervals must be arrays")
    result = []
    for interval in values:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ConfigurationError("timeline interval must have two timestamps")
        start, end = map(_number, interval)
        if end < start:
            raise ConfigurationError("timeline interval ends before it starts")
        result.append((start, end))
    return tuple(result)


def report_spans(report):
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ConfigurationError("timeline requires comparison schema_version 1")
    result, seen = [], set()
    for field in ("matches", "unmatched", "excluded"):
        rows = report.get(field, [])
        if not isinstance(rows, list):
            raise ConfigurationError(f"comparison {field} must be an array")
        for row in rows:
            if not isinstance(row, dict):
                raise ConfigurationError("comparison entries must be objects")
            candidates = (
                [(side, row.get(side)) for side in ("simulated", "profiled")]
                if field == "matches"
                else [(row.get("side"), row)]
            )
            for side, span in candidates:
                if side not in {"simulated", "profiled"} or not isinstance(span, dict):
                    raise ConfigurationError("comparison entry has no valid side/span")
                if any(key not in span for key in ("id", "rank", "clock", "operation")):
                    raise ConfigurationError("comparison span lacks identity fields")
                if not isinstance(span["clock"], str):
                    raise ConfigurationError("comparison span clock must be a string")
                identity = side, str(span["id"])
                if identity not in seen:
                    seen.add(identity)
                    result.append(
                        (
                            side,
                            span,
                            (
                                "matched"
                                if field == "matches"
                                else row.get("status", field)
                            ),
                        )
                    )
    return result


def _span_bars(side, span, status, detail):
    start, end = _intervals([(span.get("start_us"), span.get("end_us"))])[0]
    details = span.get("details", {})
    if not isinstance(details, dict):
        raise ConfigurationError("comparison span details must be an object")
    intervals = _intervals(details.get("service_intervals", [(start, end)]))
    waits = _intervals(details.get("collective_wait_intervals", []))
    if any(a < start - 0.01 or b > end + 0.01 for a, b in intervals + waits):
        raise ConfigurationError("device interval lies outside its module span")
    base = dict(
        side=side,
        rank=str(span["rank"]),
        clock=str(span["clock"]),
        track="modules" if detail == "module" else "operators",
        start_us=start,
        end_us=end,
        name=str(span["operation"]),
        operation=str(span["operation"]),
        category=str(span.get("category", "compute")),
        span_id=str(span["id"]),
        status=status,
        intervals=intervals,
        waits=waits,
    )
    if detail == "module":
        return [TimelineBar(**base)]
    if side == "simulated":
        grouping = details.get("grouping_rule", {})
        if not isinstance(grouping, dict):
            raise ConfigurationError("comparison grouping_rule must be an object")
        names = grouping.get("simulated", [])
        if not isinstance(names, list):
            raise ConfigurationError("simulated grouping names must be an array")
        names = (
            names if len(names) == len(intervals) else [base["name"]] * len(intervals)
        )
        activities = [
            dict(name=name, start_us=a, end_us=b, category=base["category"])
            for name, (a, b) in zip(names, intervals)
        ]
    else:
        activities = details.get("device_activities")
        if activities is None:
            names = details.get("kernel_names", [])
            if not isinstance(names, list) or len(names) != len(intervals):
                raise ConfigurationError("kernel view requires retained activity names")
            activities = [
                dict(name=name, start_us=a, end_us=b, category=base["category"])
                for name, (a, b) in zip(names, intervals)
            ]
    if not isinstance(activities, list) or not all(
        isinstance(a, dict) for a in activities
    ):
        raise ConfigurationError("device_activities must be an array of objects")
    activity_intervals = _intervals(
        [(activity.get("start_us"), activity.get("end_us")) for activity in activities]
    )
    if activity_intervals != intervals:
        raise ConfigurationError(
            "activity timestamps differ from retained service intervals"
        )
    bars = []
    for activity in activities:
        a, b = _intervals([(activity.get("start_us"), activity.get("end_us"))])[0]
        if a < start - 0.01 or b > end + 0.01:
            raise ConfigurationError("activity lies outside its module span")
        name = str(activity.get("name", base["name"]))
        operation = kernel_operation(name) or base["operation"]
        category = str(activity.get("category", base["category"]))
        if category == "gpu_memcpy" or name.lower().startswith("memcpy"):
            category = "gpu_memcpy"
        stream = activity.get("stream")
        track = (
            "operators"
            if side == "simulated"
            else (f"stream {stream}" if stream is not None else "stream unavailable")
        )
        bars.append(
            TimelineBar(
                **dict(
                    base,
                    start_us=a,
                    end_us=b,
                    name=name,
                    operation=operation,
                    category=category,
                    track=track,
                    intervals=((a, b),),
                    waits=tuple(
                        (max(a, x), min(b, y)) for x, y in waits if x < b and y > a
                    ),
                )
            )
        )
    return bars


def prepare_timeline(
    report,
    *,
    forward=None,
    layer=None,
    ranks=None,
    detail="kernel",
    alignment="global",
    start_us=None,
    end_us=None,
):
    if detail not in {"kernel", "module"} or alignment not in {"global", "selection"}:
        raise ConfigurationError("invalid timeline detail or alignment")
    for name, value, minimum in (("forward", forward, 0), ("layer", layer, -1)):
        if value is not None and (type(value) is not int or value < minimum):
            raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    spans = report_spans(report)
    clock_alignment = report.get("alignment", {})
    if not isinstance(clock_alignment, dict):
        raise ConfigurationError("comparison alignment must be an object")
    anchors = clock_alignment.get("anchors", {})
    if not isinstance(anchors, dict):
        raise ConfigurationError("comparison alignment anchors must be an object")
    shared = clock_alignment.get("cross_rank_overlap_available", False)
    if type(shared) is not bool:
        raise ConfigurationError("cross_rank_overlap_available must be a boolean")
    shifts, warnings = {}, []
    for side, span, _ in spans:
        if side == "profiled" and span["clock"] not in shifts:
            clock = span["clock"]
            anchor = anchors.get(clock)
            if anchor is not None:
                if not isinstance(anchor, dict):
                    raise ConfigurationError(
                        "comparison clock anchor must be an object"
                    )
                shifts[clock] = _number(anchor.get("translation_us"))
            else:
                shifts[clock] = -min(
                    _number(s.get("start_us"))
                    for k, s, _ in spans
                    if k == side and s["clock"] == clock
                )
                warnings.append(
                    f"{clock} has no matched anchor; its own origin is used"
                )
    if shared and shifts:
        known = {value for clock, value in shifts.items() if clock in anchors}
        if len(known) > 1:
            raise ConfigurationError(
                "shared profile clocks have inconsistent translations"
            )
        shared_shift = next(iter(known)) if known else max(shifts.values())
        for clock in shifts.keys() - anchors.keys():
            shifts[clock] = shared_shift
        warnings = [
            note.replace("its own origin is used", "the shared profile origin is used")
            for note in warnings
        ]
    if isinstance(ranks, str):
        ranks = [ranks]
    ranks = {str(rank) for rank in ranks} if ranks is not None else None
    bars = []
    for side, span, status in spans:
        if (
            forward is not None
            and span.get("forward") != forward
            or layer is not None
            and span.get("layer") != layer
            or ranks is not None
            and str(span["rank"]) not in ranks
        ):
            continue
        for bar in _span_bars(side, span, status, detail):
            bars.append(replace(bar, translation_us=shifts.get(bar.clock, 0)))
    if not bars:
        raise ConfigurationError("no intervals match the timeline selection")
    if len({bar.clock for bar in bars if bar.side == "profiled"}) > 1 and not shared:
        warnings.append(
            "Profile clocks are independently aligned; cross-rank overlap is unavailable"
        )
    origin = min(bar.start_us + bar.translation_us for bar in bars)
    origins = {}
    for bar in bars:
        key = bar.side if shared or bar.side == "simulated" else bar.clock
        value = bar.start_us + bar.translation_us
        origins[key] = min(origins.get(key, value), value)
    adjusted = []
    for bar in bars:
        key = bar.side if shared or bar.side == "simulated" else bar.clock
        offset = origins[key] if alignment == "selection" else origin
        adjusted.append(replace(bar, translation_us=bar.translation_us - offset))
    lower = min(bar.start_us + bar.translation_us for bar in adjusted)
    upper = max(bar.end_us + bar.translation_us for bar in adjusted)
    lower = _number(start_us) if start_us is not None else lower
    upper = _number(end_us) if end_us is not None else upper
    if upper <= lower:
        raise ConfigurationError("timeline end must be greater than its start")
    visible = [
        bar
        for bar in adjusted
        if bar.start_us + bar.translation_us <= upper
        and bar.end_us + bar.translation_us >= lower
    ]
    if not visible:
        raise ConfigurationError("no intervals intersect the timeline window")
    return dict(
        bars=visible,
        start_us=lower,
        end_us=upper,
        warnings=warnings,
        alignment=alignment,
        detail=detail,
        filters=dict(
            forward=forward, layer=layer, ranks=sorted(ranks) if ranks else None
        ),
        origins_us=origins if alignment == "selection" else {"global": origin},
    )


def pack_tracks(bars, side):
    """Keep real streams separate and retain overlapping intervals on subtracks."""
    groups = defaultdict(list)
    for bar in bars:
        if bar.side == side:
            groups[bar.rank, bar.clock, bar.track].append(bar)
    result = []
    for key, values in sorted(groups.items()):
        tracks = []
        for bar in sorted(values, key=lambda b: (b.start_us, b.end_us, b.span_id)):
            for track in tracks:
                end = track[-1].end_us
                # Clock arithmetic can place adjacent endpoints a few ULPs apart.
                # Packing ignores that roundoff without rewriting any timestamps.
                tolerance = max(1e-9, 4 * math.ulp(end), 4 * math.ulp(bar.start_us))
                if end <= bar.start_us + tolerance:
                    track.append(bar)
                    break
            else:
                tracks.append([bar])
        for index, track in enumerate(tracks):
            label = f"rank {key[0]} / {key[2]}"
            if len({clock for rank, clock, _ in groups if rank == key[0]}) > 1:
                label += f" / {key[1]}"
            if len(tracks) > 1:
                label += f" / overlap {index + 1}"
            result.append((label, track))
    return result


def bar_record(bar):
    return dict(
        asdict(bar),
        display_start_us=bar.start_us + bar.translation_us,
        display_end_us=bar.end_us + bar.translation_us,
    )
