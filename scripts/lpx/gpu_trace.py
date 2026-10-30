"""Partition explicitly selected decode windows without double-counting GPU work."""

from collections import Counter, defaultdict
import math

from TokenSim.errors import WorkloadValidationError

CATEGORIES = ("transformer_layers", "vocabulary_head", "logits_sampling")
PARTITIONS = (
    *CATEGORIES,
    "unclassified_gpu",
    "gpu_category_overlap",
    "no_gpu_activity",
)
GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def number(value, field):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise WorkloadValidationError(f"{field} must be finite and nonnegative")
    return value


def require_basis(value):
    if not isinstance(value, str) or not value.strip():
        raise WorkloadValidationError("Selection and category groups need evidence")


def decode_windows(selection, expected_steps):
    if not isinstance(selection, dict):
        raise WorkloadValidationError("Trace selection must be an object")
    require_basis(selection.get("basis"))
    windows = selection.get("decode_steps")
    if (
        not isinstance(expected_steps, int)
        or isinstance(expected_steps, bool)
        or expected_steps < 1
        or not isinstance(windows, list)
        or len(windows) != expected_steps
    ):
        raise WorkloadValidationError("Decode window count must match measured tokens")
    result = []
    for window in windows:
        if not isinstance(window, dict):
            raise WorkloadValidationError("Decode windows must be objects")
        start = number(window.get("start_us"), "start_us")
        end = number(window.get("end_us"), "end_us")
        if end <= start or (result and result[-1][1] != start):
            raise WorkloadValidationError(
                "Decode windows must be positive, ordered and contiguous"
            )
        result.append((start, end))
    return result


def assignments(selection, event_count):
    groups = selection.get("categories", [])
    if not isinstance(groups, list):
        raise WorkloadValidationError("categories must be a list")
    result, names = {}, set()
    for group in groups:
        if not isinstance(group, dict):
            raise WorkloadValidationError("Category groups must be objects")
        name = group.get("name")
        if name not in CATEGORIES or name in names:
            raise WorkloadValidationError("Unknown or duplicate GPU category")
        names.add(name)
        require_basis(group.get("basis"))
        indices = group.get("event_indices")
        if not isinstance(indices, list) or not indices:
            raise WorkloadValidationError("Category groups need event indices")
        for index in indices:
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < event_count
                or index in result
            ):
                raise WorkloadValidationError(
                    "Invalid or multiply assigned event index"
                )
            result[index] = name
    return result


def gpu_interval(event):
    categories = str(event.get("cat", "")).split(",")
    if not GPU_CATEGORIES.intersection(category.strip() for category in categories):
        return None
    if event.get("ph") != "X":
        raise WorkloadValidationError("GPU activity must use complete-duration events")
    start = number(event.get("ts"), "GPU ts")
    duration = number(event.get("dur"), "GPU dur")
    end = number(start + duration, "GPU end")
    if "pid" not in event:
        raise WorkloadValidationError("GPU activity must identify its device lane")
    return start, end


def selected_activity(events, windows, labels):
    lower, upper = windows[0][0], windows[-1][1]
    changes = defaultdict(Counter)
    for start, end in windows:
        changes[start]
        changes[end]
    matched, devices, names = set(), set(), {}
    kernel_count, activity_count = 0, 0
    for index, event in enumerate(events):
        interval = gpu_interval(event)
        if interval is None:
            continue
        start, end = max(interval[0], lower), min(interval[1], upper)
        if end <= start:
            continue
        matched.add(index)
        devices.add(str(event["pid"]))
        label = labels.get(index, "unclassified_gpu")
        changes[start][label] += 1
        changes[end][label] -= 1
        name = str(event.get("name", "unnamed"))
        key = (label, name)
        if key not in names:
            names[key] = dict(category=label, name=name, events=0, summed_activity_us=0)
        names[key]["events"] += 1
        names[key]["summed_activity_us"] += end - start
        activity_count += 1
        kernel_count += "kernel" in {
            part.strip() for part in str(event.get("cat", "")).split(",")
        }
    if labels.keys() - matched:
        raise WorkloadValidationError(
            "Assigned events must be GPU activities intersecting selected decode"
        )
    if not kernel_count:
        raise WorkloadValidationError("Selected decode contains no GPU kernels")
    if len(devices) != 1:
        raise WorkloadValidationError("Decode attribution requires one GPU device lane")
    return changes, list(names.values()), activity_count, sorted(devices)


def partition(changes, windows):
    totals = [dict.fromkeys(PARTITIONS, 0.0) for window in windows]
    active = Counter()
    boundaries = sorted(changes)
    step_index = 0
    for start, end in zip(boundaries, boundaries[1:]):
        active.update(changes[start])
        while start >= windows[step_index][1]:
            step_index += 1
        labels = {label for label, count in active.items() if count > 0}
        category = (
            "no_gpu_activity"
            if not labels
            else next(iter(labels)) if len(labels) == 1 else "gpu_category_overlap"
        )
        totals[step_index][category] += end - start
    return totals


def summarize_decode(events, selection, expected_steps):
    windows = decode_windows(selection, expected_steps)
    labels = assignments(selection, len(events))
    changes, names, count, devices = selected_activity(events, windows, labels)
    per_step = partition(changes, windows)
    wall = windows[-1][1] - windows[0][0]
    totals = {name: sum(step[name] for step in per_step) for name in PARTITIONS}
    return dict(
        analysis_status="gpu_partition_only",
        selection_basis=selection["basis"],
        expected_decode_steps=expected_steps,
        selected_wall_us=wall,
        mean_step_us=wall / expected_steps,
        gpu_busy_us=wall - totals["no_gpu_activity"],
        selected_gpu_events=count,
        gpu_device_lanes=devices,
        partitions={
            name: dict(
                total_us=total,
                mean_step_us=total / expected_steps,
                fraction=total / wall,
            )
            for name, total in totals.items()
        },
        steps=[
            dict(start_us=start, end_us=end, wall_us=end - start, partitions_us=parts)
            for (start, end), parts in zip(windows, per_step)
        ],
        activity_names=sorted(
            names, key=lambda row: (-row["summed_activity_us"], row["name"])
        ),
        pending_components=[
            "Audit selected boundaries against prefill/decode evidence",
            "Audit semantic assignments against kernel and correlation evidence",
            "CPU scheduling attribution and CPU/GPU overlap",
            "Frontend streaming and cross-process clock alignment",
            "Instrumentation overhead and graph/eager interpretation",
        ],
        note="GPU activity is unioned in trace-clock microseconds; concurrent distinct categories are explicit overlap; no_gpu_activity does not identify its CPU cause; summed activity durations are non-additive diagnostics",
    )
