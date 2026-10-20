"""Select nondominated native serving points in deterministic order."""


def pareto(rows):
    points = sorted(
        (row for row in rows if row["status"] == "ok"),
        key=lambda row: (-row["tokens_per_s_per_user"], -row["tokens_per_s_per_mw"]),
    )
    frontier, best = [], -1.0
    for row in points:
        if row["tokens_per_s_per_mw"] > best:
            frontier.append(row)
            best = row["tokens_per_s_per_mw"]
    return list(reversed(frontier))
