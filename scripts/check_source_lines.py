"""Check the repository's 500-physical-line limit for source, tests and scripts."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".py", ".sh", ".js", ".ts", ".tsx", ".c", ".cpp", ".h", ".rs"}
MAX_LINES = 500


def main():
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        text=True,
    ).split("\0")
    checked = 0
    violations = []
    for name in sorted(set(files)):
        path = ROOT / name
        if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
            continue
        checked += 1
        lines = len(path.read_text(encoding="utf-8").splitlines())
        if lines > MAX_LINES:
            violations.append(f"{name}: {lines} lines (maximum {MAX_LINES})")
    if violations:
        print("\n".join(violations))
        return 1
    print(f"Checked {checked} source, test and script files: all <= {MAX_LINES} lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
