"""Load the repository's optional example package only in example tests."""

from contextlib import contextmanager
from pathlib import Path
import sys


@contextmanager
def example_storage_package():
    path = str(Path(__file__).resolve().parents[2] / "examples/storage/hbf")
    previous = list(sys.path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = previous
