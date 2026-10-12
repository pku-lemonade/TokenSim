"""Locate bundled runtime data in a checkout or an installed wheel."""

from pathlib import Path
import os
import sys


def default_data_root():
    override = os.environ.get("TOKENSIM_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2] / "data"
    if (checkout / "devices").is_dir():
        return checkout
    return Path(sys.prefix) / "share" / "tokensim" / "data"
