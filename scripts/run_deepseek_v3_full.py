"""Full 61-layer DeepSeek-V3 decode simulation using the operator-table AF runtime."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_deepseek_mla import main

if __name__ == "__main__":
    main(sys.argv[1:], full_default=True)
