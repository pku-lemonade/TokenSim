#!/usr/bin/env python3
"""Run TokenSim from a source checkout; installations provide ``tokensim``."""

from TokenSim.cli import cli

if __name__ == "__main__":
    raise SystemExit(cli())
