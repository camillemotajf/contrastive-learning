"""Backward-compatible wrapper for :mod:`src.data_fetch`.

Prefer ``python -m src.data_fetch --help``. All command-line options are shared.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetch.__main__ import main


if __name__ == "__main__":
    main()
