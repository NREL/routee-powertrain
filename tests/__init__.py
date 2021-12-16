# -*- coding: utf-8 -*-

"""Unit test package for routee."""

from pathlib import Path

def test_dir() -> Path:
    """Return the path to the test directory."""
    return Path(__file__).parent