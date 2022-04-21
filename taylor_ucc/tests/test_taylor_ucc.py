"""
Unit and regression test for the taylor_ucc package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import taylor_ucc


def test_taylor_ucc_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "taylor_ucc" in sys.modules
