"""Security and correctness tests for the default calculator tool."""

import pytest

from src.tools.registry import compute_math


def test_calculator_evaluates_basic_arithmetic():
    assert compute_math("128 * 4 + 50") == 562.0
    assert compute_math("-(8 + 2) / 5") == -2.0


@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo unsafe')",
    "2 ** 101",
    "1 / 0",
    "9" * 257,
    "1 + " * 40 + "1",
])
def test_calculator_rejects_unsafe_or_unbounded_expressions(expression):
    with pytest.raises((ValueError, ZeroDivisionError)):
        compute_math(expression)
