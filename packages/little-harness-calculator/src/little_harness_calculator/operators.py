"""Maps AST operator types to guarded `Number` operations."""

from __future__ import annotations

import ast
from collections.abc import Callable

from little_harness_calculator.number import Number

BinaryOperation = Callable[[Number, Number], Number]
UnaryOperation = Callable[[Number], Number]


def binary_operations() -> dict[type[ast.operator], BinaryOperation]:
    """Return a mapping of AST binary operator types to ``Number`` operations."""
    return {
        ast.Add: Number.add,
        ast.Sub: Number.subtract,
        ast.Mult: Number.multiply,
        ast.Div: Number.divide,
        ast.FloorDiv: Number.floor_divide,
        ast.Mod: Number.modulo,
        ast.Pow: Number.power,
    }


def unary_operations() -> dict[type[ast.unaryop], UnaryOperation]:
    """Return a mapping of AST unary operator types to ``Number`` operations."""
    return {
        ast.UAdd: Number.positive,
        ast.USub: Number.negated,
    }
