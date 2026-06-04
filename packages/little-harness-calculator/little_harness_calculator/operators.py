"""Maps AST operator types to guarded `Number` operations."""

from __future__ import annotations

import ast
from collections.abc import Callable

from little_harness.domain.values.number import Number

BinaryOperation = Callable[[Number, Number], Number]
UnaryOperation = Callable[[Number], Number]


def binary_operations() -> dict[type[ast.operator], BinaryOperation]:
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
    return {
        ast.UAdd: Number.positive,
        ast.USub: Number.negated,
    }
