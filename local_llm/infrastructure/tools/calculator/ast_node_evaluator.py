"""Polymorphic AST evaluation: one evaluator per node kind, no isinstance chain.

The tree evaluator dispatches by node type to a strategy. Recursion-bearing
strategies (binary/unary) hold a reference to the tree evaluator; the constant
strategy is a leaf and needs nothing.

Example:
    value = ExpressionTreeEvaluator().evaluate(tree.body)
"""

from __future__ import annotations

import ast
from typing import Protocol, cast

from local_llm.domain.values.number import Number
from local_llm.infrastructure.tools.calculator.operators import (
    binary_operations,
    unary_operations,
)


class NodeEvaluator(Protocol):
    def evaluate(self, node: ast.AST) -> Number: ...


class ConstantEvaluator:
    def evaluate(self, node: ast.AST) -> Number:
        return number_from_constant(cast("ast.Constant", node))


class BinaryOperationEvaluator:
    def __init__(self, tree_evaluator: NodeEvaluator) -> None:
        self._tree_evaluator = tree_evaluator
        self._operations = binary_operations()

    def evaluate(self, node: ast.AST) -> Number:
        binary = cast("ast.BinOp", node)
        operation = self._operations.get(type(binary.op))

        if operation is None:
            raise ValueError(
                f"Unsupported binary operator: {type(binary.op).__name__}. "
                "Expected one of +, -, *, /, //, %, **."
            )

        left = self._tree_evaluator.evaluate(binary.left)
        right = self._tree_evaluator.evaluate(binary.right)
        return operation(left, right)


class UnaryOperationEvaluator:
    def __init__(self, tree_evaluator: NodeEvaluator) -> None:
        self._tree_evaluator = tree_evaluator
        self._operations = unary_operations()

    def evaluate(self, node: ast.AST) -> Number:
        unary = cast("ast.UnaryOp", node)
        operation = self._operations.get(type(unary.op))

        if operation is None:
            raise ValueError(
                f"Unsupported unary operator: {type(unary.op).__name__}. "
                "Expected + or -."
            )

        return operation(self._tree_evaluator.evaluate(unary.operand))


class ExpressionTreeEvaluator:
    def __init__(self) -> None:
        self._evaluators: dict[type[ast.AST], NodeEvaluator] = {
            ast.Constant: ConstantEvaluator(),
            ast.BinOp: BinaryOperationEvaluator(self),
            ast.UnaryOp: UnaryOperationEvaluator(self),
        }

    def evaluate(self, node: ast.AST) -> Number:
        evaluator = self._evaluators.get(type(node))

        if evaluator is None:
            raise ValueError(
                f"Unsupported expression node: {type(node).__name__}. "
                "Expected a safe numeric expression."
            )

        return evaluator.evaluate(node)


def number_from_constant(node: ast.Constant) -> Number:
    value = node.value

    if isinstance(value, bool):
        raise ValueError(f"Unsupported boolean value: {value}. Expected a number.")

    if isinstance(value, int | float):
        return Number(float(value))

    raise ValueError(f"Unsupported constant value: {value}. Expected a number.")
