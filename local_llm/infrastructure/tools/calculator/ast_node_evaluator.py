"""Safe AST evaluation: one strategy per node kind, dispatched by node type.

The tree evaluator narrows each node to its concrete kind and delegates to a
small strategy. Recursion-bearing strategies (binary/unary) hold a reference to
the tree evaluator; the constant strategy is a leaf and needs nothing.

Example:
    value = ExpressionTreeEvaluator().evaluate(tree.body)
"""

from __future__ import annotations

import ast
from typing import Protocol

from local_llm.domain.values.number import Number
from local_llm.infrastructure.tools.calculator.operators import (
    binary_operations,
    unary_operations,
)


class NodeEvaluator(Protocol):
    def evaluate(self, node: ast.AST) -> Number: ...


class ConstantEvaluator:
    def evaluate(self, node: ast.Constant) -> Number:
        return number_from_constant(node)


class BinaryOperationEvaluator:
    def __init__(self, tree_evaluator: NodeEvaluator) -> None:
        self._tree_evaluator = tree_evaluator
        self._operations = binary_operations()

    def evaluate(self, node: ast.BinOp) -> Number:
        operation = self._operations.get(type(node.op))

        if operation is None:
            raise ValueError(
                f"Unsupported binary operator: {type(node.op).__name__}. "
                "Expected one of +, -, *, /, //, %, **."
            )

        left = self._tree_evaluator.evaluate(node.left)
        right = self._tree_evaluator.evaluate(node.right)
        return operation(left, right)


class UnaryOperationEvaluator:
    def __init__(self, tree_evaluator: NodeEvaluator) -> None:
        self._tree_evaluator = tree_evaluator
        self._operations = unary_operations()

    def evaluate(self, node: ast.UnaryOp) -> Number:
        operation = self._operations.get(type(node.op))

        if operation is None:
            raise ValueError(
                f"Unsupported unary operator: {type(node.op).__name__}. "
                "Expected + or -."
            )

        return operation(self._tree_evaluator.evaluate(node.operand))


class ExpressionTreeEvaluator:
    def __init__(self) -> None:
        self._constant = ConstantEvaluator()
        self._binary = BinaryOperationEvaluator(self)
        self._unary = UnaryOperationEvaluator(self)

    def evaluate(self, node: ast.AST) -> Number:
        if isinstance(node, ast.Constant):
            return self._constant.evaluate(node)

        if isinstance(node, ast.BinOp):
            return self._binary.evaluate(node)

        if isinstance(node, ast.UnaryOp):
            return self._unary.evaluate(node)

        raise ValueError(
            f"Unsupported expression node: {type(node).__name__}. "
            "Expected a safe numeric expression."
        )


def number_from_constant(node: ast.Constant) -> Number:
    value = node.value

    if isinstance(value, bool):
        raise ValueError(f"Unsupported boolean value: {value}. Expected a number.")

    if isinstance(value, int | float):
        return Number(float(value))

    raise ValueError(f"Unsupported constant value: {value}. Expected a number.")
