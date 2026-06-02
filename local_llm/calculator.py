from __future__ import annotations

import ast
import operator
from collections.abc import Callable

from local_llm.tools import ToolInputSchema, ToolRunRequest, ToolRunResult, ToolSpec

MAX_POWER_BASE = 1_000_000
MAX_POWER_EXPONENT = 12


class CalculatorTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calculator",
            description="Evaluate safe arithmetic and numeric expressions.",
            input_schema=ToolInputSchema(
                description="A numeric expression using +, -, *, /, //, %, **",
                examples=("144 / 12", "2 ** 8", "(10 + 5) * 3"),
            ),
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        try:
            result = evaluate_math_expression(request.raw_input)
            output = format_number(result)
            return ToolRunResult(request.tool_name, output, True)
        except ValueError as error:
            return ToolRunResult(request.tool_name, f"Calculator error: {error}", False)


def evaluate_math_expression(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError(
            f"Invalid expression: {expression}. Expected a numeric expression."
        ) from error

    return evaluate_ast_node(tree.body)


def evaluate_ast_node(node: ast.AST) -> float:
    binary_operators: dict[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: divide,
        ast.FloorDiv: floor_divide,
        ast.Mod: operator.mod,
        ast.Pow: safe_power,
    }
    unary_operators: dict[type[ast.unaryop], Callable[[float], float]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    if isinstance(node, ast.Constant):
        return evaluate_constant(node)

    if isinstance(node, ast.BinOp):
        return evaluate_binary_operation(node, binary_operators)

    if isinstance(node, ast.UnaryOp):
        return evaluate_unary_operation(node, unary_operators)

    raise ValueError(
        f"Unsupported expression node: {type(node).__name__}. "
        "Expected a safe numeric expression."
    )


def evaluate_binary_operation(
    node: ast.BinOp,
    operators: dict[type[ast.operator], Callable[[float, float], float]],
) -> float:
    left = evaluate_ast_node(node.left)
    right = evaluate_ast_node(node.right)
    operator_type = type(node.op)
    operation = operators.get(operator_type)

    if operation is None:
        raise ValueError(
            f"Unsupported binary operator: {operator_type.__name__}. "
            "Expected one of +, -, *, /, //, %, **."
        )

    return operation(left, right)


def evaluate_unary_operation(
    node: ast.UnaryOp,
    operators: dict[type[ast.unaryop], Callable[[float], float]],
) -> float:
    value = evaluate_ast_node(node.operand)
    operator_type = type(node.op)
    operation = operators.get(operator_type)

    if operation is None:
        raise ValueError(
            f"Unsupported unary operator: {operator_type.__name__}. Expected + or -."
        )

    return operation(value)


def evaluate_constant(node: ast.Constant) -> float:
    value = node.value

    if isinstance(value, bool):
        raise ValueError(f"Unsupported boolean value: {value}. Expected a number.")

    if isinstance(value, int | float):
        return float(value)

    raise ValueError(f"Unsupported constant value: {value}. Expected a number.")


def safe_power(left: float, right: float) -> float:
    if abs(left) > MAX_POWER_BASE:
        raise ValueError(f"Power base too large: {left}. Expected <= {MAX_POWER_BASE}.")

    if abs(right) > MAX_POWER_EXPONENT:
        raise ValueError(
            f"Power exponent too large: {right}. Expected <= {MAX_POWER_EXPONENT}."
        )

    return left**right


def divide(left: float, right: float) -> float:
    return left / right


def floor_divide(left: float, right: float) -> float:
    return left // right


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))

    return str(value)
