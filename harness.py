"""A tiny, dependency-free example of a measurable AI agent harness.

The agent in this demo is deliberately simple and deterministic. Replace
``ToyMathAgent`` with an LLM call and keep the harness loop unchanged:

    propose -> evaluate -> diagnose -> improve -> stop
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass, replace
from typing import Iterable


@dataclass(frozen=True)
class Task:
    expression: str
    expected: float
    capability: str = "basic"


@dataclass(frozen=True)
class Policy:
    structured_output: bool = False
    operator_precedence: bool = False
    parentheses: bool = False
    unary_minus: bool = False


@dataclass(frozen=True)
class AgentResponse:
    answer: object
    explanation: str


@dataclass(frozen=True)
class Measurement:
    accuracy: float
    contract: float
    score: float
    failures: tuple[str, ...]


@dataclass(frozen=True)
class Iteration:
    number: int
    policy: Policy
    measurement: Measurement


TASKS = (
    Task("2 + 3", 5),
    Task("10 - 4 + 2", 8),
    Task("2 + 3 * 4", 14, "operator_precedence"),
    Task("(2 + 3) * 4", 20, "parentheses"),
    Task("-3 + 8", 5, "unary_minus"),
    Task("18 / 3 * 2", 12),
)


class ToyMathAgent:
    """A stand-in for an LLM whose behavior is controlled by a policy."""

    def answer(self, task: Task, policy: Policy) -> AgentResponse:
        try:
            if policy.operator_precedence:
                value = _safe_ast_evaluate(
                    task.expression,
                    allow_parentheses=policy.parentheses,
                    allow_unary_minus=policy.unary_minus,
                )
            else:
                value = _left_to_right_evaluate(task.expression)
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
            value = float("nan")

        answer: object = value if policy.structured_output else f"The answer is {value}"
        return AgentResponse(answer=answer, explanation="Calculated from the expression.")


class Evaluator:
    """Measures correctness independently from the agent."""

    def evaluate(
        self, tasks: Iterable[Task], responses: Iterable[AgentResponse]
    ) -> Measurement:
        task_list = list(tasks)
        response_list = list(responses)
        correct = 0
        valid_contract = 0
        failures: list[str] = []

        if len(task_list) != len(response_list):
            raise ValueError("Every evaluation task must have exactly one response")

        for task, response in zip(task_list, response_list):
            if isinstance(response.answer, (int, float)):
                valid_contract += 1
                numeric_answer = float(response.answer)
            else:
                try:
                    numeric_answer = float(str(response.answer).replace("The answer is ", ""))
                except ValueError:
                    numeric_answer = float("nan")

            if numeric_answer == task.expected:
                correct += 1
            else:
                failures.append(task.capability)

        total = len(task_list)
        accuracy = correct / total
        contract = valid_contract / total
        # Correctness matters most, but a correct value in an unusable shape is
        # not a fully correct system output.
        score = 0.8 * accuracy + 0.2 * contract
        if contract < 1:
            failures.insert(0, "structured_output")

        return Measurement(
            accuracy=accuracy,
            contract=contract,
            score=score,
            failures=tuple(dict.fromkeys(failures)),
        )


class Harness:
    """Owns the goal, measurement, feedback, and stopping condition."""

    def __init__(self, tasks: Iterable[Task] = TASKS) -> None:
        self.tasks = tuple(tasks)
        self.agent = ToyMathAgent()
        self.evaluator = Evaluator()

    def run(self, max_iterations: int = 10) -> list[Iteration]:
        policy = Policy()
        history: list[Iteration] = []

        for number in range(1, max_iterations + 1):
            responses = [self.agent.answer(task, policy) for task in self.tasks]
            measurement = self.evaluator.evaluate(self.tasks, responses)
            history.append(Iteration(number, policy, measurement))

            if measurement.accuracy == 1.0 and measurement.contract == 1.0:
                return history

            policy = self._improve(policy, measurement.failures)

        raise RuntimeError("The measurable outcome was not reached before the limit.")

    @staticmethod
    def _improve(policy: Policy, failures: tuple[str, ...]) -> Policy:
        """Turn evaluator feedback into one controlled policy change."""
        for capability in failures:
            if not getattr(policy, capability):
                return replace(policy, **{capability: True})
        raise RuntimeError("No actionable feedback remains.")


def _left_to_right_evaluate(expression: str) -> float:
    """Intentionally naive baseline that ignores normal precedence."""
    if "(" in expression or expression.lstrip().startswith("-"):
        raise ValueError("Unsupported syntax")
    tokens = expression.split()
    value = float(tokens[0])
    for index in range(1, len(tokens), 2):
        operator, operand = tokens[index], float(tokens[index + 1])
        value = _apply(operator, value, operand)
    return value


def _safe_ast_evaluate(
    expression: str, *, allow_parentheses: bool, allow_unary_minus: bool
) -> float:
    if not allow_parentheses and ("(" in expression or ")" in expression):
        raise ValueError("Parentheses are not enabled")

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            operators = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
            }
            symbol = operators.get(type(node.op))
            if symbol is None:
                raise ValueError("Operator is not allowed")
            return _apply(symbol, visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if not allow_unary_minus:
                raise ValueError("Unary minus is not enabled")
            return -visit(node.operand)
        raise ValueError("Expression contains unsupported syntax")

    return visit(ast.parse(expression, mode="eval"))


def _apply(operator: str, left: float, right: float) -> float:
    operations = {
        "+": lambda: left + right,
        "-": lambda: left - right,
        "*": lambda: left * right,
        "/": lambda: left / right,
    }
    if operator not in operations:
        raise ValueError(f"Unsupported operator: {operator}")
    return operations[operator]()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    args = parser.parse_args()
    history = Harness().run()

    if args.json:
        print(json.dumps([asdict(item) for item in history], indent=2))
        return

    print("Measurable AI harness\n")
    for item in history:
        m = item.measurement
        enabled = [name for name, value in asdict(item.policy).items() if value]
        print(
            f"Iteration {item.number}: score={m.score:.3f} "
            f"accuracy={m.accuracy:.0%} contract={m.contract:.0%}"
        )
        print(f"  policy: {', '.join(enabled) if enabled else 'baseline'}")
        print(f"  feedback: {', '.join(m.failures) if m.failures else 'goal reached'}")


if __name__ == "__main__":
    main()
