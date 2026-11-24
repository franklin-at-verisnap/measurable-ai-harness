# Measurable AI Harness

A tiny, dependency-free Python example showing why an AI agent needs measurable outcomes—not just instructions—to move toward a correct solution.

The harness owns a closed feedback loop:

```text
propose -> evaluate -> diagnose -> improve -> stop
```

The demo agent begins with an intentionally weak policy. It answers a small suite of arithmetic tasks, while an independent evaluator measures:

- **Accuracy:** Is the numerical answer correct?
- **Output contract:** Is the result returned in a machine-usable format?
- **Composite score:** `80% accuracy + 20% contract compliance`

After each iteration, the harness converts concrete failures into one controlled policy improvement. It stops only when both accuracy and contract compliance reach 100%.

## Run it

Requires Python 3.9 or newer. There are no third-party dependencies.

```bash
python3 harness.py
```

Example output:

```text
Measurable AI harness

Iteration 1: score=0.400 accuracy=50% contract=0%
  policy: baseline
  feedback: structured_output, operator_precedence, parentheses, unary_minus
Iteration 2: score=0.600 accuracy=50% contract=100%
  policy: structured_output
  feedback: operator_precedence, parentheses, unary_minus
Iteration 3: score=0.733 accuracy=67% contract=100%
  policy: structured_output, operator_precedence
  feedback: parentheses, unary_minus
Iteration 4: score=0.867 accuracy=83% contract=100%
  policy: structured_output, operator_precedence, parentheses
  feedback: unary_minus
Iteration 5: score=1.000 accuracy=100% contract=100%
  policy: structured_output, operator_precedence, parentheses, unary_minus
  feedback: goal reached
```

Machine-readable output is also available:

```bash
python3 harness.py --json
```

## Run the tests

```bash
python3 -m unittest discover -s tests -v
```

## Why this illustrates an AI harness

The agent proposes an answer, but it does not decide whether the answer is good enough. The harness defines correctness, evaluates the output, supplies actionable feedback, controls the next change, and enforces the stopping condition.

The `ToyMathAgent` is deterministic so the example is free, safe, and easy to reproduce. In a real system, it can be replaced with an LLM call while keeping the same control loop. The important engineering idea is unchanged: without an external measure, an agent can generate more output, but it cannot reliably know whether it is getting closer to the required result.

## Repository structure

```text
.
├── harness.py              # agent, evaluator, feedback loop, and CLI
└── tests/
    └── test_harness.py     # verifies convergence and monotonic scoring
```

## License

MIT
