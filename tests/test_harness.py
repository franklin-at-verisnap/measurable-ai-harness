import unittest

from harness import Harness, Policy, TASKS, ToyMathAgent


class HarnessTests(unittest.TestCase):
    def test_harness_reaches_the_required_outcome(self) -> None:
        history = Harness().run()

        self.assertEqual(history[-1].measurement.accuracy, 1.0)
        self.assertEqual(history[-1].measurement.contract, 1.0)
        self.assertTrue(all(asdict_value(history[-1].policy)))

    def test_score_never_moves_backwards(self) -> None:
        scores = [item.measurement.score for item in Harness().run()]

        self.assertEqual(scores, sorted(scores))

    def test_baseline_fails_a_known_precedence_case(self) -> None:
        task = next(task for task in TASKS if task.capability == "operator_precedence")
        response = ToyMathAgent().answer(task, Policy(structured_output=True))

        self.assertNotEqual(response.answer, task.expected)


def asdict_value(policy: Policy) -> tuple[bool, ...]:
    return (
        policy.structured_output,
        policy.operator_precedence,
        policy.parentheses,
        policy.unary_minus,
    )


if __name__ == "__main__":
    unittest.main()
