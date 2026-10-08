import unittest

from eval.run_eval import (
    check_text,
    run_evaluation,
)


class FakeLLM:
    def usage(self):
        return {"calls": 2, "prompt_tokens": 20, "completion_tokens": 3, "cost_usd": 0.01}


class FakeAgent:
    instances = []

    def __init__(self):
        self.messages = []
        self.llm = FakeLLM()
        self.__class__.instances.append(self)


def fake_turn(agent, message):
    agent.messages.append(message)
    return {
        "message": message,
        "route": "search_kb",
        "context": "The Pro plan costs exactly $18 per user per month.",
        "answer": "It costs $18 per user per month.",
        "error": None,
        "seconds": 0.25,
    }


class TextChecksTest(unittest.TestCase):
    def test_required_forbidden_and_exact_values(self):
        expectation = {
            "contains_all": ["per user", "month"],
            "contains_none": ["$9"],
            "exact_values": ["$18", "TKT-1001"],
        }
        reasons = check_text(
            "TKT-1001 costs $18 per user per month, not eighteen dollars.", expectation
        )
        self.assertFalse(reasons)

        reasons = check_text(
            "TKT-10010 costs $180 per user per month and also $9.", expectation
        )
        self.assertTrue(any("TKT-1001" in reason for reason in reasons))
        self.assertTrue(any("$18" in reason for reason in reasons))
        self.assertTrue(any("$9" in reason for reason in reasons))

    def test_polarity_rejects_a_conflicting_claim(self):
        expectation = {
            "contains_all": ["one-way", "read-only"],
            "contains_none": ["two-way sync is available"],
        }
        self.assertFalse(check_text("It is one-way and imports read-only cards.", expectation))
        self.assertTrue(check_text(
            "It is one-way with read-only cards, but two-way sync is available.", expectation
        ))


class HarnessTest(unittest.TestCase):
    def setUp(self):
        FakeAgent.instances = []

    def test_only_bare_messages_reach_agent_and_turns_share_state(self):
        sentinel = "SCORING_ONLY_SENTINEL"
        cases = [
            {
                "id": "conversation",
                "kind": "multi_turn",
                "turns": [
                    {"message": "First customer message."},
                    {
                        "message": "Follow-up message.",
                        "expect": {"answer": {"contains_all": [sentinel]}},
                    },
                ],
            }
        ]

        results = run_evaluation(cases, agent_factory=FakeAgent, turn_runner=fake_turn)

        self.assertEqual(len(FakeAgent.instances), 1)
        self.assertEqual(
            FakeAgent.instances[0].messages,
            ["First customer message.", "Follow-up message."],
        )
        self.assertNotIn(sentinel, " ".join(FakeAgent.instances[0].messages))
        self.assertFalse(results["outcomes"][0]["passed"])

    def test_scenarios_use_fresh_agents_and_scores_stay_separate(self):
        cases = [
            {
                "id": "one",
                "kind": "single_turn",
                "turns": [
                    {
                        "message": "Question one?",
                        "expect": {
                            "route": "search_kb",
                            "context": {"contains_all": ["$18"]},
                            "answer": {"contains_all": ["missing fact"]},
                        },
                    }
                ],
            },
            {
                "id": "two",
                "kind": "single_turn",
                "turns": [{"message": "Question two?"}],
            },
        ]

        results = run_evaluation(cases, agent_factory=FakeAgent, turn_runner=fake_turn)
        self.assertEqual(len(FakeAgent.instances), 2)
        self.assertEqual(results["stages"]["routing"], [1, 1])
        self.assertEqual(results["stages"]["retrieval"], [1, 1])
        self.assertEqual(results["stages"]["answer"], [0, 1])
        self.assertEqual(results["workload"]["seconds"], 0.5)
        self.assertEqual(results["workload"]["calls"], 4)
        self.assertEqual(results["workload"]["prompt_tokens"], 40)
        self.assertEqual(results["workload"]["completion_tokens"], 6)
        self.assertEqual(results["workload"]["cost_usd"], 0.02)

    def test_execution_error_is_diagnosed_and_next_case_runs(self):
        calls = []

        def sometimes_fails(agent, message):
            calls.append(message)
            if message == "Break this turn.":
                return {
                    "message": message,
                    "route": None,
                    "context": None,
                    "answer": None,
                    "error": "RuntimeError: backend failed",
                    "seconds": 0.0,
                }
            return fake_turn(agent, message)

        cases = [
            {
                "id": "broken",
                "kind": "single_turn",
                "turns": [{"message": "Break this turn."}],
            },
            {
                "id": "healthy",
                "kind": "single_turn",
                "turns": [{"message": "Keep evaluating."}],
            },
        ]

        results = run_evaluation(cases, agent_factory=FakeAgent, turn_runner=sometimes_fails)
        self.assertEqual(calls, ["Break this turn.", "Keep evaluating."])
        self.assertEqual(results["stages"]["execution"], [1, 2])
        self.assertFalse(results["outcomes"][0]["passed"])
        self.assertTrue(results["outcomes"][1]["passed"])

    def test_case_schema_rejects_prose_references(self):
        cases = [
            {
                "id": "leaky",
                "kind": "single_turn",
                "turns": [{"message": "Question?", "reference": "Do not send me."}],
            }
        ]
        with self.assertRaises(ValueError):
            run_evaluation(cases, agent_factory=FakeAgent, turn_runner=fake_turn)

    def test_case_schema_rejects_misspelled_assertions(self):
        for expected in (
            {"answer": {"contain_all": ["silently ignored"]}},
            {"answers": {"contains_all": ["silently ignored"]}},
        ):
            cases = [{"id": "typo", "kind": "single_turn",
                      "turns": [{"message": "Question?", "expect": expected}]}]
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                run_evaluation(cases, agent_factory=FakeAgent, turn_runner=fake_turn)


if __name__ == "__main__":
    unittest.main()
