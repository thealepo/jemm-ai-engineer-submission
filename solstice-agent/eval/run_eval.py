"""Small, uncontaminated Solstice evaluation. Run with ``python -m eval.run_eval``.

Only authored customer messages enter the agent; scoring data never does.
"""

import json
import os
import re
import time
from collections import Counter
from unittest import mock

from helpdesk import agent as agent_module
from helpdesk.agent import Agent

# Compatibility for an existing audit script; the clean evaluator never calls these.
from .run_legacy_eval import JUDGE_PROMPT, grade  # noqa: F401


CASES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_cases.json")
CHECKS = {"empty", "contains_all", "contains_any", "contains_none", "exact_values"}
STAGES = {
    "execution": "Execution",
    "routing": "Routing",
    "retrieval": "KB final-selection evidence",
    "tool": "Record-tool evidence",
    "answer": "Deterministic answer assertions",
}


def normalize(text):
    text = str(text or "").lower()
    for dash in ("\u2010", "\u2011", "\u2013", "\u2014"):
        text = text.replace(dash, "-")
    return " ".join(text.split())


def check_text(text, expectation):
    """Return failed deterministic checks; an empty list means pass."""
    unknown = set(expectation) - CHECKS
    if unknown:
        raise ValueError("unknown text assertion(s): %s" % ", ".join(sorted(unknown)))
    normalized = normalize(text)
    reasons = []
    if "empty" in expectation and (not normalized) != expectation["empty"]:
        reasons.append("expected %s text" %
                       ("empty" if expectation["empty"] else "non-empty"))
    for phrase in expectation.get("contains_all", []):
        if normalize(phrase) not in normalized:
            reasons.append("missing phrase %r" % phrase)
    alternatives = expectation.get("contains_any", [])
    if alternatives and not any(normalize(phrase) in normalized for phrase in alternatives):
        reasons.append("missing any of %r" % alternatives)
    for phrase in expectation.get("contains_none", []):
        if normalize(phrase) in normalized:
            reasons.append("forbidden phrase present %r" % phrase)
    for value in expectation.get("exact_values", []):
        pattern = r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(normalize(value))
        if not re.search(pattern, normalized):
            reasons.append("missing exact value %r" % value)
    return reasons


def run_turn(agent, message):
    """Call the real agent once while recording its route and tool context."""
    trace = {"message": message, "route": None, "context": None, "answer": None,
             "error": None, "seconds": 0.0}
    real_route, real_tools = agent_module.route, agent_module.TOOLS

    def traced_route(*args, **kwargs):
        trace["route"] = real_route(*args, **kwargs)
        return trace["route"]

    def wrap_tool(tool):
        def traced_tool(*args, **kwargs):
            trace["context"] = tool(*args, **kwargs)
            return trace["context"]
        return traced_tool
    traced_tools = {name: wrap_tool(tool) for name, tool in real_tools.items()}
    started = time.perf_counter()
    try:
        with mock.patch.object(agent_module, "route", traced_route), \
                mock.patch.object(agent_module, "TOOLS", traced_tools):
            trace["answer"] = agent.handle(message)
    except Exception as exc:
        trace["error"] = "%s: %s" % (type(exc).__name__, exc)
    trace["seconds"] = time.perf_counter() - started
    return trace


def run_evaluation(cases, agent_factory=Agent, turn_runner=run_turn, show=False):
    """Run every scenario and return plain counters."""
    for case in cases:
        for turn in case["turns"]:
            if any(key in turn for key in ("reference", "expected", "expected_answer")):
                raise ValueError("case %s contains a prohibited prose reference" % case["id"])
            expected = turn.get("expect", {})
            unknown = set(expected) - {"route", "context", "answer"}
            if unknown:
                raise ValueError("case %s has unknown expectation(s): %s" % (
                    case["id"], ", ".join(sorted(unknown))))
            for field in ("context", "answer"):
                if field in expected:
                    check_text("", expected[field])  # catches misspelled checks before running
    outcomes = []
    stage_scores = {stage: [0, 0] for stage in STAGES}
    scenario_scores = {kind: [0, 0] for kind in ("single_turn", "multi_turn")}
    workload = Counter()
    for case in cases:
        agent = agent_factory()  # fresh conversation per scenario
        failures = []
        for number, turn in enumerate(case["turns"], 1):
            trace = turn_runner(agent, turn["message"])
            expected = turn.get("expect", {})
            checks = [("execution", [trace["error"]] if trace["error"] else [])]
            if not trace["error"]:
                if "route" in expected:
                    errors = [] if trace["route"] == expected["route"] else [
                        "expected %s, got %s" % (expected["route"], trace["route"])]
                    checks.append(("routing", errors))
                if "context" in expected:
                    route = expected.get("route", trace["route"])
                    stage = "retrieval" if route == "search_kb" else "tool"
                    checks.append((stage, check_text(trace["context"], expected["context"])))
                if "answer" in expected:
                    checks.append(("answer", check_text(trace["answer"], expected["answer"])))
            failed = [(stage, errors) for stage, errors in checks if errors]
            if failed:
                failures.append((number, trace, failed))
            workload["seconds"] += trace.get("seconds", 0.0)
            for stage, errors in checks:
                stage_scores[stage][1] += 1
                stage_scores[stage][0] += int(not errors)
            if trace["error"]:
                break

        workload.update(agent.llm.usage())
        passed = not failures
        outcomes.append({"id": case["id"], "kind": case["kind"], "passed": passed})
        scenario_scores[case["kind"]][1] += 1
        scenario_scores[case["kind"]][0] += int(passed)
        if show:
            print("[%s] %s (%s)" % ("PASS" if passed else "FAIL", case["id"], case["kind"]))
            for number, trace, failed in failures:
                print("  turn %d first failure: %s" % (number, failed[0][0]))
                for stage, errors in failed:
                    print("    %s: %s" % (stage, "; ".join(errors)))
                print("    message: %s" % preview(trace["message"]))
                print("    context: %s" % preview(trace["context"]))
                print("    answer:  %s" % preview(trace["answer"]))
    workload["cost_usd"] = round(workload["cost_usd"], 4)
    report = {"outcomes": outcomes, "stages": stage_scores,
              "scenarios": scenario_scores, "workload": dict(workload)}
    if show:
        print_summary(report)
    return report


def preview(text, limit=180):
    value = " ".join(str(text or "<empty>").split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def print_summary(report):
    print("\nStage scores")
    for stage, label in STAGES.items():
        passed, total = report["stages"][stage]
        if total:
            print("  %-32s %2d/%-2d %5.1f%%" %
                  (label, passed, total, 100.0 * passed / total))

    print("\nEnd-to-end scenario scores")
    for kind in ("single_turn", "multi_turn"):
        passed, total = report["scenarios"][kind]
        percentage = 100.0 * passed / total if total else 0.0
        print("  %-32s %2d/%-2d %5.1f%%" % (kind, passed, total, percentage))

    work = report["workload"]
    print("\nWorkload")
    print("  %(seconds).1fs wall time, %(calls)d model attempts, %(prompt_tokens)d "
          "prompt tokens, %(completion_tokens)d completion tokens, $%(cost_usd).4f" % work)
    print("\nMetric note: these are deterministic assertion-coverage scores, "
          "not a general semantic-accuracy score.")


def main():
    with open(CASES_PATH) as f:
        return run_evaluation(json.load(f), show=True)


if __name__ == "__main__":
    main()
