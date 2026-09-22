from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "traces.json"


class FakeJudge:
    """Fires when the question's keyword appears in the state. No model, no network."""

    name = "fake"
    budget = None

    def __init__(self, fire_on: dict[str, str] | None = None):
        self.fire_on = fire_on or {}
        self.calls: list[tuple[str, str]] = []

    def ask(self, state, question):
        from faultgate.judge import Verdict

        self.calls.append((state, question))
        for key, needles in self.fire_on.items():
            needles = [needles] if isinstance(needles, str) else needles
            for needle in needles:
                if key in question and needle in state:
                    return Verdict(True, 0.9, f"fake: saw {needle!r}")
        return Verdict(False, 0.1, "fake: clean")


@pytest.fixture
def fake_judge():
    return FakeJudge
