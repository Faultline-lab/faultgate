import pytest

from faultgate.judge import Judge, LayaJudge, LiteLLMJudge, Verdict, get_judge, parse_json_verdict


def test_fake_satisfies_protocol(fake_judge):
    j: Judge = fake_judge()
    assert isinstance(j.ask("state", "q"), Verdict)


def test_get_judge_specs():
    assert isinstance(get_judge("laya"), LayaJudge)
    j = get_judge("litellm:openai/gpt-5.6-luna")
    assert isinstance(j, LiteLLMJudge) and j.model == "openai/gpt-5.6-luna" and j.name == "litellm:openai/gpt-5.6-luna"
    for bad in ("", "gpt", "litellm:"):
        with pytest.raises(ValueError):
            get_judge(bad)


@pytest.mark.parametrize("text,detected", [
    ('{"detected": true, "confidence": 0.8, "reason": "loop"}', True),
    ('```json\n{"detected": false, "confidence": 0.2, "reason": "fine"}\n```', False),
    ("I think it is a loop", False),
])
def test_parse_json_verdict(text, detected):
    v = parse_json_verdict(text)
    assert v.detected is detected


def test_litellm_judge_uses_completion(monkeypatch):
    import types

    seen = {}

    def completion(**kw):
        seen.update(kw)
        msg = types.SimpleNamespace(content='{"detected": true, "confidence": 0.7, "reason": "r"}')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setitem(__import__("sys").modules, "litellm", types.SimpleNamespace(completion=completion))
    v = LiteLLMJudge("m").ask("STATE", "QUESTION")
    assert v.detected and seen["model"] == "m" and seen["temperature"] == 0.0
    assert "QUESTION" in seen["messages"][0]["content"] and seen["messages"][1]["content"] == "STATE"
