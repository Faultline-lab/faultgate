"""Judges: anything that answers a yes/no question about a trace.

The contract is the product. A judge is any object with a ``name`` and
``ask(state, question) -> Verdict``. Swap the model, keep the gate.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Protocol

LAYA_CHECKPOINT = "convaiinnovations/laya"
LAYA_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"  # pinned: the revision FAULTLINE P5 calibrated
LAYA_CONTEXT = 512


@dataclass(frozen=True)
class Verdict:
    detected: bool
    confidence: float
    reason: str


class Judge(Protocol):
    name: str

    def ask(self, state: str, question: str) -> Verdict: ...


class LayaJudge:
    """Local, free, ~35 ms per question. Ported from FAULTLINE faultline_p2/judge/laya_judge.py."""

    name = "laya"

    def __init__(self) -> None:
        self._agent = None
        self._lock = threading.Lock()
        self.calls = 0
        self.truncated = 0

    def _load(self):
        if self._agent is None:
            with self._lock:
                if self._agent is None:
                    import laya  # heavy import, deferred
                    from huggingface_hub import snapshot_download

                    # laya.load has no revision arg; resolve the pinned snapshot ourselves
                    local = snapshot_download(LAYA_CHECKPOINT, revision=LAYA_REVISION, allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
                    self._agent = laya.load(local)
        return self._agent

    def ask(self, state: str, question: str) -> Verdict:
        agent = self._load()
        n_tokens = len(agent.tok(state, add_special_tokens=False)["input_ids"])
        self.calls += 1
        if n_tokens > LAYA_CONTEXT:
            self.truncated += 1
        with self._lock:
            res = agent.predict(state, {"q": {"type": "noul", "instructions": question}})
        ans = res["answers"]["q"]
        p = float(ans["noul"])
        return Verdict(detected=p >= 0.5, confidence=float(ans.get("confidence", max(p, 1 - p))), reason=f"laya p={p:.2f}")


class LiteLLMJudge:
    """Any API model via litellm (``pip install faultgate[api]``). Temperature 0, JSON answer."""

    SYSTEM = (
        "You are an expert AI reliability judge evaluating an agent trace.\n{question}\n"
        'Respond ONLY with valid JSON: {{"detected": bool, "confidence": float, "reason": str}}'
    )

    def __init__(self, model: str) -> None:
        self.model = model
        self.name = f"litellm:{model}"

    def ask(self, state: str, question: str) -> Verdict:
        import litellm  # optional extra

        resp = litellm.completion(
            model=self.model,
            messages=[{"role": "system", "content": self.SYSTEM.format(question=question)}, {"role": "user", "content": state}],
            temperature=0.0,
            max_tokens=512,
        )
        return parse_json_verdict(resp.choices[0].message.content or "")


def parse_json_verdict(text: str) -> Verdict:
    """Tolerant parse: strips code fences; on failure, detected=False with the raw text as reason."""
    t = text.strip()
    if "```" in t:
        t = t.split("```", 2)[1]
        t = t[4:] if t.startswith("json") else t
    try:
        d = json.loads(t.strip())
        return Verdict(bool(d.get("detected", False)), float(d.get("confidence", 0.0)), str(d.get("reason", ""))[:200])
    except (ValueError, AttributeError):
        return Verdict(False, 0.0, f"unparseable judge output: {text[:100]}")


def get_judge(spec: str) -> Judge:
    """``laya`` or ``litellm:<model>``."""
    if spec == "laya":
        return LayaJudge()
    if spec.startswith("litellm:") and len(spec) > 8:
        return LiteLLMJudge(spec[8:])
    raise ValueError(f"unknown judge {spec!r}; use 'laya' or 'litellm:<model>'")


JUDGE_SPECS = {
    "laya": "local decision model, free, offline after the first ~800 MB download (default)",
    "litellm:<model>": "any API model via litellm, e.g. litellm:openai/gpt-5.6-luna (needs faultgate[api] and the provider's API key)",
}
