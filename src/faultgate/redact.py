"""Mask personal data and secrets in anything that can leave the machine.

Rules run on the raw trace in memory (a policy has to compare the real customer
id); what gets masked is what could travel: the rendered trace sent to an API
judge, and every verdict reason that lands in the terminal, the JSON report, the
Markdown PR comment or the audit files. ``faultgate`` writes no trace payloads to
disk itself.

``builtin`` is stdlib regex, no download: emails, credit cards (Luhn-checked),
US SSNs, phone numbers, IPv4/IPv6 addresses, and API keys / bearer tokens / JWTs.
``presidio`` (``pip install 'faultgate[pii]'``) adds Microsoft Presidio's
recognisers, including person names and locations, on top of ``builtin``.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

REDACTIONS = ("builtin", "presidio", "off")


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _card(m: re.Match[str]) -> bool:
    digits = re.sub(r"\D", "", m.group(0))
    return 13 <= len(digits) <= 19 and _luhn(digits)


def _phone(m: re.Match[str]) -> bool:
    return 9 <= len(re.sub(r"\D", "", m.group(0))) <= 15


# order matters: secrets and cards before the looser phone / IP patterns
PATTERNS: list[tuple[str, re.Pattern[str], Callable[[re.Match[str]], bool] | None]] = [
    ("SECRET", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), None),
    ("SECRET", re.compile(r"(?i)(?<=bearer )[A-Za-z0-9._~+/-]{16,}=*"), None),
    ("SECRET", re.compile(r"\b(?:sk|pk|rk)-(?:[a-z]+-)?[A-Za-z0-9_-]{16,}|\bAKIA[0-9A-Z]{16}\b|\bgh[pousr]_[A-Za-z0-9]{30,}"
                          r"|\bxox[abpr]-[A-Za-z0-9-]{10,}|\bAIza[0-9A-Za-z_-]{35}"), None),
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"), None),
    ("CREDIT_CARD", re.compile(r"(?<![\w.])\d(?:[ -]?\d){12,18}(?![\w.])"), _card),
    ("US_SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), None),
    ("IP_ADDRESS", re.compile(r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\w.])"), None),
    ("IP_ADDRESS", re.compile(r"(?<![\w:])(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}(?![\w:])"), None),
    # a phone is a whole number group: not part of a longer run of digit blocks (a card that failed Luhn, an id)
    ("PHONE", re.compile(r"(?<![\w.+-])(?<!\d[ .-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)|\d{2,4})[ .-]\d{3,4}[ .-]?\d{3,4}"
                         r"(?![\w.-])(?![ .-]?\d)"), _phone),
]


@dataclass(frozen=True)
class Redactor:
    name: str
    fn: Callable[[str], str]

    def __call__(self, text: str) -> str:
        return self.fn(text) if text else text


def _builtin(text: str) -> str:
    for kind, rx, ok in PATTERNS:
        text = rx.sub(lambda m: f"<{kind}>" if ok is None or ok(m) else m.group(0), text)
    return text


def builtin() -> Redactor:
    return Redactor("builtin", _builtin)


def presidio() -> Redactor:
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
    except ImportError as e:
        raise ValueError("--redact presidio needs the extra: pip install 'faultgate[pii]' (and a spaCy model)") from e
    analyzer, anonymizer = AnalyzerEngine(), AnonymizerEngine()

    def fn(text: str) -> str:
        found = analyzer.analyze(text=text, language="en")
        masked = anonymizer.anonymize(text=text, analyzer_results=found).text if found else text
        return _builtin(masked)

    return Redactor("presidio", fn)


def get_redactor(spec: str) -> Redactor | None:
    if spec == "off":
        return None
    if spec == "builtin":
        return builtin()
    if spec == "presidio":
        return presidio()
    raise ValueError(f"unknown redaction {spec!r}; use {', '.join(REDACTIONS)}")
