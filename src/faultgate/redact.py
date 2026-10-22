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

import hashlib
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


def _card_spans(m: re.Match[str]) -> list[tuple[int, int]]:
    """Return every Luhn-valid 13-19 digit window of whole digit groups in m as (start, end)."""
    text = m.group(0)
    base = m.start()
    groups = [(g.start(), g.end()) for g in re.finditer(r"\d+", text)]
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(groups):
        for j in range(len(groups), i, -1):
            digits = "".join(text[a:b] for a, b in groups[i:j])
            if 13 <= len(digits) <= 19 and _luhn(digits):
                spans.append((base + groups[i][0], base + groups[j - 1][1]))
                i = j
                break
        else:
            i += 1
    return spans


def _card(m: re.Match[str], tag: str) -> str:
    """Mask every Luhn-valid 13-19 digit window of whole digit groups in a run of groups.

    The run can carry more than the card: an SSN or an order number written next to it
    ("4111 1111 1111 1111 123-45-6789"). Testing only the whole run let the card through.
    """
    text = m.group(0)
    base = m.start()
    out, done = [], 0
    for start, end in _card_spans(m):
        out += [text[done:start - base], tag]
        done = end - base
    return "".join(out) + text[done:]


def _phone_spans(m: re.Match[str]) -> list[tuple[int, int]]:
    return [(m.start(), m.end())] if 9 <= len(re.sub(r"\D", "", m.group(0))) <= 15 else []


def _phone(m: re.Match[str], tag: str) -> str:
    return tag if _phone_spans(m) else m.group(0)


_SPANS_FN: dict[Callable[[re.Match[str], str], str], Callable[[re.Match[str]], list[tuple[int, int]]]] = {
    _card: _card_spans,
    _phone: _phone_spans,
}


# Order matters: secrets, SSNs and cards before the looser phone / IP patterns.
# Every pattern is anchored by a lookbehind (so a match can only start at a token boundary) and every
# repetition is bounded: worst case is linear in the input, whatever an agent wrote into its trace.
TOKEN = r"[A-Za-z0-9_-]"
PATTERNS: list[tuple[str, re.Pattern[str], Callable[[re.Match[str], str], str] | None]] = [
    ("SECRET", re.compile(r"(?<![\w.-])eyJ[A-Za-z0-9_-]{8,4096}\.[A-Za-z0-9_-]{8,4096}\.[A-Za-z0-9_-]{8,4096}"), None),
    ("SECRET", re.compile(r"(?i)(?<=bearer )[A-Za-z0-9._~+/-]{16,4096}=*"), None),
    ("SECRET", re.compile(
        r"(?<![\w-])(?:"
        r"(?:sk|pk|rk)-(?:[a-z]{1,12}-){0,2}" + TOKEN + r"{16,256}"          # OpenAI, Anthropic, generic
        r"|[sr]k_(?:live|test)_[0-9A-Za-z]{16,256}"                          # Stripe
        r"|AKIA[0-9A-Z]{16}"                                                 # AWS access key id
        r"|gh[pousr]_[A-Za-z0-9]{30,255}"                                    # GitHub
        r"|xox[abposre]-[A-Za-z0-9-]{10,255}|xapp-[A-Za-z0-9-]{10,255}"     # Slack bot / user / app
        r"|AIza[0-9A-Za-z_-]{35}|ya29\.[0-9A-Za-z_-]{20,2048}"               # Google API key / OAuth token
        r"|SG\.[\w-]{16,64}\.[\w-]{16,128}"                                  # SendGrid
        r"|hf_[A-Za-z0-9]{30,64}"                                            # Hugging Face
        r")(?![\w-])"), None),
    ("EMAIL", re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){0,8}"
                         r"\.[A-Za-z]{2,24}(?![A-Za-z0-9-])"), None),
    # SSNs before cards: an SSN written beside a card must not join its digit run
    ("US_SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), None),
    # a run of digit groups (cards are written 4-4-4-4, 4-6-5 or unbroken); a sentence's full stop may follow, a decimal may not
    ("CREDIT_CARD", re.compile(r"(?<![\w.])\d{3,19}(?:[ -]\d{3,19}){0,15}+(?!\w)(?!\.\d)"), _card),
    ("IP_ADDRESS", re.compile(r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\w.])"), None),
    ("IP_ADDRESS", re.compile(r"(?<![\w:])(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}(?![\w:])"), None),
    # a phone is a whole number group: not part of a longer run of digit blocks (a card that failed Luhn, an id)
    ("PHONE", re.compile(r"(?<![\w.+-])(?<!\d[ .-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)|\d{2,4})[ .-]\d{3,4}[ .-]?\d{3,4}"
                         r"(?![\w.-])(?![ .-]?\d)"), _phone),
]


def find(text: str) -> list[tuple[int, int, str]]:
    """Return every private value in text as (start, end, kind), in PATTERNS order with overlaps removed."""
    if not text:
        return []
    taken: list[tuple[int, int, str]] = []
    work = text
    for kind, rx, sub in PATTERNS:
        found: list[tuple[int, int]] = []
        for m in rx.finditer(work):
            if sub is None:
                found.append((m.start(), m.end()))
            elif sub in _SPANS_FN:
                found.extend(_SPANS_FN[sub](m))
            else:
                found.append((m.start(), m.end()))
        found = [(s, e) for s, e in found if e > s]
        if not found:
            continue
        taken.extend((s, e, kind) for s, e in found)
        # Later patterns see earlier matches as masked (same length, so offsets stay valid): an SSN written
        # beside a card must not join the card's digit run, exactly as sequential replacement behaved.
        chars = list(work)
        for s, e in found:
            chars[s:e] = "\x00" * (e - s)
        work = "".join(chars)
    taken.sort(key=lambda s: s[0])
    return taken


@dataclass(frozen=True)
class Redactor:
    name: str
    fn: Callable[[str], str]

    def __call__(self, text: str) -> str:
        return self.fn(text) if text else text


def _builtin(text: str) -> str:
    for start, end, kind in reversed(find(text)):
        text = text[:start] + f"<{kind}>" + text[end:]
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


def safe_key(key: str, redact: Callable[[str], str] | None) -> str:
    """A scenario key fit to write out: unchanged when there is nothing to mask, else masked plus a short hash
    of the raw key, so two masked keys stay distinct and the same raw key maps to the same output every run."""
    if redact is None:
        return key
    masked = redact(key)
    return key if masked == key else f"{masked}#{hashlib.sha256(key.encode('utf-8')).hexdigest()[:8]}"


def get_redactor(spec: str) -> Redactor | None:
    if spec == "off":
        return None
    if spec == "builtin":
        return builtin()
    if spec == "presidio":
        return presidio()
    raise ValueError(f"unknown redaction {spec!r}; use {', '.join(REDACTIONS)}")
