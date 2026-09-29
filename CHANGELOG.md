# Changelog

## 0.2.0 — 2026-09-29

- Chat-only trace support: tool calls embedded inside message parts (paired with tool responses by ID), in addition to `execute_tool` spans.
- New rule checks: `repeated_call` (same tool + normalized arguments issued >= 3 times with identical result, κ 0.95 vs a blind second rater (Claude Opus 5.5) on 97 real runs) and `unrecovered_tool_error` (structural tool error with no later success, or a span error (error.type or status code 2) with no later successful tool call; recall 0.57, κ 0.54), calibrated against FAULTLINE P18 on Exgentic real benchmark traces.
- `scripts/fetch_real_traces.py` and `make demo-real`: sample real traces across 6 benchmarks from `Exgentic/agent-llm-traces-v2` and evaluate with default rule checks.
- `demo-real` works from an empty cache via `snapshot_download`, carries span status codes into OTLP, and adds `demo` optional dependency group.
- Hybrid traces preserve message-embedded tool calls deduplicated against `execute_tool` span IDs.
- Tool and span error signals preserved (Span `status_code` parsed from OTLP status, `is_error`/error status flagged on message tool call responses).
- Hybrid trace steps ordered chronologically by span start time, preserving originating `span_id`.
- Span error recovery ported from FAULTLINE P18: middle span errors unrecovered unless a later span runs a non-error tool call.
- Pinned Exgentic dataset revision in `scripts/fetch_real_traces.py` and handled expected trace-failure exit code in `make demo-real`.

## 0.1.0 — 2026-09-22

- `faultgate check <traces.json>`: run checks over an OTLP/JSON GenAI-semconv trace export; exit 1 if any fires.
- Rule checks (no model): `search_loop` (κ 0.98 on 200 human-labelled runs, 1.00 on the P5 split), `abstention` (1/1, 0 FP).
- `policy` rule + `--policy policy.json` (allowlist, denied argument patterns, max argument size, call budget) — FAULTLINE P16 after the fact; P9's four structural attack shapes covered by tests; 0 FP on 290 real traces.
- Golden checks + `--golden golden.json` (keyed by scenario or prompt): `wrong_answer` (4/4 vs human verdict, 0 FP), `truncation` (1/1, 0 FP).
- Composite GitHub Action (`uses: samirsawarkar/faultgate@<ref>`), self-tested in CI against the shipped examples.
- `laya` is an extra (`faultgate[laya]`); the base install is pydantic only.
- Judge checks (opt-in `--judge`): `wrong_direction`. Judges: `laya` (local, pinned `1c5edc17`, budgeted render), `litellm:<model>` (`[api]` extra).
- `faultgate baseline` + `check --band`: tolerance band from ≥ 2 baseline runs (FAULTLINE P11 rule), PASS/WARN/FAIL, named regressions.
- `scripts/calibrate.py`: score any check/judge against FAULTLINE P4 human labels.
- Examples: five real runs (`examples/traces.json`); two `glm-5.3-flash` baselines and `gpt-5.6-luna` / `qwen3.7-flash` candidates on P11's 30 golden scenarios (`examples/gate/`).
