# Changelog

## 0.1.0 — unreleased

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
