# Changelog

## 0.1.0 — unreleased

- `faultgate check <traces.json>`: run checks over an OTLP/JSON GenAI-semconv trace export; exit 1 if any fires.
- Rule checks (no model): `search_loop` (κ 0.98 on 200 human-labelled runs, 1.00 on the P5 split), `abstention` (1/1, 0 FP).
- Judge checks (opt-in `--judge`): `wrong_direction`. Judges: `laya` (local, pinned `1c5edc17`, budgeted render), `litellm:<model>` (`[api]` extra).
- `faultgate baseline` + `check --band`: tolerance band from ≥ 2 baseline runs (FAULTLINE P11 rule), PASS/WARN/FAIL, named regressions.
- `scripts/calibrate.py`: score any check/judge against FAULTLINE P4 human labels.
- Examples: five real runs (`examples/traces.json`); two `glm-5.3-flash` baselines and `gpt-5.6-luna` / `qwen3.7-flash` candidates on P11's 30 golden scenarios (`examples/gate/`).
