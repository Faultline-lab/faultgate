# Changelog

## 0.1.0 — unreleased

- `faultgate check <traces.json>`: run built-in checks over an OTLP/JSON trace export; exit 1 if any fires.
- Judges: `laya` (local, pinned to revision `1c5edc17`), `litellm:<model>` (any API model, `[api]` extra).
- Checks: `search_loop`, `abstention`, `wrong_direction` — rule text from FAULTLINE P5.
- `examples/traces.json`: five real agent runs from FAULTLINE P03 (openai/gpt-5.6-luna).
