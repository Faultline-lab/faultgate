# faultgate

**A local-first release gate for AI agents.** Hand it the traces your agent recorded, it checks every run for known failure modes, measures your run-to-run noise, and exits non-zero only when a candidate is a real regression. Free, offline, no model required for the default checks.

```
$ faultgate check examples/traces.json
trace         search_loop       abstention
------------------------------------------------
fc2cdab304bd  FAIL 1.00         ok   1.00
6d1f9d67a2d4  ok   1.00         FAIL 1.00
8bf3879339eb  ok   1.00         ok   1.00
3a9e507a0f6d  ok   1.00         ok   1.00
2ab3b6c1e891  ok   1.00         ok   1.00

2/5 traces failed · search_loop=1, abstention=1 · judge none · 0.0s
$ echo $?
1
```

Real output, unedited, on five real agent runs (`openai/gpt-5.6-luna` on a multi-hop retrieval task, recorded by [FAULTLINE](https://github.com/samirsawarkar/faultline-ai-reliability)). Trace 1 looped for 12 steps and never answered; trace 2 said it couldn't find the answer; 3–5 answered.

## Install

```bash
pip install faultgate
```

The built-in checks are code — nothing to download. Enabling a judge (`--judge laya`) downloads [Laya](https://huggingface.co/convaiinnovations/laya) once (~800 MB, Apache-2.0, pinned revision) and runs offline after that. Read [How much to trust a verdict](#how-much-to-trust-a-verdict) before you enable one.

## Usage

```bash
faultgate check traces.json                          # built-in rule checks
faultgate check traces.json --json report.json       # full per-trace verdicts
faultgate check traces.json --golden golden.json     # expected answers: correctness + truncation
faultgate check traces.json --policy policy.json     # what the agent was allowed to do
faultgate check traces.json --judge laya             # also run judge checks, locally
faultgate check traces.json --judge litellm:openai/gpt-5.6-luna   # any API model
faultgate checks                                     # list checks and their kind
faultgate judges                                     # list judge specs

faultgate baseline run1.json run2.json --out band.json   # measure your noise (>= 2 runs)
faultgate check candidate.json --band band.json          # regression, or just noise?
```

Exit codes: `0` clean · `1` at least one check fired (or, with `--band`, gate FAIL) · `2` bad input.

### Input format

An [OTLP/JSON](https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding) export using the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). One trace is one agent run:

| Span | Attributes read |
|---|---|
| root `invoke_agent` | `gen_ai.agent.name`, `gen_ai.input.messages` (the user prompt), `gen_ai.output.messages` (the final answer), optional `faultgate.termination` (why the run stopped, e.g. `step cap`) |
| child `execute_tool` | `gen_ai.tool.name`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result` |

Legacy `gen_ai.prompt` / `gen_ai.completion` are accepted as fallback. If your exporter records structure but not content, `faultgate` warns you. [`examples/traces.json`](examples/traces.json) is a complete reference file.

## Checks

| Check | Kind | Fires when |
|---|---|---|
| `search_loop` | rule | The run ended with no answer at its step budget and most tool calls returned nothing. |
| `abstention` | rule | The final answer explicitly says the information could not be found. |
| `policy` | rule | A tool call broke the policy: unlisted tool, denied argument pattern, oversized argument, or call budget. |
| `wrong_answer` | golden | Answered, and the expected answer is not in the final answer. |
| `truncation` | golden | The answer has the head of the expected token but not the whole token (`Onyx` for `Onyx-4413`). |
| `wrong_direction` | judge | On a multi-hop task, the agent walked the dependency chain backwards. |

A **rule** is plain code over the trace structure: instant, deterministic, no model. A **golden** check is a rule that also needs the expected answer (`--golden golden.json`, keyed by scenario or by prompt text). A **judge** check is a yes/no question a model answers from the rendered trace; it only runs when you pass `--judge`. All three kinds are calibrated the same way, below.

### Policy: what was the agent allowed to do?

```json
{"allow": ["search", "lookup", "calc"], "max_calls": 10, "max_arg_chars": 200}
```

`faultgate check traces.json --policy policy.json` then flags every step that called a tool outside `allow`, passed an argument matching a denied pattern (path traversal, `file://`, `/etc/passwd`, `~/.ssh` out of the box), exceeded `max_arg_chars`, or pushed the run past `max_calls` — with the step number in the reason. It is FAULTLINE P16's runtime policy applied after the fact: the four structural injection categories from P9 (`tool_exfil_path`, `unlisted_tool`, `query_dump`, `budget_loop`) are each caught by a test in `tests/test_policy.py`. Without a policy file only the denied-pattern rule is active, and it fires on 0 of the 290 real traces shipped here.

Semantic injections — the agent *obeying* text it read in a document — are a judge question, not a rule, and are not in this release.

## Regression or noise? Tolerance bands

Agents are not deterministic. FAULTLINE P11 ran the same model on the same 30 inputs five times and got pass rates from 0.60 to 0.87. A fixed threshold turns that into a coin-flip gate. `faultgate` measures your spread instead:

```
$ faultgate baseline examples/gate/baseline_glm_run1.json examples/gate/baseline_glm_run2.json --out band.json
examples/gate/baseline_glm_run1.json: 30 traces · clean rate 0.93 · 0.0s
examples/gate/baseline_glm_run2.json: 30 traces · clean rate 0.93 · 0.0s
band [0.93, 0.93] over 30 scenarios × 2 runs · 26 clean in every run · written to band.json

$ faultgate check examples/gate/candidate_gpt.json --band band.json
18/30 traces failed · search_loop=18, abstention=0 · judge none · 0.0s
gate: FAIL · clean rate 0.40 vs band [0.93, 0.93] from 2 baseline runs · fail below 0.83
regressions (clean in every baseline, not now): r-0208, r-0214, r-0216, r-0224, r-0226, r-0232, r-0258, ...

$ faultgate check examples/gate/candidate_qwen.json --band band.json
gate: FAIL · clean rate 0.70 vs band [0.93, 0.93] from 2 baseline runs · fail below 0.83

$ faultgate check examples/gate/baseline_glm_run2.json --band band.json
gate: PASS · clean rate 0.93 vs band [0.93, 0.93] from 2 baseline runs · fail below 0.83
```

Real runs: two `z-ai/glm-5.3-flash` baselines and `openai/gpt-5.6-luna` / `qwen/qwen3.7-flash` candidates on the same 30 multi-hop scenarios. The verdicts match FAULTLINE P11's live drill (R4 FAIL / R1 FAIL / R2 PASS), which cost $1.84 and an API judge; this took 0.0 s and no model.

| Verdict | Rule (n = scenarios in the band) | Exit |
|---|---|---|
| **PASS** | clean rate ≥ min observed − 1/n | 0 |
| **WARN** | in between — a human should look | 0 |
| **FAIL** | clean rate < min observed − 3/n | 1 |

It also lists **regressions**: scenarios clean in *every* baseline run and not clean now — the ones to read first. Scenarios are matched across runs by `faultgate.scenario`, else `gen_ai.agent.id`, else a hash of the prompt.

With expected answers the gate grades correctness too, and the numbers land on P11's exactly:

```
$ faultgate baseline examples/gate/baseline_glm_run*.json --golden examples/gate/golden.json --out band.json
examples/gate/baseline_glm_run1.json: 30 traces · clean rate 0.80 · 0.0s      # P11: 24/30 grounded
examples/gate/baseline_glm_run2.json: 30 traces · clean rate 0.87 · 0.0s      # P11: 26/30 grounded
band [0.80, 0.87] over 30 scenarios × 2 runs · 21 clean in every run

$ faultgate check examples/gate/candidate_gpt.json --golden examples/gate/golden.json --band band.json
21/30 traces failed · search_loop=18, abstention=0, policy=0, wrong_answer=3, truncation=3
gate: FAIL · clean rate 0.30 vs band [0.80, 0.87] from 2 baseline runs · fail below 0.70

$ faultgate check examples/gate/candidate_qwen.json --golden examples/gate/golden.json --band band.json
18/30 traces failed · search_loop=9, abstention=0, policy=0, wrong_answer=9, truncation=8
gate: FAIL · clean rate 0.40 vs band [0.80, 0.87] from 2 baseline runs · fail below 0.70
```

(Qwen returns the head of the answer token without its suffix in 8 of 30 runs. Nobody had noticed.)

## How much to trust a verdict

This is the part every evaluation tool skips. Every check and judge here is scored against **human failure-mode labels** on 200 real agent runs (FAULTLINE P4), reported on all 200 and on P5's frozen test split (n = 60). Regenerate with `scripts/calibrate.py`.

| Check | Kind | Positives | TPR | TNR | κ (n=200) | κ (P5 test, n=60) |
|---|---|---|---|---|---|---|
| `search_loop` | rule | 36 | 1.00 | 0.99 | **0.98** | **1.00** |
| `abstention` | rule | 1 | 1.00 | 1.00 | 1.00 | — (0 positives) |
| `policy` | rule | synthetic | 4/4 P9 categories | 0 FP on 290 real traces | — | — |
| `wrong_answer` | golden | 4 | 1.00 | 1.00 | 1.00 | 1.00 |
| `truncation` | golden | 1 | 1.00 | 1.00 | 1.00 | — (0 positives) |
| `wrong_direction` | Laya judge | 1 | 1.00 | 0.61 | 0.02 | 0.00 |

For comparison, the same `search_loop` question put to judges instead of a rule: Laya κ 0.09, GLM-5.3 (API) κ −0.03. **Where a failure is structural, code beats every model we measured**, which is why the default checks are rules.

Read the table honestly:
- `search_loop` is solid on this task family. The human coder saw the same structural signals, so κ measures agreement with that coding, not ground truth from nowhere.
- `abstention` and `truncation` have one positive each to learn from. `abstention` is a phrase list; a paraphrase it doesn't know is a miss.
- `wrong_answer` agrees with the human verdict on all 21 answered runs. It is substring containment, case-insensitive: right for short factual answers, wrong for free-text ones — say what "expected" means for your task before you trust it.
- `policy` has no human-labelled positives: FAULTLINE's agent rejects hostile calls before they become steps, so its traces never contain an executed attack. Recall is by construction on the P9 attack shapes; the number that matters is zero false positives on real runs.
- `wrong_direction` is the only check that needs reading, and the only local judge we have is noise on it (fires on 39% of clean runs). It ships because the contract is the product: the day a local model scores κ ≥ 0.7 on it, that's a one-line adapter, and this harness is what proves it.
- Laya's context is 512 tokens. `faultgate` compacts the trace to fit (answer and termination first, middle steps elided) and counts every call that still overflows; the count is printed.

Full confusion matrices and Wilson CIs for the original P5 judges: [FAULTLINE `projects/p05_judge/DECISIONS.md`](https://github.com/samirsawarkar/faultline-ai-reliability/blob/main/projects/p05_judge/DECISIONS.md).

## Swap the judge

```python
class Judge(Protocol):
    name: str
    budget: int | None          # max characters of trace it can read
    def ask(self, state: str, question: str) -> Verdict: ...   # Verdict(detected, confidence, reason)
```

`laya` and `litellm:<model>` ship in the box. Add a judge, run `scripts/calibrate.py --judge yours`, and publish the row.

## What it doesn't do yet

- **Premature stop** (answered with an intermediate entity of the chain). Needs the traversal chain, not just the final answer.
- **Semantic injection checks** (the agent obeyed text it read). P9 has the taxonomy; needs a judge that scores.
- **A GitHub Action.** Rules-only runs need nothing but Python, so this is close.
- **Running your agent.** `faultgate` reads traces; it never executes anything.

## Built on FAULTLINE

[FAULTLINE](https://github.com/samirsawarkar/faultline-ai-reliability) is the research workbench behind this tool: 30 days of simulator, 16 measured projects, two papers. `faultgate` is the part you install. Every number in this README links back to the run that produced it.

MIT © 2026 Samir Sawarkar
