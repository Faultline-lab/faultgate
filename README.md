# faultgate

**A local-first release gate for AI agents.** Hand it the traces your agent recorded, it runs each one through a set of failure checks with a pluggable judge, and exits non-zero if the agent isn't safe to ship. Free, offline after one download, no API key required.

```
$ faultgate check examples/traces.json
trace         search_loop       abstention        wrong_direction
------------------------------------------------------------------
fc2cdab304bd  FAIL 0.76         ok   0.63         FAIL 0.52
6d1f9d67a2d4  ok   0.51         FAIL 0.58         ok   0.63
8bf3879339eb  ok   0.67         ok   0.62         ok   0.76
3a9e507a0f6d  FAIL 0.81         FAIL 0.76         FAIL 0.61
2ab3b6c1e891  FAIL 0.56         ok   0.62         ok   0.73

4/5 traces failed · search_loop=3, abstention=2, wrong_direction=2 · judge laya · 55.5s
$ echo $?
1
```

Real output, unedited, on five real agent runs (`openai/gpt-5.6-luna` on a multi-hop retrieval task, recorded by [FAULTLINE](https://github.com/samirsawarkar/faultline-ai-reliability)). Trace 1 genuinely looped for 12 steps; trace 2 genuinely abstained; trace 3 is a clean 2-step success. Trace 4 is a 9-step success that the judge over-flags — see [How much to trust a verdict](#how-much-to-trust-a-verdict) before you wire this into CI.

## Install

```bash
pip install faultgate
```

The first `check` downloads the default judge, [Laya](https://huggingface.co/convaiinnovations/laya) (~800 MB, Apache-2.0), pinned to the revision we calibrated. After that it runs fully offline on CPU or Apple Silicon. Torch comes with it, so it's a developer-machine install, not a slim CI image — see [What it doesn't do yet](#what-it-doesnt-do-yet).

## Usage

```bash
faultgate check traces.json                          # all checks, Laya judge
faultgate check traces.json --check search_loop,abstention
faultgate check traces.json --json report.json       # full per-trace verdicts
faultgate check traces.json --judge litellm:openai/gpt-5.6-luna   # any API model
faultgate checks                                     # list checks
faultgate judges                                     # list judge specs
```

Exit codes: `0` clean · `1` at least one check fired · `2` bad input.

### Input format

An [OTLP/JSON](https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding) export using the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). One trace is one agent run:

| Span | Attributes read |
|---|---|
| root `invoke_agent` | `gen_ai.agent.name`, `gen_ai.input.messages` (the user prompt), `gen_ai.output.messages` (the final answer) |
| child `execute_tool` | `gen_ai.tool.name`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result` |

Legacy `gen_ai.prompt` / `gen_ai.completion` are accepted as fallback. If your exporter records structure but not content, `faultgate` warns you: the judge is then grading an empty page. [`examples/traces.json`](examples/traces.json) is a complete reference file.

## Checks

| Check | Fires when |
|---|---|
| `search_loop` | The agent issued over-constrained queries that returned nothing and kept reformulating until it ran out of steps. |
| `abstention` | The agent explicitly said it could not find the information instead of answering. |
| `wrong_direction` | On a multi-hop task, the agent walked the dependency chain backwards. |

Each check is one yes/no question. The rule text comes from FAULTLINE project P5, where it was scored against human-labelled failure modes. Two more P5 modes (answer truncation, premature stop) need an expected answer and are not here yet.

## How much to trust a verdict

This is the part every evaluation tool skips. Every judge here was measured against **human labels** on a frozen, content-addressed test split (n = 60 agent runs) in FAULTLINE P5. Cohen's κ, sensitivity (TPR) and specificity (TNR) per check, for the default judge:

| Check | Positives in split | Laya TPR | Laya TNR | Laya κ | GLM-5.3 κ (API judge) |
|---|---|---|---|---|---|
| `search_loop` | 14 | 1.00 | 0.37 | **0.21** | −0.03 |
| `abstention` | 0 | — | 0.93 | 0.00 | 1.00* |
| `wrong_direction` | 0 | — | 0.52 | 0.00 | 1.00* |

\* degenerate: zero positives, the API judge said "no" to everything.

Read it honestly: on the one check with real signal, Laya caught **14 of 14** loops the API judge missed entirely — and flagged 29 innocents with them. It is a **high-recall, low-precision** judge. Use it as a first pass that surfaces traces for a human or a stronger judge, not as the sole vote on a release. The test split has positives on only one check, so the other two rows are a specificity number, not a verdict on the check.

Two more caveats measured on this machine:
- Laya's context is 512 tokens. Long traces are truncated and `faultgate` tells you how many (`3/15` in the demo above). Verdicts on truncated traces saw only the opening steps.
- Laya itself warns at load that this checkpoint ships out-of-range temperatures and clamps them; treat its confidence column as a ranking, not a probability.

Full numbers, confusion matrices and Wilson CIs: [FAULTLINE `projects/p05_judge/DECISIONS.md`](https://github.com/samirsawarkar/faultline-ai-reliability/blob/main/projects/p05_judge/DECISIONS.md).

## Swap the judge

The judge is a four-line contract:

```python
class Judge(Protocol):
    name: str
    def ask(self, state: str, question: str) -> Verdict: ...   # Verdict(detected, confidence, reason)
```

`laya` and `litellm:<model>` ship in the box. When a better local model lands, it's a new adapter, and P5 is the harness that tells you whether it's actually better on your failure modes before you trust it.

## What it doesn't do yet

- **Tolerance bands.** A pass rate of 0.72 might be inside your model's run-to-run noise (FAULTLINE P11 measured 0.60–0.87 on identical inputs). v1 gates on a band measured from repeated runs, not a threshold.
- **Golden answers.** `--golden` unlocks the truncation and premature-stop checks.
- **Policy and injection checks.** FAULTLINE P9/P16 have the taxonomy; they're not wired in.
- **A CI action** with cached weights. Today's install is too heavy for a cold CI runner.
- **Running your agent.** `faultgate` reads traces; it never executes anything.

## Built on FAULTLINE

[FAULTLINE](https://github.com/samirsawarkar/faultline-ai-reliability) is the research workbench behind this tool: 30 days of simulator, 16 measured projects, two papers. `faultgate` is the part you install. Every number in this README links back to the project that produced it.

MIT © 2026 Samir Sawarkar
