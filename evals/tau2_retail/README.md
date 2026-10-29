# tau2-bench retail smoke test

Run a small slice of the public [tau2-bench](https://github.com/sierra-research/tau2-bench)
retail benchmark, export the actual agent/tool spans as OTLP/JSON, and gate them
with faultgate's retail policy. This is an integration smoke test, not a
benchmark score.

## Pinned versions

| Component | Version |
|---|---|
| tau2-bench | [`7aad81c37a3a78b4ed3e4866769f851400c64415`](https://github.com/sierra-research/tau2-bench/commit/7aad81c37a3a78b4ed3e4866769f851400c64415) (2026-10-28) |
| faultgate | 0.4.1 (this checkout) |
| Python | 3.12.15 |
| uv | 0.12.24 |
| opentelemetry-sdk | 1.45.2 |

The runner records `tau2_commit` in `results.json` and refuses to run against a
different checkout unless `--allow-other-commit` is passed; a run against a
different commit is not comparable.

## Setup

```bash
git clone https://github.com/sierra-research/tau2-bench.git
cd tau2-bench
git checkout 7aad81c37a3a78b4ed3e4866769f851400c64415
uv sync
# The runner instruments tau2 directly; tau2 does not depend on the SDK.
uv pip install --python .venv/bin/python "opentelemetry-sdk>=1.27"
```

On Windows the interpreter is `.venv\Scripts\python.exe`.

Install faultgate in its own environment from the faultgate checkout:

```bash
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

### Credentials

The runner reads the standard environment variables, then falls back to the
tau2 checkout's `.env` for any missing `LLM_*` / `OPENAI_*` value:

```bash
export LLM_API_BASE=https://your-openai-compatible-endpoint/v1
export LLM_API_KEY=...
export LLM_MODEL=openai/qwen3.8-flash
# Only for gateways that route by session header:
export LLM_EXTRA_HEADERS='{"x-opencode-session":"..."}'
```

The run in `artifacts/` used an OpenAI-compatible OpenCode inference gateway
and `qwen3.8-flash` for the agent, user simulator, and NL-assertion judge. The
gateway needs `x-opencode-session`; a normal OpenAI or LiteLLM endpoint does
not. No key or endpoint credential is written to the artifacts.

## Run

From the faultgate checkout, with the pinned tau2 checkout beside it:

```bash
python evals/tau2_retail/run.py \
  --tau2-dir ../tau2-bench \
  --out-dir evals/tau2_retail/artifacts/smoke \
  --task-ids 0,16 \
  --model openai/qwen3.8-flash
```

The smoke pair is retail task `0` (two-item exchange) and `16` (cancel two
orders, return a watch, report the total). The expanded slice is the first ten
tasks of tau2's published `retail_50` stratified subset:

```bash
python evals/tau2_retail/run.py \
  --tau2-dir ../tau2-bench \
  --out-dir evals/tau2_retail/artifacts/retail_50_first10 \
  --task-ids 6,8,11,13,14,16,17,18,19,21 \
  --model openai/qwen3.8-flash
```

Outputs:

| File | Contents |
|---|---|
| `traces.json` | OTLP/JSON, one `invoke_agent` root per task and one `execute_tool` child per live tool call |
| `results.json` | tau2 commit, model, per-task reward/termination/answer and `reward_info` |
| `cost.json` | call count, prompt/completion tokens, models |
| `faultgate-report.json` | faultgate per-check verdicts and reasons |
| `precheck.txt`, `check.txt` | exact CLI output and exit codes |

The exporter deliberately does **not** record tau2's evaluation replay. tau2
replays each trajectory on a fresh environment to check the final database
state; recording that replay would duplicate every write and make faultgate's
`max_calls_per_arg` and `invalidated_by` rules fire on calls the agent never
made.

## Gate

```bash
faultgate precheck evals/tau2_retail/artifacts/smoke/traces.json \
  --policy examples/policy_retail.json | tee evals/tau2_retail/artifacts/smoke/precheck.txt
faultgate check evals/tau2_retail/artifacts/smoke/traces.json \
  --policy examples/policy_retail.json \
  --json evals/tau2_retail/artifacts/smoke/faultgate-report.json \
  | tee evals/tau2_retail/artifacts/smoke/check.txt
```

`check` exits 1 when a check fires and 2 on bad input. Capture `$?` if you
need the exit code in CI.

## Cost

`cost.json` records every model call and the prompt/completion token totals for
the run. The gateway used here did not return a per-call USD price, so the
files contain no dollar estimate; multiply the token totals by the rates of the
provider you use. The two-task smoke made 35 calls and used 189,276 prompt /
8,251 completion tokens over 198 seconds. The ten-task expanded run made 157
calls and used 768,057 prompt / 34,102 completion tokens over 738 seconds.

No model-based faultgate judge was enabled. tau2's NL-assertion judge is a
model call and is included in the token totals.

## How to read the comparison

tau2 and faultgate answer different questions:

- **tau2 reward** is task success: the final database state and, where present,
  the natural-language assertion.
- **faultgate** is a process gate: loops, abstention, policy violations,
  repeated calls, unrecovered tool errors, context overflow. It does not score
  correctness unless `--golden` is supplied.

So the useful cells are the disagreements:

| tau2 | faultgate | reading |
|---|---|---|
| pass | pass | clean run; no known process fault |
| fail | fail | both saw a problem, possibly different ones |
| **pass** | **fail** | faultgate false positive, or a real process fault that did not change the task outcome |
| **fail** | **pass** | task failure outside faultgate's checks (e.g. stopped before acting, wrong business outcome) |

The manual review in `artifacts/*/FINDINGS.md` reads the flagged traces against
tau2's messages before calling either side wrong.
