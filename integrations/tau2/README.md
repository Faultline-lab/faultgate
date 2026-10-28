# τ-bench retail smoke run

This recipe runs the public τ-bench retail `llm_agent` on two fixed tasks, exports the full conversation and tool-call history to OTLP/JSON, and checks it with Faultgate's retail policy. The smoke run uses task IDs `17` and `3` from the pinned τ-bench checkout.

## Pinned source

- Repository: <https://github.com/sierra-research/tau2-bench>
- Commit: `ebf990be93b594a08ed23cb8b7a9773cb9a03678`
- Faultgate policy: [`../../examples/policy_retail.json`](../../examples/policy_retail.json)

The runner refuses to start against a different τ-bench commit. The upstream source and the virtual environment are local-only and excluded from Git.

## Setup (Windows PowerShell)

From the Faultgate repository root:

```powershell
git clone https://github.com/sierra-research/tau2-bench.git .vendor/tau2-bench-src
git -C .vendor/tau2-bench-src checkout ebf990be93b594a08ed23cb8b7a9773cb9a03678
py -3.13 -m venv .venv-tau2
\.venv-tau2\Scripts\python.exe -m pip install audioop-lts
\.venv-tau2\Scripts\python.exe -m pip install -e .vendor\tau2-bench-src -e .
```

τ-bench requires Python 3.12–3.13. Python 3.13 no longer bundles `audioop`, so this setup adds the `audioop-lts` compatibility package required by τ-bench's import path.

The runner uses the existing `AICREDITS_API_KEY` variable and the OpenAI-compatible AICredits endpoint used by Faultgate's other LiteLLM recipes. Set the key in the shell environment; do not put it in a file or commit it. An `OPENAI_API_KEY` can also be used with the standard OpenAI endpoint.

## Run the two-task smoke test

```powershell
$env:AICREDITS_API_KEY = Read-Host "AICredits API key"
\.venv-tau2\Scripts\python.exe integrations\tau2\run.py --run-id smoke-20261028 --task-ids 17 3
Remove-Item Env:AICREDITS_API_KEY
```

The default agent and user simulator are both `openai/gpt-6-luna`; override with `--agent-llm` and `--user-llm` if needed. The run makes live model calls. It does not run the full benchmark.

The runner saves τ-bench console output and the exact command under `integrations/tau2/runs/<run-id>/`, plus `traces.otlp.json`, `faultgate-report.json`, and `findings.json`. The raw τ-bench `results.json` remains under the ignored upstream checkout at `.vendor/tau2-bench-src/data/simulations/<run-id>/results.json`.

To regenerate the OTLP trace and Faultgate reports without another model call:

```powershell
\.venv-tau2\Scripts\python.exe integrations\tau2\run.py --run-id smoke-20261028 --task-ids 17 3 --reuse-results .vendor\tau2-bench-src\data\simulations\smoke-20261028\results.json
```

## Smoke result and manual review

The committed run used the pinned commit and tasks `17` and `3`. τ-bench returned reward `1.0` for both tasks, with the database check passing for both. Faultgate precheck recognized all 13 tool steps across the two traces with no warnings. Faultgate check returned clean for both traces: no policy, search-loop, abstention, repeated-call, unrecovered-tool-error, or context-overflow finding.

Manual review found no disagreement in the overall pass/fail result. τ-bench's action-match diagnostics were `3/6` for task `17` and `11/12` for task `3`, while its overall reward remained `1.0` for both because the reward basis was database state and natural-language assertions. The unmatched reference actions were three reads for task `17` (user details and two order lookups) and one product lookup for task `3`. The agent still reached the expected final database state. Faultgate's clean result is consistent with its scope: the retail policy passed, but Faultgate does not score exact reference-action coverage. A clean Faultgate policy verdict should not be read as a claim that every expected information-gathering action was taken.

## Cost

The run took about 95 seconds of wall time across two tasks. τ-bench recorded user-simulator cost of `$0.0008312` total. It returned `null` for both agent-cost fields, so the recorded amount is only a lower bound and the total model/API spend is unknown. The run does not include any separate provider billing fees; check the provider's usage page for the complete charge. No model-based Faultgate judge was enabled.

## Version and limitations

This is a two-task smoke test, not a benchmark score or a statistically meaningful estimate of agent quality. τ-bench's task reward and Faultgate's policy/reliability checks measure different things. The exported traces retain the benchmark's synthetic conversation and tool results so reviewers can inspect each call.
