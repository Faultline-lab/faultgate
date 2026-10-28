# Two-task τ-bench / Faultgate smoke findings

- τ-bench commit: `ebf990be93b594a08ed23cb8b7a9773cb9a03678`
- Agent and user model: `openai/gpt-6-luna`
- Tasks: retail `17` and `3`, one trial each
- τ-bench rewards: `1.0` / `1.0` (both database checks passed)
- Faultgate precheck: OK, 2 traces and 13 tool steps, no warnings
- Faultgate check: 0/2 traces failed; no policy or reliability check fired
- Recorded user-simulator cost: `$0.0008312` total
- Agent cost: unavailable (`null`) on both tasks; total API spend is therefore unknown
- Faultgate judge: none

## Manual review

There was no disagreement in the overall task pass/fail outcome: both τ-bench rewards were 1.0 and both Faultgate checks were clean. The per-action diagnostics were `3/6` for task 17 and `11/12` for task 3. The unmatched reference actions were three reads for task 17 (user details and two order lookups) and one product lookup for task 3. τ-bench's reward basis for these tasks was database state plus natural-language assertions, so those action diagnostics did not reduce the task reward. Faultgate checks policy compliance and selected trace failure modes; it does not compare the run against every reference action. The final database state was correct in both runs.

This smoke run does not establish detection rates. It does show that Faultgate's clean policy verdict cannot substitute for τ-bench's task correctness checks or exact action diagnostics.
