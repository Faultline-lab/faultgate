# tau2-bench retail: faultgate vs tau2

Two runs at tau2-bench `7aad81c37a3a78b4ed3e4866769f851400c64415`, model
`openai/qwen3.8-flash` for the agent, user simulator, and NL-assertion judge.
The exact commands are in `README.md`.

## Smoke test: 2 tasks

Tasks `0` and `16`, chosen before the run: task `0` exercises a two-item
exchange and task `16` exercises two cancellations, one return, and a computed
refund total.

| task | tau2 reward | termination | tool steps | faultgate | agreement |
|---|---|---|---|---|---|
| 0 | 1.0 | user_stop | 8 | pass | yes (both clean) |
| 16 | 1.0 | user_stop | 8 | pass | yes (both clean) |

`faultgate precheck`: 2 of 2 traces, 16 tool steps read, 0 errors, 0 warnings.
`faultgate check --policy examples/policy_retail.json`: 0 of 2 traces failed,
every check `ok 1.00`, exit 0.

Manual review of the two traces:

- **Task 0.** The agent found `yusuf_rossi_9620`, read the order and both
  products, then made one `exchange_delivered_order_items` call covering both
  items. The order was `delivered`, the call carried the matching `order_id`,
  and the exchange used the original payment method. tau2's DB check and the
  empty NL-assertion set both scored 1.0. There is no faultgate finding to
  disagree with.
- **Task 16.** The agent read all three orders, calculated
  `3131.10 + 4777.75 + 367.38 = 8276.23`, cancelled the two pending orders with
  the allowed reason `no longer needed`, and returned only the watch from the
  delivered order. Each write was preceded by the matching `get_order_details`
  call with the same `order_id`; no order was written twice. tau2's DB check
  and the $8,276.23 NL assertion both scored 1.0.

No disagreements in the smoke test.

### Trace-capture pitfall found during the smoke test

The first version of the runner instrumented `Environment.get_response` for
the whole `run_simulation` call. tau2 evaluates a run by replaying its
trajectory on a fresh environment, so the trace contained a second copy of
every write. faultgate correctly flagged those duplicate spans as
`max_calls_per_arg` violations and as writes that invalidated their own
`get_order_details` prerequisite. The fix records only the live environment
(`run.py`, `state["env"]`), and the policy findings disappeared without
changing the checker. The stale two-task artifact was discarded.

This is a reminder for any tau2 integration: capture the simulation, not the
evaluation replay.

## Expanded run

The first ten tasks of tau2's published `retail_50` stratified subset:
`6,8,11,13,14,16,17,18,19,21`. The machine-readable join is
`artifacts/retail_50_first10/comparison.json`; the CLI table is
`comparison.txt`.

| task | tau2 reward | faultgate | finding |
|---|---|---|---|
| 6 | 1.0 | pass | - |
| 8 | 1.0 | pass | - |
| 11 | 1.0 | pass | - |
| 13 | 1.0 | pass | - |
| 14 | 1.0 | pass | - |
| 16 | 1.0 | **FAIL** | `unrecovered_tool_error` (reviewed below) |
| 17 | 1.0 | pass | - |
| 18 | **0.0** | pass | tau2 DB check failed (reviewed below) |
| 19 | 1.0 | pass | - |
| 21 | 1.0 | pass | - |

tau2 passed 9 of 10 (mean reward 0.9); faultgate passed 9 of 10; the two
systems agreed on 8 of 10. There is one disagreement in each direction.

### Disagreement A: task 16, tau2 pass / faultgate FAIL

Trace `974062dc7ad2621aad5ccf62fb2fdbb5`.

Step 1 called `find_user_id_by_email` with
`sofia.doe_496@example.com` before the user had given an email. The tool
returned `Error: User not found`. The agent then asked for identity details,
received Fatima Johnson's name and ZIP, and succeeded with
`find_user_id_by_name_zip`. It went on to read all three orders, cancel the two
pending ones, and return only the watch. tau2 scored the final DB state and the
$8,276.23 assertion at 1.0.

Faultgate's `unrecovered_tool_error` fires because a tool errored and no later
call **to the same tool** succeeded. That is not what happened here: the agent
recovered through the other identity tool. In the retail domain the two
lookups are alternative ways to establish the same principal, so the verdict
is a false positive for "unrecovered".

The flag is still useful: the fabricated email is a genuine bad step that
tau2's task reward ignored. But the reason is wrong. A future rule could treat
a configured group of equivalent tools as one recovery unit, or check whether
the intent (identify the user) succeeded rather than the same function name.

### Disagreement B: task 18, tau2 fail / faultgate pass

Trace `e058f9be20d72c97aa15a745f024edee`.

The agent found the office chair, quoted the return, and the user changed their
mind to an exchange. The agent described the exchange and asked for
confirmation. The user replied "Yeah, go ahead. I'll take the same one." with
`###STOP###` appended; tau2 ended the simulation on that user stop before the
agent got another turn. No `exchange_delivered_order_items` call was made, so
the database did not change and tau2 returned 0.0. The NL assertion still
scored 1.0.

Faultgate passes because there is no loop, abstention, policy violation,
repeated call, tool error, or overflow. The failure is a turn-taking artifact
of the half-duplex user simulator, not a process fault in faultgate's list. A
single sample cannot tell us whether the agent would have acted on a real
follow-up turn. This is the expected scope limit: a clean faultgate verdict is
not a task-success claim.
