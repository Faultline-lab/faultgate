<!-- faultgate -->
## faultgate: FAIL

2/10 traces failed · judge none · redaction builtin

**Outcome:** 9/10 true (0.90) · 0 without an outcome · structural clean rate 0.90

| check | fired |
|---|---|
| search_loop | 0 |
| abstention | 0 |
| policy | 0 |
| repeated_call | 0 |
| unrecovered_tool_error | 1 |
| context_overflow | 0 |
| outcome | 1 |

### Failing traces

| scenario | trace | check | reason |
|---|---|---|---|
| retail-16 | `974062dc7ad2` | unrecovered_tool_error | Tool 'find_user_id_by_email' failed at step 1 (Response starts with 'Error:') and no subsequent call to 'find_user_id_by_email' succeeded. |
| retail-18 | `e058f9be20d7` | outcome | outcome false in the outcomes file |

### Reproduce locally

```sh
pip install 'faultgate==0.4.1'
faultgate check faultgate/evals/tau2_retail/artifacts/retail_50_first10/traces.json --policy policy_retail.json --outcomes faultgate/evals/tau2_retail/artifacts/retail_50_first10/outcomes.json
```
