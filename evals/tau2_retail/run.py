#!/usr/bin/env python3
"""Run selected tau2-bench retail tasks and export OTLP/JSON traces.

The exporter writes the GenAI semantic-convention attributes faultgate reads:
one root ``invoke_agent`` span per task, one ``execute_tool`` child per tool
call.  Credentials are injected only at the LiteLLM transport boundary and are
never written to the output directory.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PINNED_COMMIT = "7aad81c37a3a78b4ed3e4866769f851400c64415"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tau2-dir", required=True, type=Path, help="tau2-bench checkout")
    p.add_argument("--out-dir", required=True, type=Path, help="where traces/results are written")
    p.add_argument("--task-ids", default="0,16", help="comma-separated retail task ids")
    p.add_argument("--model", default=None)
    p.add_argument("--user-model", default=None, help="defaults to --model")
    p.add_argument("--api-base", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--extra-headers", default=None,
                   help='JSON object, e.g. {"x-opencode-session":"..."}')
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--expected-commit", default=PINNED_COMMIT,
                   help="refuse to run against another tau2 commit; empty disables")
    p.add_argument("--allow-other-commit", action="store_true")
    return p.parse_args()


def load_dotenv(tau2_dir: Path) -> None:
    """Load LLM_* / OPENAI_* keys from the checkout's .env without overriding."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    env_file = tau2_dir / ".env"
    if not env_file.exists():
        return
    for key, value in dotenv_values(env_file).items():
        if value and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    args = parse_args()
    tau2_dir = args.tau2_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(tau2_dir / "src"))
    load_dotenv(tau2_dir)

    model = args.model or os.environ.get("LLM_MODEL") or "openai/gpt-4.1-nano"
    user_model = args.user_model or model
    api_base = args.api_base or os.environ.get("LLM_API_BASE") or os.environ.get("OPENAI_API_BASE")
    api_key = args.api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    extra_headers = args.extra_headers or os.environ.get("LLM_EXTRA_HEADERS", "")
    if not api_key or not api_base:
        print("set LLM_API_KEY and LLM_API_BASE (or pass --api-key/--api-base)", file=sys.stderr)
        return 2
    headers = json.loads(extra_headers) if extra_headers else {}

    # Imported after the path is fixed so the pinned checkout wins.
    import litellm
    from loguru import logger
    logger.remove()

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    import tau2.config as tau2_config
    import tau2.evaluator.evaluator_nl_assertions as nl_eval
    import tau2.utils.llm_utils as llm_utils
    from tau2.environment.environment import Environment
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.build import build_agent, build_environment, build_user
    from tau2.runner.helpers import load_tasks
    from tau2.runner.simulation import run_simulation

    nl_eval.DEFAULT_LLM_NL_ASSERTIONS = model
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS = model

    ledger: dict[str, Any] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "models": {}}
    original_completion = llm_utils.completion

    def tracked_completion(*call_args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("api_base", api_base)
        kwargs.setdefault("api_key", api_key)
        kwargs["num_retries"] = 0
        kwargs.setdefault("timeout", args.timeout)
        if headers:
            kwargs.setdefault("extra_headers", headers)
        response = original_completion(*call_args, **kwargs)
        usage = getattr(response, "usage", None)
        with_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        ledger["calls"] += 1
        ledger["prompt_tokens"] += with_tokens
        ledger["completion_tokens"] += out_tokens
        name = str(kwargs.get("model", "unknown"))
        ledger["models"][name] = ledger["models"].get(name, 0) + 1
        cost = getattr(response, "_hidden_params", {}).get("response_cost")
        if isinstance(cost, (int, float)):
            ledger["response_cost_usd"] = round(ledger.get("response_cost_usd", 0.0) + float(cost), 6)
        return response

    llm_utils.completion = tracked_completion
    llm_utils.get_response_cost = lambda response: 0.0  # tracked above; unpriced proxy models are common
    litellm.suppress_debug_info = True

    class MemoryExporter(SpanExporter):
        def __init__(self) -> None:
            self.spans: list[Any] = []

        def export(self, spans: Any) -> SpanExportResult:
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            pass

    exporter = MemoryExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("faultgate-tau2-retail")

    # Capture the authenticated customer id from tau2's own lookup result.  A
    # policy identity rule reads it from the root span.
    state: dict[str, Any] = {"user_id": None, "env": None}
    original_get_response = Environment.get_response

    def instrumented_tool(self: Environment, message: Any) -> Any:
        # Evaluation replays the trajectory on a fresh environment; only the
        # live run's environment belongs in the trace.
        if self is not state.get("env"):
            return original_get_response(self, message)
        with tracer.start_as_current_span("execute_tool") as span:
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.tool.name", message.name)
            span.set_attribute("gen_ai.tool.call.arguments", json.dumps(message.arguments))
            result = original_get_response(self, message)
            span.set_attribute("gen_ai.tool.call.result", result.content or "")
            if result.error:
                span.set_attribute("error.type", "ToolError")
                span.set_status(trace.Status(trace.StatusCode.ERROR, "ToolError"))
            elif message.name.startswith("find_user_id_by"):
                raw = (result.content or "").strip()
                try:
                    value = json.loads(raw)
                except (TypeError, ValueError):
                    value = raw.strip('"')
                if isinstance(value, str) and value:
                    state["user_id"] = value
            return result

    Environment.get_response = instrumented_tool

    def attr_value(value: Any) -> str:
        return value if isinstance(value, str) else json.dumps(value, default=str)

    def to_otlp(spans: list[Any]) -> dict[str, Any]:
        out = []
        for span in spans:
            item: dict[str, Any] = {
                "traceId": format(span.context.trace_id, "032x"),
                "spanId": format(span.context.span_id, "016x"),
                "name": span.name,
                "startTimeUnixNano": str(span.start_time),
                "endTimeUnixNano": str(span.end_time),
                "attributes": [
                    {"key": key, "value": {"stringValue": attr_value(value)}}
                    for key, value in (span.attributes or {}).items()
                ],
                "status": {
                    "code": 2 if span.status.status_code.name == "ERROR" else 0,
                    "message": span.status.description or "",
                },
            }
            if span.parent is not None:
                item["parentSpanId"] = format(span.parent.span_id, "016x")
            out.append(item)
        return {"resourceSpans": [{"scopeSpans": [{"spans": out}]}]}

    tasks = load_tasks("retail", "base")
    wanted = [task_id.strip() for task_id in args.task_ids.split(",") if task_id.strip()]
    by_id = {task.id: task for task in tasks}
    missing = [task_id for task_id in wanted if task_id not in by_id]
    if missing:
        print(f"unknown retail task ids: {missing}", file=sys.stderr)
        return 2
    selected = [by_id[task_id] for task_id in wanted]

    commit = subprocess.check_output(
        ["git", "-C", str(tau2_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if args.expected_commit and commit != args.expected_commit and not args.allow_other_commit:
        print(f"tau2 checkout is {commit}; expected {args.expected_commit} "
              "(pass --allow-other-commit to override)", file=sys.stderr)
        return 2
    started = time.time()
    all_spans: list[Any] = []
    results: list[dict[str, Any]] = []
    trace_file = out_dir / "traces.json"

    for index, task in enumerate(selected, 1):
        exporter.spans = []
        state["user_id"] = None
        env = build_environment("retail")
        state["env"] = env
        agent = build_agent("llm_agent", env, llm=model, llm_args={"num_retries": 0})
        user = build_user("user_simulator", env, task, llm=user_model, llm_args={"num_retries": 0})
        orchestrator = Orchestrator(
            domain="retail", agent=agent, user=user, environment=env, task=task,
            max_steps=args.max_steps, max_errors=10, seed=args.seed, timeout=args.timeout,
        )
        row: dict[str, Any] = {"task_id": task.id, "purpose": task.user_scenario.instructions.reason_for_call}
        started_task = time.time()
        with tracer.start_as_current_span("invoke_agent") as root:
            root.set_attribute("gen_ai.operation.name", "invoke_agent")
            root.set_attribute("gen_ai.agent.name", "tau2-retail")
            root.set_attribute("gen_ai.agent.id", f"tau2-retail-{task.id}")
            root.set_attribute("faultgate.scenario", f"retail-{task.id}")
            root.set_attribute("metadata.scenario", f"retail-{task.id}")
            root.set_attribute("gen_ai.prompt", task.user_scenario.instructions.reason_for_call)
            try:
                simulation = run_simulation(orchestrator)
                messages = simulation.get_messages()
                answers = [m.content for m in messages if m.role == "assistant" and m.content]
                answer = answers[-1] if answers else ""
                root.set_attribute("gen_ai.completion", answer)
                root.set_attribute("faultgate.termination", str(simulation.termination_reason))
                if state["user_id"]:
                    root.set_attribute("enduser.id", state["user_id"])
                reward_info = simulation.reward_info
                row.update({
                    "reward": reward_info.reward if reward_info else None,
                    "termination_reason": str(simulation.termination_reason),
                    "duration_s": round(simulation.duration, 3),
                    "answer": answer,
                    "reward_info": reward_info.model_dump(mode="json") if reward_info else None,
                })
                (out_dir / f"task-{task.id}.simulation.json").write_text(
                    simulation.model_dump_json(indent=2), encoding="utf-8"
                )
                print(f"[{index}/{len(selected)}] task {task.id} reward={row['reward']} "
                      f"termination={row['termination_reason']}", flush=True)
            except Exception as exc:  # keep the other task and the trace
                root.set_status(trace.Status(trace.StatusCode.ERROR, type(exc).__name__))
                root.set_attribute("faultgate.termination", f"agent error: {type(exc).__name__}")
                row.update({"reward": None, "error": type(exc).__name__})
                print(f"[{index}/{len(selected)}] task {task.id} FAILED {type(exc).__name__}", flush=True)
        row["tool_steps"] = sum(1 for span in exporter.spans if span.name == "execute_tool")
        row["wall_s"] = round(time.time() - started_task, 3)
        row["authenticated_user_id"] = state["user_id"]
        results.append(row)
        all_spans.extend(exporter.spans)
        trace_file.write_text(json.dumps(to_otlp(all_spans), indent=2), encoding="utf-8")

    report = {
        "tau2_commit": commit,
        "model": model,
        "user_model": user_model,
        "task_ids": wanted,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": round(time.time() - started, 3),
        "tasks": results,
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "cost.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"wrote {trace_file} ({len(all_spans)} spans), results.json, cost.json", flush=True)
    provider.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
