"""Run two pinned τ-bench retail tasks and check their recorded traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TAU2_COMMIT = "ebf990be93b594a08ed23cb8b7a9773cb9a03678"
DEFAULT_TASK_IDS = ("17", "3")
DEFAULT_MODEL = "openai/gpt-6-luna"


def stamp(value: str | None) -> str:
    if not value:
        return "0"
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return str(int(parsed.timestamp() * 1_000_000_000))


def attr(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    elif value is None:
        value = ""
    return {"key": key, "value": {"stringValue": str(value)}}


def text_messages(messages: list[dict[str, Any]]) -> str:
    return json.dumps(messages, ensure_ascii=False)


def to_otlp(results: dict[str, Any]) -> dict[str, Any]:
    spans: list[dict[str, Any]] = []
    for simulation in results.get("simulations", []):
        messages = simulation.get("messages") or []
        identifier = str(simulation.get("id") or simulation.get("task_id"))
        trace_id = hashlib.sha256(identifier.encode()).hexdigest()[:32]
        root_span_id = "0000000000000001"
        dialog = [
            {
                "role": message.get("role", ""),
                "content": message.get("content"),
                "tool_calls": message.get("tool_calls"),
            }
            for message in messages
            if message.get("role") in {"assistant", "user", "tool"}
        ]
        final_answer = next(
            (
                message.get("content")
                for message in reversed(messages)
                if message.get("role") == "assistant" and message.get("content")
            ),
            "",
        )
        root_attrs = [
            attr("gen_ai.operation.name", "invoke_agent"),
            attr("gen_ai.agent.name", "tau2-retail-llm-agent"),
            attr("gen_ai.input.messages", text_messages(dialog)),
            attr("gen_ai.output.messages", text_messages([{"role": "assistant", "content": final_answer}])),
            attr("faultgate.scenario", simulation.get("task_id", "")),
            attr("tau2.reward", (simulation.get("reward_info") or {}).get("reward")),
        ]
        spans.append(
            {
                "traceId": trace_id,
                "spanId": root_span_id,
                "name": "invoke_agent",
                "startTimeUnixNano": stamp(simulation.get("start_time")),
                "endTimeUnixNano": stamp(simulation.get("end_time")),
                "attributes": root_attrs,
            }
        )

        tool_results = {
            message.get("id"): message
            for message in messages
            if message.get("role") == "tool" and message.get("id")
        }
        span_number = 2
        for message in messages:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                call_id = call.get("id")
                result = tool_results.get(call_id, {})
                call_time = message.get("timestamp") or simulation.get("start_time")
                result_time = result.get("timestamp") or call_time
                span_attrs = [
                    attr("gen_ai.operation.name", "execute_tool"),
                    attr("gen_ai.tool.name", call.get("name", "")),
                    attr("gen_ai.tool.call.arguments", call.get("arguments", {})),
                    attr("gen_ai.tool.call.result", result.get("content", "")),
                ]
                spans.append(
                    {
                        "traceId": trace_id,
                        "spanId": f"{span_number:016x}",
                        "parentSpanId": root_span_id,
                        "name": "execute_tool",
                        "startTimeUnixNano": stamp(call_time),
                        "endTimeUnixNano": stamp(result_time),
                        "attributes": span_attrs,
                        **(
                            {"status": {"code": 2, "message": "τ-bench tool error"}}
                            if result.get("error")
                            else {}
                        ),
                    }
                )
                span_number += 1
    return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}


def run(command: list[str], cwd: Path, log_prefix: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    log_prefix.with_suffix(".stdout.txt").write_text(result.stdout, encoding="utf-8")
    log_prefix.with_suffix(".stderr.txt").write_text(result.stderr, encoding="utf-8")
    return result


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-root", type=Path, default=project_root / ".vendor" / "tau2-bench-src")
    parser.add_argument("--run-id", default="smoke-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--task-ids", nargs=2, default=list(DEFAULT_TASK_IDS), metavar=("ID1", "ID2"))
    parser.add_argument("--agent-llm", default=DEFAULT_MODEL)
    parser.add_argument("--user-llm", default=DEFAULT_MODEL)
    parser.add_argument("--reuse-results", type=Path, help="Reuse an existing τ-bench results.json; makes no model calls")
    args = parser.parse_args()

    api_key = os.environ.get("AICREDITS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.reuse_results:
        parser.error("set AICREDITS_API_KEY (or OPENAI_API_KEY) in the environment")
    # τ-bench calls LiteLLM's OpenAI provider; the existing Faultgate recipe uses
    # AICredits' OpenAI-compatible endpoint and key.
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    if os.environ.get("AICREDITS_API_KEY"):
        os.environ.setdefault("OPENAI_API_BASE", os.environ.get("AICREDITS_BASE_URL", "https://aicredits.in/v1"))
    os.environ.setdefault("PYTHONUTF8", "1")

    tau_root = args.tau2_root.resolve()
    git = shutil.which("git")
    if not git:
        parser.error("git is required")
    revision = subprocess.check_output([git, "-C", str(tau_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != TAU2_COMMIT:
        parser.error(f"tau-bench must be checked out at {TAU2_COMMIT}; found {revision}")

    stamp_id = args.run_id
    tau_output = tau_root / "data" / "simulations" / stamp_id
    if tau_output.exists() and not args.reuse_results:
        parser.error(f"tau-bench output already exists: {tau_output}")
    artifacts = project_root / "integrations" / "tau2" / "runs" / stamp_id
    artifacts.mkdir(parents=True, exist_ok=bool(args.reuse_results))
    tau_cli = Path(sys.executable).with_name("tau2.exe" if os.name == "nt" else "tau2")
    tau_command = [
        str(tau_cli), "run", "--domain", "retail", "--num-trials", "1",
        "--agent-llm", args.agent_llm, "--user-llm", args.user_llm,
        "--task-ids", *args.task_ids, "--save-to", stamp_id,
    ]
    if args.reuse_results:
        results_path = args.reuse_results.resolve()
    else:
        tau_run = run(tau_command, tau_root, artifacts / "tau2")
        (artifacts / "tau2-command.json").write_text(
            json.dumps({"command": tau_command, "commit": revision, "task_ids": args.task_ids}, indent=2),
            encoding="utf-8",
        )
        if tau_run.returncode:
            print(f"tau-bench failed with exit code {tau_run.returncode}; see {artifacts}", file=sys.stderr)
            return tau_run.returncode
        results_path = tau_output / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    selected = [s for s in results.get("simulations", []) if str(s.get("task_id")) in args.task_ids]
    if {str(s.get("task_id")) for s in selected} != set(args.task_ids):
        raise RuntimeError(f"Expected tasks {args.task_ids}; τ-bench saved {[s.get('task_id') for s in selected]}")

    trace_path = artifacts / "traces.otlp.json"
    trace_path.write_text(json.dumps(to_otlp({"simulations": selected}), indent=2, ensure_ascii=False), encoding="utf-8")
    task_costs = []
    for simulation in selected:
        task_costs.append({
            "task_id": simulation.get("task_id"),
            "tau2_reward": (simulation.get("reward_info") or {}).get("reward"),
            "agent_cost_usd": simulation.get("agent_cost"),
            "user_cost_usd": simulation.get("user_cost"),
            "duration_seconds": simulation.get("duration"),
            "tool_calls": sum(len(m.get("tool_calls") or []) for m in simulation.get("messages", [])),
        })
    known_costs = [
        value
        for task in task_costs
        for value in (task["agent_cost_usd"], task["user_cost_usd"])
        if isinstance(value, (int, float))
    ]
    cost_report = {
        "upstream_commit": revision,
        "agent_model": args.agent_llm,
        "user_model": args.user_llm,
        "tasks": task_costs,
        "recorded_known_agent_and_user_cost_usd": sum(known_costs),
        "tasks_with_unavailable_agent_cost": [t["task_id"] for t in task_costs if t["agent_cost_usd"] is None],
        "cost_note": "τ-bench returned user_cost but agent_cost was null for both runs, so this is a lower bound, not total API spend. Post-hoc evaluator charges, if any, are not included.",
    }
    (artifacts / "findings.json").write_text(json.dumps(cost_report, indent=2), encoding="utf-8")

    faultgate = Path(sys.executable).with_name("faultgate.exe" if os.name == "nt" else "faultgate")
    precheck = run(
        [str(faultgate), "precheck", str(trace_path), "--sample", "0", "--policy", str(project_root / "examples" / "policy_retail.json")],
        project_root, artifacts / "faultgate-precheck",
    )
    check = run(
        [str(faultgate), "check", str(trace_path), "--policy", str(project_root / "examples" / "policy_retail.json"), "--json", str(artifacts / "faultgate-report.json")],
        project_root, artifacts / "faultgate-check",
    )
    print(f"tau-bench artifacts: {artifacts}")
    print(f"tau-bench rewards and recorded costs: {artifacts / 'findings.json'}")
    print(f"Faultgate precheck exit: {precheck.returncode}")
    print(f"Faultgate check exit: {check.returncode}")
    print(check.stdout.rstrip())
    return 0 if precheck.returncode == 0 else precheck.returncode


if __name__ == "__main__":
    raise SystemExit(main())
