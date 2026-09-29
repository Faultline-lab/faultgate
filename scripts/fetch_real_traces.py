"""Fetch and sample real agent traces from cached Exgentic dataset shards into OTLP JSON format.

Samples --n traces spread evenly across the 6 benchmarks (appworld, browsecompplus,
swebench, tau2_airline, tau2_retail, tau2_telecom) via seeded round-robin selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime
import glob
import json
import os
from pathlib import Path
import random
from typing import Any

import pyarrow.parquet as pq
import huggingface_hub

REVISION = "4b8ad4ab198438e5a170f9171c19c6a2cf7c1814"


def to_otlp_value(val: Any) -> dict[str, Any]:
    if isinstance(val, bool):
        return {"boolValue": val}
    if isinstance(val, int):
        return {"intValue": str(val)}
    if isinstance(val, float):
        return {"doubleValue": val}
    if isinstance(val, list):
        return {"arrayValue": {"values": [{"stringValue": str(x)} for x in val]}}
    return {"stringValue": str(val)}


def span_to_otlp(sp: dict[str, Any], benchmark: str) -> dict[str, Any]:
    attr_list: list[dict[str, Any]] = []
    attrs = sp.get("attributes") or {}
    for k, v in attrs.items():
        if v is not None:
            attr_list.append({"key": k, "value": to_otlp_value(v)})

    attr_list.append({"key": "faultgate.scenario", "value": {"stringValue": benchmark}})
    attr_list.append({"key": "gen_ai.agent.name", "value": {"stringValue": f"agent-{benchmark}"}})

    st_raw = sp.get("start_time")
    if st_raw:
        try:
            dt = datetime.datetime.fromisoformat(st_raw)
            start_ns = str(int(dt.timestamp() * 1e9))
        except Exception:
            start_ns = "0"
    else:
        start_ns = "0"

    span_dict: dict[str, Any] = {
        "traceId": sp["trace_id"],
        "spanId": sp["span_id"],
        "name": sp.get("name", ""),
        "startTimeUnixNano": start_ns,
        "attributes": attr_list,
    }
    if sp.get("parent_span_id"):
        span_dict["parentSpanId"] = sp["parent_span_id"]
    status_raw = sp.get("status")
    if isinstance(status_raw, dict) and "code" in status_raw and status_raw["code"] is not None:
        try:
            st_code = int(status_raw["code"])
        except ValueError:
            st_code = 2 if "ERROR" in str(status_raw["code"]) else (1 if "OK" in str(status_raw["code"]) else 0)
        st_obj: dict[str, Any] = {"code": st_code}
        if status_raw.get("message") is not None:
            st_obj["message"] = str(status_raw["message"])
        span_dict["status"] = st_obj
    return span_dict


def fetch_traces(
    n: int = 30,
    out_path: str | Path = ".demo/real_traces.json",
    seed: int = 42,
    hf_cache: str | Path | None = None,
) -> Path:
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    snapshot_path = huggingface_hub.snapshot_download(
        repo_id="Exgentic/agent-llm-traces-v2",
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=["data/train/*.parquet"],
        cache_dir=hf_cache,
    )
    files = sorted(glob.glob(f"{snapshot_path}/data/train/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards found in {snapshot_path}/data/train/")

    by_bench: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for f in files:
        t = pq.read_table(f, columns=["benchmark"])
        b_list = t["benchmark"].to_pylist()
        for row_idx, b in enumerate(b_list):
            by_bench[b].append((f, row_idx, b))

    benchmarks = sorted(by_bench.keys())
    rng = random.Random(seed)
    for b in benchmarks:
        rng.shuffle(by_bench[b])

    selected: list[tuple[str, int, str]] = []
    ptrs = {b: 0 for b in benchmarks}
    while len(selected) < n:
        added = 0
        for b in benchmarks:
            if len(selected) >= n:
                break
            if ptrs[b] < len(by_bench[b]):
                selected.append(by_bench[b][ptrs[b]])
                ptrs[b] += 1
                added += 1
        if added == 0:
            break

    by_shard: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for shard_path, row_idx, b in selected:
        by_shard[shard_path].append((row_idx, b))

    all_spans: list[dict[str, Any]] = []
    for shard_path, row_items in by_shard.items():
        pf = pq.ParquetFile(shard_path)
        offsets = []
        curr = 0
        for rg in range(pf.num_row_groups):
            nrows = pf.metadata.row_group(rg).num_rows
            offsets.append((curr, curr + nrows, rg))
            curr += nrows

        for row_idx, benchmark in row_items:
            for s, e, rg in offsets:
                if s <= row_idx < e:
                    tbl = pf.read_row_group(rg, columns=["spans"])
                    spans_py = tbl["spans"][row_idx - s].as_py()
                    for sp in spans_py:
                        all_spans.append(span_to_otlp(sp, benchmark))
                    break

    doc = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": all_spans,
                    }
                ]
            }
        ]
    }

    out_p.write_text(json.dumps(doc), encoding="utf-8")
    counts = Counter(b for _, _, b in selected)
    print(f"Wrote {len(selected)} traces ({len(all_spans)} spans) to {out_p}")
    print(f"Per-benchmark counts: {dict(sorted(counts.items()))}")
    return out_p


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and sample Exgentic traces across benchmarks")
    parser.add_argument("--n", type=int, default=30, help="Number of traces to fetch (default: 30)")
    parser.add_argument(
        "--out",
        type=str,
        default=".demo/real_traces.json",
        help="Output OTLP JSON path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()
    fetch_traces(n=args.n, out_path=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
