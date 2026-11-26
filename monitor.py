import json
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path


class Monitor:
    def __init__(self, path="metrics.jsonl"):
        self.path = Path(path)
        self.cur = None


    def start_turn(self, session_id: str):
        self.cur = {
            "turn_id": uuid.uuid4().hex[:8],
            "session": session_id[:8],
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "path": [], "node_ms": {}, "tools": [], "flags": [], "human_wait_ms": 0,
        }
        self._t0 = time.perf_counter()

    def end_turn(self, state=None, usage=None, error=None):
        c, usage = self.cur, usage or {}
        c["total_ms"] = round((time.perf_counter() - self._t0) * 1000)
        c["agent_ms"] = c["total_ms"] - c["human_wait_ms"]
        c["input_tokens"] = sum(u.get("input_tokens", 0) for u in usage.values())
        c["output_tokens"] = sum(u.get("output_tokens", 0) for u in usage.values())
        c["status"] = "error" if error else (state or {}).get("status", "resolved")
        if error:
            c["error"] = f"{type(error).__name__}: {error}"
        with self.path.open("a") as f:
            f.write(json.dumps(c) + "\n")
        print(f"   [monitor] {c['status']} | intent={c.get('intent')} | {c['agent_ms']} ms | "
              f"tokens {c['input_tokens']}/{c['output_tokens']} | {' > '.join(c['path'])}"
              + (f" | flags={c['flags']}" if c["flags"] else ""))
        self.cur = None


    def track(self, name):
        def deco(fn):
            @wraps(fn)
            def wrapper(state):
                t = time.perf_counter()
                try:
                    return fn(state)
                finally:
                    if self.cur is not None:
                        self.cur["node_ms"][name] = round((time.perf_counter() - t) * 1000)
                        if name not in self.cur["path"]:
                            self.cur["path"].append(name)
            return wrapper
        return deco

    def log(self, **fields):
        if self.cur is not None:
            self.cur.update(fields)

    def flag(self, name):
        if self.cur is not None and name not in self.cur["flags"]:
            self.cur["flags"].append(name)

    @contextmanager
    def human_wait(self):
        t = time.perf_counter()
        yield
        self.cur["human_wait_ms"] += round((time.perf_counter() - t) * 1000)


def dashboard(path="metrics.jsonl"):
    p = Path(path)
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []
    if not rows:
        return print("No metrics yet — chat with the bot first.")

    n = len(rows)
    lat = sorted(r["agent_ms"] for r in rows)
    pct = lambda q: lat[min(n - 1, int(q * n))]
    status = Counter(r["status"] for r in rows)
    tokens_in = sum(r["input_tokens"] for r in rows)
    tokens_out = sum(r["output_tokens"] for r in rows)
    node_ms = defaultdict(list)
    for r in rows:
        for node, ms in r["node_ms"].items():
            node_ms[node].append(ms)
    rag = [r["rag_top_score"] for r in rows if "rag_top_score" in r]
    rate = lambda k: f"{status[k] / n:.0%}"
    line = "-" * 52

    print(f"\n{line}\n REFUND AGENT DASHBOARD   ({n} turns, {len({r['session'] for r in rows})} sessions)\n{line}")
    print(f" Success (no errors) : {1 - status['error'] / n:.0%}")
    print(f" Resolved by bot     : {rate('resolved')}   Escalated: {rate('escalated')}   "
          f"Blocked: {rate('blocked')}   Errors: {status['error']}")
    print(f" Latency (excl human): avg {sum(lat) // n} ms | p50 {pct(.5)} | p95 {pct(.95)} | max {lat[-1]}")
    print(f" Tokens              : {tokens_in:,} in / {tokens_out:,} out | avg {(tokens_in + tokens_out) // n}/turn")
    if rag:
        print(f" RAG top score       : avg {sum(rag) / len(rag):.2f} | low-confidence {sum('low_rag_confidence' in r['flags'] for r in rows)}")
    print(f" Intents             : {dict(Counter(r.get('intent') for r in rows))}")
    print(f" Tool calls          : {dict(Counter(t for r in rows for t in r['tools']))}")
    print(f" Guardrails/events   : {dict(Counter(f for r in rows for f in r['flags']))}")
    print(f" Human decisions     : {dict(Counter(r['human_decision'] for r in rows if 'human_decision' in r))}")
    print(" Avg ms per node     : " + ", ".join(f"{k} {sum(v) // len(v)}" for k, v in node_ms.items()))
    errors = [r for r in rows if r["status"] == "error"][-3:]
    for r in errors:
        print(f" ! {r['ts']} {r['error'][:80]}")
    print(line)


if __name__ == "__main__":
    import sys
    dashboard(sys.argv[1] if len(sys.argv) > 1 else "metrics.jsonl")
