"""Draft acceptance and decode speed by prompt kind against a running server (stdlib only).

Sends fixed prompts (code, explain, prose, German), one at a time, greedy, thinking off, not streamed, and reports per
kind: completion tokens, decode tok/s, and tokens per round. TensorFold: from each reply's `tensorfold` block (rounds,
drafted, accepted). vLLM: from /metrics before and after each request (spec_decode drafts / accepted tokens); tokens per
round = 1 + accepted / drafts. Prints one JSON line per prompt and one summary line per kind.

  python3 accept_probe.py BASE MODEL [--engine tf|vllm] [--max-tokens 384] [--out FILE]
"""
import argparse
import json
import re
import time
import urllib.request

PROMPTS = [
    ("code", "Write a Python function that parses an ISO 8601 duration string like P3DT4H5M into seconds, with "
             "docstring and unit tests using unittest."),
    ("code", "Implement a thread-safe LRU cache class in Go with Get and Put methods and a small usage example."),
    ("code", "Write a SQL schema for a library (books, authors, loans, members) and three queries: overdue loans, "
             "top authors by loans, and members with no loans."),
    ("explain", "Explain how a GPU executes a matrix multiplication: warps, shared memory tiling and tensor cores."),
    ("explain", "Explain the difference between TCP and QUIC for a web developer, with the trade-offs."),
    ("explain", "How does speculative decoding speed up language model inference? Explain step by step."),
    ("prose", "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle."),
    ("prose", "Write a warm product description for a handmade ceramic coffee mug, about 200 words."),
    ("german", "Erkläre auf Deutsch, wie eine Wärmepumpe funktioniert und wann sie sich für ein Einfamilienhaus lohnt."),
]

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("model")
ap.add_argument("--engine", choices=["tf", "vllm"], default="tf")
ap.add_argument("--max-tokens", type=int, default=384)
ap.add_argument("--out", default="")
args = ap.parse_args()


def spec_counters():
    text = urllib.request.urlopen(f"{args.base}/metrics", timeout=30).read().decode(errors="ignore")
    out = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0}
    names = {"vllm:spec_decode_num_drafts_total": "drafts", "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
             "vllm:spec_decode_num_accepted_tokens_total": "accepted"}
    for line in text.splitlines():
        m = re.match(r"^([a-zA-Z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if m and m.group(1) in names:
            out[names[m.group(1)]] += float(m.group(3))
    return out


def ask(prompt):
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": args.max_tokens,
            "temperature": 0, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{args.base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    return d, time.time() - t0


rows = []
for kind, prompt in PROMPTS:
    before = spec_counters() if args.engine == "vllm" else None
    d, wall = ask(prompt)
    toks = d.get("usage", {}).get("completion_tokens", 0)
    row = {"kind": kind, "tokens": toks, "wall_s": round(wall, 2)}
    if args.engine == "tf":
        tf = d.get("tensorfold", {})
        sp = d.get("speculative") or tf.get("speculative") or tf
        row.update({"tok_s": round(tf.get("tokens_per_second") or 0, 1), "rounds": sp.get("rounds"),
                    "drafted": sp.get("drafted"), "accepted": sp.get("accepted"),
                    "tokens_per_round": round(sp["tokens_per_round"], 2) if sp.get("tokens_per_round") else None,
                    "keys": sorted(d.keys())})
    else:
        after = spec_counters()
        drafts = after["drafts"] - before["drafts"]
        acc = after["accepted"] - before["accepted"]
        row.update({"tok_s": round(toks / wall, 1) if wall > 0 else 0, "rounds": drafts, "drafted":
                    after["draft_tokens"] - before["draft_tokens"], "accepted": acc,
                    "tokens_per_round": round(1 + acc / drafts, 2) if drafts else None})
    rows.append(row)
    print(json.dumps(row), flush=True)

for kind in dict.fromkeys(k for k, _ in PROMPTS):
    rs = [r for r in rows if r["kind"] == kind]
    tpr = [r["tokens_per_round"] for r in rs if r.get("tokens_per_round")]
    tps = [r["tok_s"] for r in rs if r.get("tok_s")]
    print(json.dumps({"summary": kind, "prompts": len(rs), "tokens": sum(r["tokens"] for r in rs),
                      "tokens_per_round": round(sum(tpr) / len(tpr), 2) if tpr else None,
                      "tok_s": round(sum(tps) / len(tps), 1) if tps else None}), flush=True)
if args.out:
    json.dump(rows, open(args.out, "w"), indent=1)
