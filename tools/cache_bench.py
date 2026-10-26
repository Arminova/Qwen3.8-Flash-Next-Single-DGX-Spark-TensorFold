"""cache_bench.py BASE MODEL TEXT_DIR [--sizes ...] [--gen 32] [--out FILE]: what the prompt cache saves (stdlib only).

For each size N: one fresh document of about N tokens (documentation text from TEXT_DIR, a unique marker first), sent
three times in one user message: (1) with question A (cold), (2) with question B (a follow-up about the same
document), (3) with question A again (an exact repeat). Each request streams GEN greedy tokens, thinking off; the
row keeps the time to the first token, the prompt tokens and the cached tokens the server reports
(usage.prompt_tokens_details.cached_tokens), and pp = prompt tokens / time to first token for the cold request.
"""
import argparse
import json
import os
import time
import urllib.request
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("model")
ap.add_argument("text_dir")
ap.add_argument("--sizes", default="8192,32768,131072,250000")
ap.add_argument("--gen", type=int, default=32)
ap.add_argument("--max-context", type=int, default=0, help="skip sizes whose prompt + gen exceed this")
ap.add_argument("--out", default="")
args = ap.parse_args()

text = []
for root, _, files in sorted(os.walk(args.text_dir)):
    for f in sorted(files):
        if f.endswith(".md"):
            text.append(open(os.path.join(root, f), errors="ignore").read())
corpus = "\n\n".join(text)
chars_per_token = 3.6
QUESTION_A = "Summarize the documentation above in one paragraph."
QUESTION_B = "List three topics the documentation above covers."


def ask(prompt):
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": args.gen,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{args.base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    usage = {}
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("error"):
                raise RuntimeError(f"server error event: {json.dumps(d['error'])[:300]}")
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    first = first or time.time()
    details = usage.get("prompt_tokens_details") or {}
    return {"ttft_s": round((first or time.time()) - t0, 2), "prompt_tokens": usage.get("prompt_tokens", 0),
            "cached_tokens": details.get("cached_tokens")}


rows = []
for n in [int(x) for x in args.sizes.split(",")]:
    if args.max_context and n + args.gen + 64 > args.max_context:
        print(f"skip {n}: over --max-context {args.max_context}", flush=True)
        continue
    need = int(n * chars_per_token)
    doc = f"[run {uuid.uuid4().hex}] Read this documentation.\n\n" + (corpus * (need // max(1, len(corpus)) + 1))[:need]
    row = {"target": n}
    try:
        row["cold"] = ask(f"{doc}\n\n{QUESTION_A}")
        row["follow_up"] = ask(f"{doc}\n\n{QUESTION_B}")
        row["repeat"] = ask(f"{doc}\n\n{QUESTION_A}")
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    cold = row.get("cold", {})
    if cold.get("prompt_tokens") and cold.get("ttft_s"):
        row["cold_pp_tok_s"] = round(cold["prompt_tokens"] / cold["ttft_s"], 1)
        chars_per_token = chars_per_token * n / cold["prompt_tokens"]
    rows.append(row)
    print(json.dumps(row), flush=True)
if args.out:
    json.dump(rows, open(args.out, "w"), indent=1)
