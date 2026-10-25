"""pp / tg at several context sizes against an OpenAI-compatible server (stdlib only).

For each target size N: a prompt of about N tokens (documentation text from TEXT_DIR, with a unique marker first so no
prompt cache can serve it), streamed, greedy, GEN new tokens. pp = the server's prompt tokens / time to the first
streamed token; tg = (streamed tokens - 1) / (time of the last token - time of the first). Thinking is asked off
(chat_template_kwargs enable_thinking false); reasoning deltas count as tokens if a server streams them anyway.

  python3 pptg_bench.py BASE MODEL TEXT_DIR --sizes 4096,16384,32768,65536,131072,262144 [--gen 256] [--out FILE]
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("model")
ap.add_argument("text_dir")
ap.add_argument("--sizes", default="4096,16384,32768,65536,131072,262144")
ap.add_argument("--gen", type=int, default=256)
ap.add_argument("--max-context", type=int, default=0, help="skip sizes whose prompt + gen exceed this")
ap.add_argument("--out", default="")
args = ap.parse_args()

text = []
for root, _, files in sorted(os.walk(args.text_dir)):
    for f in sorted(files):
        if f.endswith(".md"):
            text.append(open(os.path.join(root, f), errors="ignore").read())
corpus = "\n\n".join(text)
print(f"corpus: {len(corpus)} chars from {len(text)} files", flush=True)
chars_per_token = 3.6


def request(n_tokens):
    need = int(n_tokens * chars_per_token)
    body_text = (corpus * (need // max(1, len(corpus)) + 1))[:need]
    prompt = (f"[run {uuid.uuid4().hex}] Read this documentation, then summarize it.\n\n{body_text}\n\n"
              f"Summarize the documentation above in about 300 words.")
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": args.gen,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{args.base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = last = None
    count = 0
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
                    now = time.time()
                    first = first or now
                    last = now
                    count += 1
    ptoks = usage.get("prompt_tokens", 0)
    ctoks = usage.get("completion_tokens", count)
    ttft = (first or time.time()) - t0
    tg = (ctoks - 1) / (last - first) if first and last and last > first else 0.0
    return ptoks, ctoks, ttft, tg


rows = []
for n in [int(x) for x in args.sizes.split(",")]:
    if args.max_context and n + args.gen > args.max_context:
        print(f"skip {n}: over --max-context {args.max_context}", flush=True)
        continue
    try:
        p, c, ttft, tg = request(n)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")[:300]
        print(f"{n}: FAILED HTTP {e.code}: {detail}", flush=True)
        rows.append({"target": n, "error": f"HTTP {e.code}: {detail}"})
        continue
    except Exception as e:
        print(f"{n}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
        rows.append({"target": n, "error": str(e)[:200]})
        continue
    if not p:
        print(f"{n}: FAILED no prompt tokens reported (ttft {ttft:.2f}s)", flush=True)
        rows.append({"target": n, "error": "no prompt tokens reported"})
        continue
    if p:
        chars_per_token = chars_per_token * n / p  # aim the next sizes better
    row = {"target": n, "prompt_tokens": p, "completion_tokens": c, "ttft_s": round(ttft, 2),
           "pp_tok_s": round(p / ttft, 1) if ttft > 0 else 0, "tg_tok_s": round(tg, 1)}
    rows.append(row)
    print(json.dumps(row), flush=True)
if args.out:
    json.dump(rows, open(args.out, "w"), indent=1)
