#!/usr/bin/env python3
"""bench.py BASE MODEL decode|greedy|ident OUT [--streams 1,3] [--reps 3] [--gen 384]
decode: mixed load (code / explanation / prose prompts, temperature 1.0, own seed per stream, thinking off); N streams
        at once, per-stream rate = (tokens-1)/(last chunk - first chunk), aggregate = sum over streams. Median of reps.
greedy: the same greedy request (code, then chat) once and three at once, 256 tokens, aggregate tok/s, median of reps.
ident:  greedy replies (thinking off, max 256 tokens) to fixed prompts, one at a time, then three at once; stores the
        text's sha256 so two builds can be compared token for token (the text is the tokens).
Stdlib only."""
import argparse, hashlib, json, statistics, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("mode"); ap.add_argument("out")
ap.add_argument("--streams", default="1,3"); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--gen", type=int, default=384)
a = ap.parse_args()

PROMPTS = [
    "Write a Python module that implements an LRU cache class with get, put and a size limit, then a small pytest file that tests eviction order, updates and capacity one. Show both files in full.",
    "Explain how a mixture-of-experts transformer routes tokens to experts, why load balancing matters, and what changes when it is served on one GPU with a small memory budget. Use several paragraphs.",
    "Write a long, vivid short story about a lighthouse keeper who finds a message in a bottle that was written in her own handwriting. Keep going until the story reaches a satisfying end.",
]
IDENT = PROMPTS + [
    "Write a C function that reverses a singly linked list in place, and explain its complexity.",
    "What are the differences between TCP and UDP? Answer in a short list.",
    "Translate into German: 'The quick brown fox jumps over the lazy dog, and then goes back to sleep under the old oak tree.'",
    "List the first 30 prime numbers, then say which of them are twin primes.",
    "Summarise the plot of Romeo and Juliet in five sentences.",
]

def stream(prompt, seed=None, temp=1.0, gen=384, res=None, idx=0):
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": gen, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": temp, "chat_template_kwargs": {"enable_thinking": False}}
    if temp > 0:
        body.update({"top_p": 0.95, "top_k": 20})
        if seed is not None: body["seed"] = seed
    req = urllib.request.Request(a.base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; text = []; usage = None; n = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"): continue
            d = line[5:].strip()
            if d == b"[DONE]": break
            j = json.loads(d)
            if j.get("usage"): usage = j["usage"]
            for c in j.get("choices", []):
                t = c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content")
                if t:
                    now = time.time()
                    if first is None: first = now
                    last = now; text.append(t); n += 1
    toks = (usage or {}).get("completion_tokens", n)
    res[idx] = {"tokens": toks, "ttft": (first or t0) - t0, "rate": (toks - 1) / (last - first) if last and first and last > first else 0.0,
                "text": "".join(text), "prompt_tokens": (usage or {}).get("prompt_tokens")}

def run(n, temp, seeded):
    res = {}; th = []
    for i in range(n):
        th.append(threading.Thread(target=stream, args=(PROMPTS[i % 3], (1000 + i) if seeded else None, temp, a.gen, res, i)))
    [t.start() for t in th]; [t.join() for t in th]
    return [res[i] for i in range(n)]

out = {}
stream("Say hi.", None, 0, 16, {}, 0)  # warm
if a.mode == "decode":
    for n in [int(x) for x in a.streams.split(",")]:
        aggs = []; per = []
        for rep in range(a.reps):
            r = run(n, 1.0, True)
            aggs.append(sum(x["rate"] for x in r)); per.append([round(x["rate"], 1) for x in r])
            print(f"streams {n} rep {rep}: aggregate {aggs[-1]:.1f} tok/s per-stream {per[-1]} tokens {[x['tokens'] for x in r]}", flush=True)
        out[f"streams_{n}"] = {"aggregate_median": round(statistics.median(aggs), 1), "aggregates": [round(x, 1) for x in aggs], "per_stream": per}
elif a.mode == "greedy":
    # the same greedy request: 1 alone, then 3 identical at once (code and chat prompts); aggregate tok/s, median of reps
    for name, pr in (("code", PROMPTS[0]), ("chat", PROMPTS[1])):
        for n in (1, 3):
            aggs = []
            for rep in range(a.reps):
                res = {}; th = [threading.Thread(target=stream, args=(pr, None, 0, 256, res, j)) for j in range(n)]
                [t.start() for t in th]; [t.join() for t in th]
                aggs.append(sum(res[j]["rate"] for j in range(n)))
            out[f"{name}_x{n}"] = {"aggregate_median": round(statistics.median(aggs), 1), "aggregates": [round(x, 1) for x in aggs]}
            print(f"greedy {name} x{n}: {out[f'{name}_x{n}']}", flush=True)
else:
    one = {}
    for i, p in enumerate(IDENT):
        r = run(1, 0, False) if False else None
        res = {}; stream(p, None, 0, 256, res, 0); x = res[0]
        one[i] = {"tokens": x["tokens"], "sha": hashlib.sha256(x["text"].encode()).hexdigest()[:16], "head": x["text"][:60]}
        print(f"ident {i}: {x['tokens']} tokens {one[i]['sha']}", flush=True)
    three = {}
    for k in range(0, 6, 3):
        res = {}; th = [threading.Thread(target=stream, args=(IDENT[k + j], None, 0, 256, res, j)) for j in range(3)]
        [t.start() for t in th]; [t.join() for t in th]
        for j in range(3):
            three[k + j] = {"tokens": res[j]["tokens"], "sha": hashlib.sha256(res[j]["text"].encode()).hexdigest()[:16]}
    out = {"alone": one, "x3": three}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out)[:2000])
