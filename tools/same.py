#!/usr/bin/env python3
"""same.py BASE MODEL OUT PROMPT_FILE...: for each prompt, one greedy 48-token reply with top-5 logprobs (thinking off),
or a 128-token reply's text alone where the server refuses logprobs (HTTP 400: Flash Next's native engine), and
one seeded sampled 128-token reply (temperature 1, top_p 0.95, top_k 20, seed 7: the sampled draws' bits);
writes {file: {"tokens": n, "sha": sha256 of the reply's text and logprobs, "text": first 80 chars}} to OUT. Two runs
whose shas match produced the same bits on these prompt lengths (the bench prompts are too short for the prompt-tile
paths). Stdlib only; run with python3 -I."""
import hashlib
import json
import sys
import urllib.error
import urllib.request

base, model, out, files = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
res = {}
logprobs = True


def ask(text, sampled=False):
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 48 if logprobs else 128,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    if sampled:
        body.update({"max_tokens": 128, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 7})
    elif logprobs:
        body.update({"logprobs": True, "top_logprobs": 5})
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


for f in files:
    try:
        try:
            r = ask(open(f).read())
        except urllib.error.HTTPError as e:
            if e.code != 400 or not logprobs:
                raise
            logprobs = False
            r = ask(open(f).read())
        c = r["choices"][0]
        key = json.dumps([c["message"]["content"], c.get("logprobs")], sort_keys=True)
        res[f] = {"tokens": r["usage"]["prompt_tokens"], "sha": hashlib.sha256(key.encode()).hexdigest()[:16],
                  "logprobs": c.get("logprobs") is not None, "text": c["message"]["content"][:80]}
        sr = ask(open(f).read(), sampled=True)["choices"][0]["message"]["content"]
        res[f]["sampled_sha"] = hashlib.sha256(sr.encode()).hexdigest()[:16]
    except urllib.error.HTTPError as e:
        res[f] = {"error": f"{e} {e.read()[:200]!r}"}
    except Exception as e:  # noqa: BLE001 - recorded, the comparison shows it
        res[f] = {"error": str(e)[:200]}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
