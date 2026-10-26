# Qwen3.8-Flash-Next on one DGX Spark (TensorFold)

Qwen3.8-Flash-Next (a mixture of experts with 176B parameters) served on **one** NVIDIA DGX Spark through
[TensorFold](https://github.com/ashhart/TensorFold)'s native engine, tuned for the GB10. OpenAI-compatible API,
thinking on, the checkpoint's own MTP drafts, 3 streams of 262,144 tokens (5 with the FP8 cache).

This repository is TensorFold 1.1.5 plus four patches of our own for the GB10: prompt and decode paths, the loader,
an MXFP8 language-model head, an optional FP8 KV cache, admission of requests that arrive together, and two fixes.
1.1.5 already carries @jschmied's FP4-Local pull requests. It serves NVIDIA's public FP4-Local checkpoint as
published.

## Numbers

Measured on one DGX Spark (GB10, 128 GB) with the checkpoint below, thinking off, every server at 262,144-token
windows: **ours with the bf16 cache** (the default, 3 streams), **ours with the FP8 cache** (`KV=fp8`, 5 streams), and
**stock TensorFold 1.1.5**, the release these patches apply to, at the same 3 streams with the same two settings
(`TF_FLASHNEXT_DRAFT_HEAD=affine4` and `TF_FLASHNEXT_CHUNK=4096`). Same machine, same tools, same day.

**Mixed load (tok/s, all streams together)**: `tools/bench_concurrent.py --mixed`. Request i takes the code or the
chat prompt, at temperature 1.0 with its own seed, 256 tokens, 3 reps. From 2 streams up, the rate while every stream
is generating. This is the closest of these measures to real use.

| Streams | Stock TF 1.1.5 | Ours, bf16 | Ours, FP8 |
|---:|---:|---:|---:|
| 1 | 69.3 | **75.3** | 73.3 |
| 2 | 80.2 | **91.4** | 86.4 |
| 3 | 96.0 | **116.4** | – |
| 4 | – | – | **115.1** |
| 5 | – | – | **129.1** |

**By kind of text (tok/s, and tokens a round)**: `tools/accept_probe.py`, with 3 code, 3 explanation, 2 prose and 1
German prompt, 384 tokens each, greedy, one stream.

| | Stock TF 1.1.5 | Ours, bf16 | Ours, FP8 |
|---|---:|---:|---:|
| Code | 65.3 (4.68) | **74.9** (5.33) | 72.7 (5.14) |
| Explanations | 48.8 (3.14) | **53.6** (3.16) | 52.5 (3.00) |
| Prose | 36.2 (2.12) | 41.3 (2.22) | **43.1** (2.26) |
| German | 39.5 (2.34) | **46.1** (2.42) | **46.1** (2.51) |

**One reply each (tok/s)**: greedy, 256 tokens. One request uses `tools/bench_openai.py` (median of 3); N identical
requests sent at once use `tools/bench_concurrent.py` (all streams together).

| | Stock TF 1.1.5 | Ours, bf16 | Ours, FP8 |
|---|---:|---:|---:|
| Code (`fibonacci-raw`), 1 request | 73.6 | **81.7** | 75.9 |
| Chat (`gpu-chat-no-think`), 1 request | 45.5 | **49.8** | 46.8 |
| Same code request x2 / x3 | 129.5 / 179.4 | **143.2 / 194.3** | 132.0 / – |
| Same chat request x2 / x3 | 81.1 / 113.7 | **90.0 / 122.8** | 84.0 / – |
| Same code / chat request x5 | – | – | **246.1 / 159.2** |

A greedy request measures one reply. When a change rounds differently, the reply's text changes, and so does how
well that text drafts. The two tables above average many texts.

**Prompts (tok/s)**: `tools/pptg_bench.py`, with a fresh text per size so that no cache helps. The 4K and 16K prompts
run at the servers above; the longer ones run one stream at a 262,144-token window.

| Prompt tokens | Stock TF 1.1.5 | Ours, bf16 | Ours, FP8 |
|---:|---:|---:|---:|
| ~3,900 | 1,731 | 2,269 | **2,307** |
| ~15,300 | 1,832 | 2,266 | **2,316** |
| ~31,400 | 1,781 | 2,260 | **2,303** |
| ~64,000 | 1,834 | 2,210 | **2,250** |
| ~146,000 | 1,767 | 2,069 | **2,117** |
| ~246,000 | 1,700 | 1,858 | **1,967** |

**The prompt cache (time to the first token)**: `tools/cache_bench.py`. One document is sent three times: with a
question, with a different question (a follow-up), and with the first question again. The prompt cache is left at the
engine's own size, as this recipe runs it.

| Document | Stock TF 1.1.5 | Ours, bf16 | Ours, FP8 |
|---|---|---|---|
| ~34,000 tokens: first / follow-up / repeat | 19.3 / 3.4 / 0.1 s | 15.5 / 3.3 / 0.2 s | 15.2 / 2.9 / 2.9 s |
| ~141,000 tokens: first / follow-up | 81.0 / 78.3 s | 72.6 / **4.3 s** | 67.1 / 67.3 s |
| ~247,000 tokens: first | 146.4 s | 136.1 s | **126.8 s** |

The follow-up resumes from the document's kept state when the cache has room for it. At 141,000 tokens only the bf16
cache had room. At 247,000 tokens none of the three did, so each request processed the whole document again.

**Quality**: a 130-item task suite (math 50, JSON 15, instructions 10, tool calls 20, agent 10, and code 25 in a
sandbox). Ours scores 123 (bf16) and 124 (FP8); stock TF 1.1.5 scores 122.

**Load and memory**: ours is ready 75 s after start from a warm disk; stock TF 1.1.5 takes 170 s. While the task suite
ran, at least 37 GB stayed free.

Raw results: [`results/`](results/) (ours in `ours-bf16/` and `ours-fp8/`, stock TensorFold 1.1.5 in `tf-1.1.5/`),
with dates and provenance in [`results/README.md`](results/README.md).

## Exactness

- Each concurrent request gives the same reply as the same request alone, in 28 of 28 checks
  (`bench_concurrent.py --alone --serial`).
- Drafted replies equal `"draft": false`, on the bench prompts and on 300- to 15,000-token prompts
  (`results/ours-*/exactness/draft-false.json`).
- A prompt gives the same reply whether or not the prompt cache serves part of it: reply hashes with the cache on
  (resumed prompts) equal those with it off, for both bf16 and FP8 (`results/ours-*/exactness/`).
- Every change was measured against the previous build on the same machine. Changes meant to be exact were checked
  with reply hashes: `tools/same.py`, greedy and seeded-sampled replies on 300- to 15,000-token prompts (prompts in
  `tools/prompts/`).
- Replies are **not** bit for bit those of stock TensorFold 1.1.5, because four changes round differently:
  - the MXFP8 prompt tile
  - bf16 MoE slot rows on prompts
  - fused multiply-adds in the DeltaNet prompt chain
  - the language-model head, rounded to MXFP8 at load

  The task suite above is the check. The last three can be turned off (see Settings).
- The FP8 cache stores keys and values as 8-bit numbers, with one power-of-two scale per position and head. It is
  lossy by design, so its replies differ from the bf16 cache's, and it keeps the same exactness between requests.

## Requirements

- An NVIDIA DGX Spark (GB10, 128 GB unified memory). Other GPUs are untested.
- Docker with the NVIDIA Container Toolkit (DGX OS has both).
- About 100 GB of free disk for the checkpoint, on the internal NVMe: the model reads its n-gram table from these
  files while it serves. About 108 GB of free memory to start: stop other models first.

## Quick start

```sh
git clone https://github.com/Arminova/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
cd Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold
./prepare.sh          # downloads the checkpoint (~96 GiB, resumable) and builds the image (~15 min)
./start.sh            # ~75 s until ready; http://127.0.0.1:8888/v1, model "qwen3.8-flash-next"
./stop.sh
```

```sh
curl http://127.0.0.1:8888/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Long windows: `KV=fp8 ./start.sh` serves 5 streams of 262,144 tokens.

## The Docker image

`prepare.sh` builds it from [`Dockerfile`](Dockerfile): NVIDIA's CUDA 13.3 image, Zig 0.17.0 (checked by its
SHA-256), TensorFold `v1.1.5`, [`patches/`](patches/), and `libtfgrammar` (structured output and required tool
calls) from xgrammar `v0.2.8`. The image holds the server and that library; the checkpoint stays on your disk,
mounted read-only. Its `tf.patches` label names the patches it was built from, and `prepare.sh` rebuilds when they
change.

## Settings

Everything is in [`config.sh`](config.sh); override a value with an environment variable or a `.env` file:

| Variable | Default | Meaning |
|---|---|---|
| `MODELS_DIR` / `MODEL_DIR` | `~/models` / `$MODELS_DIR/Qwen3.8-Flash-Next-FP4-Local` | where the checkpoint goes |
| `BIND`, `PORT` | `127.0.0.1`, `8888` | where the API listens; `BIND=0.0.0.0` for your network |
| `API_KEY` | empty | require `Authorization: Bearer <key>` |
| `KV` | `bf16` | the KV cache: `bf16`, or `fp8` for long windows |
| `CONTEXT`, `PARALLEL` | 262,144 x 3 (`fp8`: 262,144 x 5) | the window (prompt plus reply) and the streams served at once |
| `SERVED_MODEL_NAME` | `qwen3.8-flash-next` | the model name clients send |
| `EXTRA_ENV` | empty | engine switches, space-separated `KEY=VALUE` (below) |

Engine switches for `EXTRA_ENV` (all on by default; each restores TensorFold's behaviour):

| Switch | Effect |
|---|---|
| `TF_FLASHNEXT_HEAD=bf16` | keep the checkpoint's bf16 language-model head (about 5 % slower decode) |
| `TF_FLASHNEXT_MOE_B16=0` | fp32 MoE slot rows on prompts (about 4 % slower prompts) |
| `TF_FLASHNEXT_GDN_FMA=0` | separate products and sums in the DeltaNet prompt chain |
| `TENSORFOLD_ADMIT_WAIT_MS=0` | no 5 ms admission window for requests sent together |
| `TF_FLASHNEXT_KV_HEAD=fp8` | with `KV=fp8`: the draft head's own cache in FP8 too (fewer accepted drafts) |

## How it works

- **Engine.** TensorFold's Zig engine on one GPU. [`patches/`](patches/) applies to TensorFold `v1.1.5`
  ([`patches/BASE`](patches/BASE) names the commit): 0001 admits requests sent together as one group; 0002 is our
  GB10 tuning (each change named in its message); 0003 makes a prompt's rounding independent of where its fill is
  split (the prompt cache, chunks); 0004 fixes the first token after a long prompt with `"draft": false` (its last
  row read stale MoE rows). The FP8 cache stays off with 1.1.5's KV paging (`TF_FLASHNEXT_KV_PAGES`): the two are
  refused together.
- **Prompts.** 4,096-row chunks, MXFP8 and bf16 prompt tiles, fused hyper-connection kernels, bf16 MoE slot rows,
  and fused multiply-adds in the DeltaNet chain.
- **Decode.** CUDA graphs for verify windows and draft steps, sampled streams drafted together, GPU top-k draws,
  and the language-model head in MXFP8 (half the bytes every verify reads).
- **Loading.** Experts packed on 8 threads straight from the mapped files, one upload a layer; loaded files leave
  the page cache so the n-gram table can stay in memory.
- **FP8 cache.** Keys and values as e4m3 rows with one power-of-two scale per position and head (the format of
  Mia's engine). The attention kernels widen them into their bf16 tiles, so the arithmetic after the load is
  unchanged. The draft head keeps a bf16 cache.

## Good to know

- Text only in this build: images and video are untested with our patches.
- Thinking is on by default; send `"chat_template_kwargs": {"enable_thinking": false}` to turn it off per request.
- Tested on a DGX Spark only.

## Upstream

@jschmied's FP4-Local pull requests are in TensorFold 1.1.5. Our own changes are not in TensorFold yet; until a
release carries them, this repository is the way to run them.

## Credits

See [CREDITS.md](CREDITS.md). Licensed under the Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
