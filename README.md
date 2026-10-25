# Qwen3.8-Flash-Next on one DGX Spark (TensorFold)

Qwen3.8-Flash-Next (a mixture of experts with 176B parameters) served on **one** NVIDIA DGX Spark through
[TensorFold](https://github.com/ashhart/TensorFold)'s native engine, tuned for the GB10. OpenAI-compatible API,
thinking on, the checkpoint's own MTP drafts, 4 streams of 65,536 tokens (or 5 of 262,144 with the FP8 cache).

This repository is TensorFold 1.1.4 plus patches: three open FP4-Local pull requests by @jschmied, and our own work
for the GB10 (prompt and decode paths, the loader, an MXFP8 language-model head, an optional FP8 KV cache, and
admission of requests that arrive together). It serves NVIDIA's public FP4-Local checkpoint as published.

## Numbers

Measured on one DGX Spark (GB10, 128 GB), checkpoint below, thinking off. "Ours" is this repository's build
(bf16 cache = the default, FP8 = `KV=fp8`). The other two columns were measured by us on the same machine with
the same tools: **Stock TF 1.1.4** is TensorFold 1.1.4 before our patches (with the same three PRs and the same two
settings, `TF_FLASHNEXT_DRAFT_HEAD=affine4` and `TF_FLASHNEXT_CHUNK=4096`), and [Mia's TensorFold recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)
at `656879f` with its own settings (262,144 x 5, FP8 cache).

A greedy request measures one reply. When a change rounds differently, the reply's text changes, and so does how
well that text drafts: between our builds, the one-request chat number below moved between 49.7 and 57.9 tok/s at
the same time per round. The first two tables average many texts; compare those first.

**Mixed load (tok/s, all streams together)** — `tools/bench_concurrent.py --mixed`: request i takes the code or the
chat prompt, temperature 1.0, its own seed, 256 tokens, 3 reps; at 2 and 4 streams, the rate while every stream is
generating. The closest of these to real use.

| Streams | Ours, bf16 | Ours, FP8 | Stock TF 1.1.4 | Mia's recipe |
|---:|---:|---:|---:|---:|
| 1 | **75.1** | 73.6 | 69.5 | 65.0 |
| 2 | **91.4** | 87.2 | 80.2 | 81.4 |
| 4 | **123.0** | 115.6 | 101.7 | 114.6 |

**By kind of text (tok/s, and tokens a round)** — `tools/accept_probe.py`: 3 code, 3 explanation, 2 prose and
1 German prompt, 384 tokens each, greedy, one stream

| | Ours, bf16 | Ours, FP8 | Mia's recipe |
|---|---:|---:|---:|
| Code | **74.7** (5.33) | 73.2 (5.14) | 58.5 (3.92) |
| Explanations | **53.6** (3.16) | 52.8 (3.00) | 49.5 (3.07) |
| Prose | 41.2 (2.22) | **42.9** (2.26) | 36.2 (2.08) |
| German | 45.8 (2.42) | **46.3** (2.51) | 38.4 (2.23) |

**One reply each (tok/s)** — greedy, 256 tokens: one request (`tools/bench_openai.py`, median of 3), and N identical
requests sent at once (`tools/bench_concurrent.py`, all streams together ¹)

| | Ours, bf16 | Ours, FP8 | Stock TF 1.1.4 | Mia's recipe |
|---|---:|---:|---:|---:|
| Code (`fibonacci-raw`), 1 request | **81.5** | 75.9 | 73.2 | 67.8 |
| Chat (`gpu-chat-no-think`), 1 request | 49.7 | 46.9 | 45.3 | **50.9** |
| Same code request x2 / x4 | **144.3 / 234.6** | 132.7 / 215.6 | 111.4 / 157.8 | 123.7 / 228.6 |
| Same chat request x2 / x4 | 90.4 / 150.4 | 84.2 / 139.8 | 70.7 / 103.5 | **95.5 / 167.3** |

¹ Both engines admit requests that arrive together as one group, so the streams run in lockstep and share their
expert reads; sent 500 ms apart, Mia's recipe did 127.9 / 103.6 and an earlier build of ours 143.9 / 108.6
(code / chat x4, [`results/lockstep/`](results/lockstep/)).

**Prompts (tok/s)** — `tools/pptg_bench.py`, a fresh text per size so no cache helps

| Prompt tokens | Ours, bf16 | Ours, FP8 | Stock TF 1.1.4 | Mia's recipe |
|---:|---:|---:|---:|---:|
| ~3,900 | 2,282 | **2,327** | 1,593 | 2,140 |
| ~15,300 | 2,292 | **2,331** | 1,623 | 2,166 |
| ~31,400 | 2,255 | **2,316** | — | 2,137 |
| ~64,000 | 2,206 | **2,264** | — | 2,148 |
| ~120,000 | 2,065 | **2,171** | — | 2,021 (132K) |

**At Mia's recipe's own setup: FP8 cache, 5 streams of 262,144 tokens** (`KV=fp8`, the same tools)

| | Ours, FP8 | Mia's recipe |
|---|---:|---:|
| Mixed load, 1 / 2 / 4 / 5 streams | **73.2 / 86.3** / 114.9 / 128.6 | 65.3 / 80.7 / **115.9 / 134.0** |
| Code, 1 request | **75.4** | 68.0 |
| Chat, 1 request | 46.5 | **51.1** |
| Same code request x2 / x4 / x5 | **133.1** / 216.4 / 247.1 | 125.5 / **228.4 / 272.0** |
| Same chat request x2 / x4 / x5 | 84.2 / 139.8 / 160.5 | **95.8 / 169.0 / 201.8** |
| Prompts, ~3,900 / ~15,300 tokens | **2,307 / 2,328** | 2,140 / 2,166 |

At four and five streams Mia's recipe is ahead here. It runs the DeltaNet layers' decode for all streams in one
launch a layer; ours runs one a stream (in a 5-stream profile, five times the launches and 0.35 s more kernel time).
That is the next piece to port.

**Quality** — a 130-item task suite (math 50, JSON 15, instructions 10, tool calls 20, agent 10, code 25 in a
sandbox): ours 123 (bf16) and 124 (FP8); stock TF 1.1.4 122; Mia's recipe 121.

**Load and memory.** Ready in 70-80 s from a warm disk (Mia's recipe: 80 s; stock TF 1.1.4: 140-170 s). With the default
4 x 65,536 tokens about 37 GB stay free while serving.

Raw results: [`results/`](results/) (ours in `ours-bf16/`, `ours-fp8/` and `ours-fp8-5x262k/`, stock TensorFold 1.1.4
+ PRs in `tf-1.1.4-prs/`, Mia's recipe in `mia-recipe/`). A dash: not measured with these settings.

Dates and provenance of every file: [`results/README.md`](results/README.md) (TF 1.1.4 + PRs on 2026-10-24, ours on
10-25, Mia's recipe 10-18 to 10-25 at its own settings). Our speed tables come from one run of the build before the
last patch (0011 touches only `"draft": false` requests; drafted replies hash the same); quality and the exactness
checks are from the build these patches make.

## Exactness

- Concurrent requests give the same replies as the same request alone (28 of 28 checks), and drafted replies equal
  `"draft": false`, on the short bench prompts and on the 300- to 15,000-token prompts below
  (`results/ours-*/quality/bench_concurrent.out`, `results/ours-bf16/exactness/draft-false.json`).
- A prompt gives the same reply whether the prompt cache serves part of it or not: reply hashes with the cache on
  (resumed prompts) equal those with it off, bf16 and FP8 (`results/ours-*/exactness/cache-on.json` against
  `speed/same.json`).
- Every change was measured against the previous build on the same machine. Changes meant to be exact were checked
  with reply hashes (`tools/same.py`: greedy and seeded-sampled replies on 300 to 15,000-token prompts, prompts in
  `tools/prompts/`).
- Replies are **not** bit for bit those of stock TensorFold 1.1.4. Four changes round differently: the MXFP8 prompt
  tile, bf16 MoE slot rows on prompts, fused multiply-adds in the DeltaNet prompt chain, and the language-model head
  rounded to MXFP8 at load. The task suite above is the check. The last three can be turned off (Settings).
- The FP8 cache stores keys and values as 8-bit numbers with one power-of-two scale per position and head: lossy by
  design (replies differ from the bf16 cache), with the same exactness between requests.

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
SHA-256), TensorFold `v1.1.4`, [`patches/`](patches/), and `libtfgrammar` (structured output and required tool
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
| `CONTEXT`, `PARALLEL` | 65,536 x 4 (`fp8`: 262,144 x 5) | the window (prompt plus reply) and the streams served at once |
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

- **Engine.** TensorFold's Zig engine on one GPU. [`patches/`](patches/) applies to TensorFold `v1.1.4`
  ([`patches/BASE`](patches/BASE) names the commit): 0001-0007 are @jschmied's FP4-Local pull requests #1143, #1171
  and #1164 (NVFP4 experts on prompt tiles, decode half tiles, hyper-connection kernels), rebased onto 1.1.4;
  0008 admits requests sent together as one group; 0009 is our GB10 tuning (each change named in its message);
  0010 makes a prompt's rounding independent of where its fill is split (the prompt cache, chunks); 0011 fixes the
  first token after a long prompt with `"draft": false` (its last row read stale MoE rows).
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

The three PRs in 0001-0007 are open on TensorFold; our changes are not offered upstream yet. Until they are in a
release, this repository is the way to run them.

## Credits

See [CREDITS.md](CREDITS.md). Licensed under the Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
