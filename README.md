# Qwen3.8-Flash-Next on one DGX Spark (TensorFold)

Qwen3.8-Flash-Next (a mixture of experts with 176B parameters) served on **one** NVIDIA DGX Spark through
[TensorFold](https://github.com/ashhart/TensorFold)'s native engine. OpenAI-compatible API, thinking on, the
checkpoint's own MTP drafts, 3 streams of 262,144 tokens with the bf16 cache. It serves NVIDIA's public FP4-Local
checkpoint as published.

This repository is a recipe for **TensorFold 1.2.10**: a pinned release, the settings the numbers below were measured
with, a Dockerfile, and scripts that download the checkpoint, build the image and start the server. It used to carry
four patches of ours on TensorFold 1.1.5 (prompt and decode paths for the GB10, the loader, an MXFP8 language-model
head, an FP8 KV cache, admission of requests sent together). TensorFold has since taken in the pull requests of ours
that matter for this model, and the patches are gone: see [Upstream](#upstream).

## Numbers

Measured on one DGX Spark (GB10, 128 GB) on 2026-11-19 with the checkpoint below, thinking off, at 262,144-token
windows and 3 streams, with `TF_FLASHNEXT_DRAFT_HEAD=affine4 TF_FLASHNEXT_CHUNK=4096`. **Ours, before** is this
repository's build until now (TensorFold 1.2.3, no patches), **Ours, now** is TensorFold 1.2.10, also without patches.
Same machine, same tools, same day, four fresh servers started in the order 1.2.3, 1.2.10, 1.2.10, 1.2.3, nothing else
running. Each figure is the median over every rep of the 2 starts of a build; raw results in [`results/`](results/).

| | Ours, before (1.2.3) | Ours, now (1.2.10) |
|---|---:|---:|
| Mixed load, 1 stream (tok/s) | 72.8 | 72.9 |
| Mixed load, 3 streams (tok/s, together) | 102.8 | **105.5** |
| Greedy code, 1 request (tok/s) | 74.3 | 74.4 |
| Greedy code, same request x3 (tok/s, together) | 180.6 | **185.6** |
| Greedy chat, 1 request (tok/s) | 50.2 | 50.4 |
| Greedy chat, same request x3 (tok/s, together) | 128.5 | **132.4** |
| Prompt, ~2,400 tokens (tok/s) | 1,916 | 2,079 |
| Prompt, ~7,000 tokens (tok/s) | 2,116 | 2,157 |
| Server ready after start (s) | 55, 55 | 55, 55 |

**Mixed load**: `tools/bench.py decode`. Stream i takes the code, the explanation or the prose prompt, at temperature
1.0 with its own seed, 384 tokens, 3 reps; the aggregate is the sum of each stream's own rate. **Greedy**:
`tools/bench.py greedy`, 256 tokens, 3 reps. **Prompts**: `tools/pptg_bench.py`, a fresh text for every request so
that no cache helps, prompt tokens over the time to the first token, 2 requests of each size in a start (the text is
TensorFold 1.2.3's `docs/*.md`; the 2,048 and 8,192-token targets came to about 2,400 and 7,000 tokens of it).

How to read them: one stream decodes the same on both (+0.1 to +0.4 %). Three streams at once are 2.6 to 3.0 % faster
on 1.2.10, and the two starts of each build agree (mixed 3 streams: 102.5 and 103.0 on 1.2.3, 106.0 and 105.4 on
1.2.10). The prompt figures are level or a little faster: between starts of the same build the 2,400-token figure
moves from 1,861 to 1,924 (1.2.3) and from 1,988 to 2,079 (1.2.10), so the ranges overlap, and at 7,000 tokens it is
inside the spread. The first request after any start is the slowest.

Settings matter more than the version. TensorFold 1.2.3 with no engine settings, with only
`TF_FLASHNEXT_DRAFT_HEAD=affine4`, and with both settings of `config.sh` (measured on 1.2.3 on 2026-11-06,
`results/settings/`; not repeated on 1.2.10):

| | no settings | `affine4` only | both |
|---|---:|---:|---:|
| Mixed load, 1 / 3 streams (tok/s) | 67.0 / 97.4 | 73.7 / 103.9 | 74.0 / 104.7 |
| Prompts of ~2,400 / ~7,000 tokens (tok/s) | 1,637-1,780 / 1,835-1,867 | 1,757-1,794 / 1,868-1,869 | 1,990-2,045 / 2,016-2,143 |

Not re-measured on 1.2.10: the 130-item task suite, long prompts (30,000 to 250,000 tokens), the prompt cache and the
FP8 cache. `tools/accept_probe.py` and `tools/cache_bench.py` are still here; the numbers of the build of commit
`3bfc9fc` (TensorFold 1.1.5 plus four patches of ours) are in that commit's README.

**Memory**: the engine reports 73.8 GiB for the model and 7.91 GiB for each 262,144-token stream, with a prompt cache of
5.1 to 5.9 GiB (it takes what is free at the start); 40.8 to 41.5 GiB of memory stayed available once loaded and 37.5
to 38.2 GiB after the benchmarks.

## Exactness

- Replies of this build equal TensorFold 1.2.3's: the same 8 greedy replies (thinking off, up to 256 tokens) have the same
  token hashes in all four starts of 2026-11-19 (two of 1.2.3, two of 1.2.10). The 1.2.3 replies themselves equal the
  unpatched 1.2.3 test image and TensorFold `main` at `c70d67bb` (6 November).
- Three identical-prompt requests at once give each the reply of its request alone, 6 of 6 in every start
  (`tools/bench.py ident`).
- Replies are **not** token for token those of the build of commit `3bfc9fc` (TensorFold 1.1.5 plus four patches): 6 of
  the 8 differ. The cause is its four rounding changes (the MXFP8 prompt tile, bf16 MoE slot rows on prompts, fused
  multiply-adds in the DeltaNet chain, the MXFP8 language-model head), which TensorFold 1.2.3 and 1.2.10 do not have.
  Check: that build with those four switched off (`TF_FLASHNEXT_HEAD=bf16 TF_FLASHNEXT_MOE_B16=0 TF_FLASHNEXT_GDN_FMA=0
  TF_FLASHNEXT_PROMPT_MX8=0`) gives 1.2.3's 8 replies exactly (`results/settings/old-rounding-off/`). The 8 prompts are
  short, so they test decode, drafting and the loader, not long-prompt rounding.

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
./start.sh            # ~1 min until ready; http://127.0.0.1:8888/v1, model "qwen3.8-flash-next"
./stop.sh
```

```sh
curl http://127.0.0.1:8888/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## The Docker image

`prepare.sh` builds it from [`Dockerfile`](Dockerfile): NVIDIA's CUDA 13.3 image, Zig 0.17.0 (checked by its SHA-256),
TensorFold `v1.2.10` at the commit `config.sh` names, any [`patches/`](patches/) (none now), and `libtfgrammar`
(structured output and required tool calls) from xgrammar `v0.2.8`. The image holds the server and that library; the
checkpoint stays on your disk, mounted read-only. It is tagged with the TensorFold version, and its `tf.build` label
is a hash of the Dockerfile, the pins and the patches: `prepare.sh` rebuilds when it changes.

## Settings

Everything is in [`config.sh`](config.sh); override a value with an environment variable or a `.env` file:

| Variable | Default | Meaning |
|---|---|---|
| `MODELS_DIR` / `MODEL_DIR` | `~/models` / `$MODELS_DIR/Qwen3.8-Flash-Next-FP4-Local` | where the checkpoint goes |
| `BIND`, `PORT` | `127.0.0.1`, `8888` | where the API listens; `BIND=0.0.0.0` for your network |
| `API_KEY` | empty | require `Authorization: Bearer <key>` |
| `CONTEXT`, `PARALLEL` | 262,144 x 3 | the window (prompt plus reply) and the streams served at once; 3 is the most that fits next to the weights |
| `SERVED_MODEL_NAME` | `qwen3.8-flash-next` | the model name clients send |
| `ENGINE_ENV` | `TF_FLASHNEXT_DRAFT_HEAD=affine4 TF_FLASHNEXT_CHUNK=4096` | the engine settings every number above was measured with: the draft head in 4 bits (about 10 % faster decode) and 4,096-row prompt chunks (about 13 % faster prompts) |
| `EXTRA_ENV` | empty | more engine switches, space-separated `KEY=VALUE` (see TensorFold's [Flash Next on CUDA](https://github.com/ashhart/TensorFold/blob/v1.2.10/docs/recipes/flash-next-cuda.md)) |

The bf16 KV cache is the only one this recipe serves: TensorFold 1.2.3 had no FP8 cache for Flash Next, and none was tested on 1.2.10 (`KV=fp8` is refused).

## Good to know

- Text only in this build: images and video are untested.
- Thinking is on by default; send `"chat_template_kwargs": {"enable_thinking": false}` to turn it off per request.
- A bigger prompt chunk makes a decoding stream wait longer for its next token while a prompt is being filled (4,096
  rows is about 2 s on a GB10).
- Tested on a DGX Spark only.

## Upstream

TensorFold 1.2.10 carries the pull requests of ours that this model needs: the loader
([#1368](https://github.com/ashhart/TensorFold/pull/1368)), prompt passes and decode with several streams
([#1763](https://github.com/ashhart/TensorFold/pull/1763), [#1764](https://github.com/ashhart/TensorFold/pull/1764),
[#1590](https://github.com/ashhart/TensorFold/pull/1590)), drafted rounds
([#1786](https://github.com/ashhart/TensorFold/pull/1786), [#2160](https://github.com/ashhart/TensorFold/pull/2160)),
and @jschmied's FP4-Local pull requests from 1.1.5. Three more of ours arrived after 1.2.3, in 1.2.4:
[#1866](https://github.com/ashhart/TensorFold/pull/1866) (a prompt-cache mark inside a prompt chunk is kept on the way
through the pass),
[#1942](https://github.com/ashhart/TensorFold/pull/1942) (one launch for a shared round's DeltaNet steps; the
3-stream gain in the table above fits it, 1.2.4 to 1.2.10 was measured as a whole) and
[#2244](https://github.com/ashhart/TensorFold/pull/2244) (the draft head reads its own pages with `TF_FLASHNEXT_KV_PAGES`;
default settings unchanged). What did not go upstream: admission of requests sent together (no gain left: identical
requests decode as fast without it), the MXFP8 head, bf16 MoE rows and the DeltaNet multiply-adds (their gain is gone
and they change the rounding), and the FP8 KV cache (5 streams of 262,144 tokens; it is in commit `3bfc9fc`).

## Credits

See [CREDITS.md](CREDITS.md). Licensed under the Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
