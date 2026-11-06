# Qwen3.8-Flash-Next on one DGX Spark (TensorFold)

Qwen3.8-Flash-Next (a mixture of experts with 176B parameters) served on **one** NVIDIA DGX Spark through
[TensorFold](https://github.com/ashhart/TensorFold)'s native engine. OpenAI-compatible API, thinking on, the
checkpoint's own MTP drafts, 3 streams of 262,144 tokens with the bf16 cache. It serves NVIDIA's public FP4-Local
checkpoint as published.

This repository is a recipe for **TensorFold 1.2.3**: a pinned release, the settings the numbers below were measured
with, a Dockerfile, and scripts that download the checkpoint, build the image and start the server. It used to carry
four patches of ours on TensorFold 1.1.5 (prompt and decode paths for the GB10, the loader, an MXFP8 language-model
head, an FP8 KV cache, admission of requests sent together). TensorFold has since taken in the pull requests of ours
that matter for this model, and the patches are gone: see [Upstream](#upstream).

## Numbers

Measured on one DGX Spark (GB10, 128 GB) on 2026-11-06 with the checkpoint below, thinking off, at 262,144-token
windows and 3 streams, with `TF_FLASHNEXT_DRAFT_HEAD=affine4 TF_FLASHNEXT_CHUNK=4096`. **Ours, now** is this
repository's build (TensorFold 1.2.3). **Ours, before** is the build this repository had until now (TensorFold 1.1.5
plus four patches of ours, commit `3bfc9fc`). Same machine, same tools, same day, a fresh server for every start,
nothing else running. Each figure is the median over every rep of 2 starts (now) or 3 starts (before); raw results in
[`results/`](results/).

| | Ours, before | Ours, now |
|---|---:|---:|
| Mixed load, 1 stream (tok/s) | **76.0** | 73.5 |
| Mixed load, 3 streams (tok/s, together) | 105.0 | 104.5 |
| Greedy code, 1 request (tok/s) | **76.8** | 75.5 |
| Greedy code, same request x3 (tok/s, together) | **190.1** | 183.4 |
| Greedy chat, 1 request (tok/s) | 47.6 | **51.0** |
| Greedy chat, same request x3 (tok/s, together) | 124.0 | **130.3** |
| Prompt, ~2,400 tokens (tok/s) | 1,982 | **2,038** |
| Prompt, ~7,000 tokens (tok/s) | 2,159 | **2,181** |
| Server ready after start (s) | 70, 70, 75 | **55, 55** |

**Mixed load**: `tools/bench.py decode`. Stream i takes the code, the explanation or the prose prompt, at temperature
1.0 with its own seed, 384 tokens, 3 reps; the aggregate is the sum of each stream's own rate. **Greedy**:
`tools/bench.py greedy`, 256 tokens, 3 reps. **Prompts**: `tools/pptg_bench.py`, a fresh text for every request so
that no cache helps, prompt tokens over the time to the first token (the 2,048 and 8,192-token targets came to about
2,400 and 7,000 tokens of this text). The earlier build's patches were written for 1.1.5; 1.2.3 itself has what
mattered of them. What is left of them (an MXFP8 language-model head, bf16 MoE rows, a fused multiply-add DeltaNet
chain) is worth 2 to 3 % on code decode (the rows above where "before" is ahead) and costs 5 to 7 % on greedy chat;
the rest is within the spread between starts.

On a 1.1.5 base the same measures were, before our patches, 69.3 tok/s at 1 stream and 96.0 at 3 (mixed load) and
1,731 tok/s on a 3,900-token prompt (the README of commit `3bfc9fc`).

Settings matter more than the patches did. TensorFold 1.2.3 with no engine settings, with only
`TF_FLASHNEXT_DRAFT_HEAD=affine4`, and with both settings of `config.sh` (`results/settings/`):

| | no settings | `affine4` only | both |
|---|---:|---:|---:|
| Mixed load, 1 / 3 streams (tok/s) | 67.0 / 97.4 | 73.7 / 103.9 | 74.0 / 104.7 |
| Prompts of ~2,400 / ~7,000 tokens (tok/s) | 1,637-1,780 / 1,835-1,867 | 1,757-1,794 / 1,868-1,869 | 1,990-2,045 / 2,016-2,143 |

Not re-measured on 1.2.3: the 130-item task suite, long prompts (30,000 to 250,000 tokens), the prompt cache and the
FP8 cache. `tools/accept_probe.py` and `tools/cache_bench.py` are still here; the earlier build's numbers for them
are in commit `3bfc9fc`.

**Memory**: the engine reports 73.8 GiB for the model and 7.91 GiB for each 262,144-token stream, with a 5.7 GiB
prompt cache; about 37 GiB of memory stayed available once loaded.

## Exactness

- Replies of this build equal unpatched TensorFold 1.2.3's: the same 8 greedy replies (thinking off, up to 256 tokens)
  match token for token between this build, the 1.2.3 test image, and TensorFold `main` at `c70d67bb` (6 November).
- Three identical-prompt requests at once give each the reply of its request alone, 6 of 6
  (`tools/bench.py ident`).
- Replies are **not** token for token those of the build before this one: 6 of the 8 differ. The cause is its four
  rounding changes (the MXFP8 prompt tile, bf16 MoE slot rows on prompts, fused multiply-adds in the DeltaNet chain,
  the MXFP8 language-model head), which 1.2.3 does not have. Check: the old build with those four switched off
  (`TF_FLASHNEXT_HEAD=bf16 TF_FLASHNEXT_MOE_B16=0 TF_FLASHNEXT_GDN_FMA=0 TF_FLASHNEXT_PROMPT_MX8=0`) gives 1.2.3's
  8 replies exactly (`results/settings/old-rounding-off/`). The 8 prompts are short, so they test decode, drafting and
  the loader, not long-prompt rounding.

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
TensorFold `v1.2.3` at the commit `config.sh` names, any [`patches/`](patches/) (none now), and `libtfgrammar`
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
| `EXTRA_ENV` | empty | more engine switches, space-separated `KEY=VALUE` (see TensorFold's [Flash Next on CUDA](https://github.com/ashhart/TensorFold/blob/v1.2.3/docs/recipes/flash-next-cuda.md)) |

The bf16 KV cache is the only one: TensorFold 1.2.3 has no FP8 cache for Flash Next (`KV=fp8` is refused).

## Good to know

- Text only in this build: images and video are untested.
- Thinking is on by default; send `"chat_template_kwargs": {"enable_thinking": false}` to turn it off per request.
- A bigger prompt chunk makes a decoding stream wait longer for its next token while a prompt is being filled (4,096
  rows is about 2 s on a GB10).
- Tested on a DGX Spark only.

## Upstream

TensorFold 1.2.3 carries the pull requests of ours that this model needs: the loader
([#1368](https://github.com/ashhart/TensorFold/pull/1368)), prompt passes and decode with several streams
([#1763](https://github.com/ashhart/TensorFold/pull/1763), [#1764](https://github.com/ashhart/TensorFold/pull/1764),
[#1590](https://github.com/ashhart/TensorFold/pull/1590)), drafted rounds
([#1786](https://github.com/ashhart/TensorFold/pull/1786), [#2160](https://github.com/ashhart/TensorFold/pull/2160)),
and @jschmied's FP4-Local pull requests from 1.1.5. [#1942](https://github.com/ashhart/TensorFold/pull/1942)
(one launch for a shared round's DeltaNet steps) is on `main` only; a `main` build of 6 November measured the same as
1.2.3 at 1 and 3 streams, so the recipe stays on the release. What did not go upstream: admission of requests sent
together (no gain left: identical requests decode as fast without it), the MXFP8 head, bf16 MoE rows and the DeltaNet
multiply-adds (their gain is gone and they change the rounding), and the FP8 KV cache (5 streams of 262,144 tokens;
it is in commit `3bfc9fc`).

## Credits

See [CREDITS.md](CREDITS.md). Licensed under the Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
