# Raw results

All on one DGX Spark (GB10, 128 GB) with `nvidia/Qwen3.8-Flash-Next-FP4-Local` at `b9ce18a6`, from the tools named in
the main README, on 2026-10-25 and 10-26. Every server ran at 262,144-token windows.

| Folder | What |
|---|---|
| `ours-bf16/` | this repository's build with the bf16 cache at 3 streams: one request (`speed/bench_openai.out`), mixed load 1 / 2 / 3 streams (`speed/mixed.out`), prompts ~4K / 16K (`speed/pptg.json`), reply hashes with the prompt cache off (`speed/same.json`), the accept probe (`by-kind/accept.json`), identical requests x2 / x3 (`identical-requests/`), prompts ~31K to ~246K at one stream (`long-prompts/`), the task suite with alone and `"draft": false` checks (`quality/`), the prompt cache (`prompt-cache/cache.json`), reply hashes with the prompt cache on (`exactness/cache-on.json`) and with `"draft": false` (`exactness/draft-false.json`), both equal to `speed/same.json` |
| `ours-fp8/` | the same with `KV=fp8` at 5 streams (mixed load 1 / 2 / 4 / 5, identical requests x2 / x4 / x5) |
| `tf-1.1.5/` | stock TensorFold 1.1.5 at 3 streams, with `TF_FLASHNEXT_DRAFT_HEAD=affine4` and `TF_FLASHNEXT_CHUNK=4096`, measured with the same tools (no `exactness/`) |

The bf16 cache serves at most 3 streams at this window: at 4 the engine refuses to start ("needs 31.44 GiB for its
streams at a 262144-token window, and the CUDA memory budget leaves 28.67 GiB").
