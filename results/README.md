# Raw results

All on one DGX Spark (GB10, 128 GB) with `nvidia/Qwen3.8-Flash-Next-FP4-Local` @ `b9ce18a6`, from the tools named in
the main README. Times are the measurement dates (2026).

Our builds: **build 23** is what [`patches/`](../patches/) make (0001-0011). **Build 22** is the same without 0011,
which changes only requests sent with `"draft": false`; drafted replies hash the same on both (`speed/same.json` here
and build 23's own run). The speed tables come from one run of build 22 (10-25 21:00-21:58); quality and the exactness
checks from build 23 (10-25 22:00-22:30).

| Folder | What | Build, when |
|---|---|---|
| `ours-bf16/speed` | one request (`bench_openai.out`), mixed load 1 / 2 / 4 (`mixed.out`), prompts 4K / 16K (`pptg.json`), reply hashes with the prompt cache off (`same.json`), accept probe (`accept.json`) | 22, 10-25 |
| `ours-bf16/identical-streams` | identical requests 1 / 2 / 4 and mixed 4 (`identical.out`, `mixed.out`) | 22, 10-25 |
| `ours-bf16/long-prompts` | prompts ~31K / 64K / 120K (`pptg.json`) | 22, 10-25 |
| `ours-bf16/quality` | task suite (`summary.txt`, `tasks.log`), identical requests with alone and `"draft": false` checks (`bench_concurrent.out`), accept probe (`accept.json`) | 23, 10-25 |
| `ours-bf16/exactness` | reply hashes with the prompt cache on, 4 resumed prompts (`cache-on.json`, build 22), and with `"draft": false` (`draft-false.json`, build 23); both equal `speed/same.json` | 22 / 23, 10-25 |
| `ours-fp8/...` | the same with `KV=fp8` (no `draft-false.json`) | 22 / 23, 10-25 |
| `ours-fp8-5x262k/` | `KV=fp8` at Mia's recipe's setup, 262,144 x 5: one request, mixed load 1 / 2 / 4 / 5, prompts (`speed/`), identical requests 1 / 2 / 4 / 5 (`identical-streams/`) | 22, 10-25 |
| `tf-1.1.4-prs/` | TensorFold 1.1.4 + #1143 / #1171 / #1164 with `TF_FLASHNEXT_DRAFT_HEAD=affine4 TF_FLASHNEXT_CHUNK=4096` (`speed/`), and its task suite and affine4 identical requests (`quality/`) | 10-24 |
| `mia-recipe/` | Mia's recipe at `656879f`, its own settings (262,144 x 5, FP8 cache): `quality/` (10-25), `mixed-load/` (10-24), `long-prompts/` (10-25), `prompts-2026-10-18/` (4K / 16K), `5x262k/` (one request, mixed load 1 / 2 / 4 / 5, identical requests 1 / 2 / 4 / 5; 10-25) | 10-18 to 10-25 |
| `lockstep/` | 4 identical requests together (`-0`) and 500 ms apart (`-500`), Mia's recipe and ours (build 18, before burst admission) | 10-25 |
