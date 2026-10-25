# Raw results

All on one DGX Spark (GB10, 128 GB) with `nvidia/Qwen3.8-Flash-Next-FP4-Local` @ `b9ce18a6`, from the tools named in
the main README. Times are the measurement dates (2026).

| Folder | What | When |
|---|---|---|
| `ours-bf16/speed` | this build, default bf16 cache: one request (`bench_openai.out`), mixed load (`mixed.out`), prompts 4K / 16K (`pptg.json`), reply hashes (`same.json`) | 10-25 |
| `ours-bf16/quality` | task suite (`summary.txt`, `tasks.log`), identical requests with alone / serial checks (`bench_concurrent.out`), accept probe (`accept.json`) | 10-25 |
| `ours-bf16/identical-streams`, `long-prompts` | identical requests 1 / 2 / 4 and mixed 4 (`identical.out`, `mixed.out`); prompts ~31K / 64K / 120K (`pptg.json`) | 10-25 |
| `ours-fp8/...` | the same with `KV=fp8`; `quality` is from the build before (10-25 16:00), whose FP8 replies hash the same as this build's (`speed/same.json`) | 10-25 |
| `tf-1.1.4-prs/` | TensorFold 1.1.4 + #1143 / #1171 / #1164 with `TF_FLASHNEXT_DRAFT_HEAD=affine4 TF_FLASHNEXT_CHUNK=4096` (`speed/`), and its task suite and affine4 identical requests (`quality/`) | 10-24 |
| `mia-recipe/` | Mia's recipe at `656879f`, its own settings (262,144 x 5, FP8 cache): `quality/` (10-25), `mixed-load/` (10-24), `long-prompts/` (10-25), `prompts-2026-10-18/` (4K / 16K) | 10-18 to 10-25 |
| `lockstep/` | 4 identical requests together (`-0`) and 500 ms apart (`-500`), Mia's recipe and ours | 10-25 |
