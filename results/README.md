# Raw results

All on one DGX Spark (GB10, 128 GB) with `nvidia/Qwen3.8-Flash-Next-FP4-Local` at `b9ce18a6`, on 2026-11-06, with
`tools/bench.py` (`decode`, `greedy`, `ident`) and `tools/pptg_bench.py` (`--sizes 2048,8192,2048,8192 --gen 16`, the
text of TensorFold's documentation). Every server ran at 3 streams x 262,144 tokens with the bf16 cache and
`TF_FLASHNEXT_DRAFT_HEAD=affine4 TF_FLASHNEXT_CHUNK=4096`, a fresh start for each folder, with nothing else running.
`start.txt` is the time `start.sh` waited until the server answered.

| Folder | What |
|---|---|
| `old-1.1.5-ours/` | the earlier build of this repository (commit `3bfc9fc`: TensorFold 1.1.5 plus four patches of ours), three starts: `before`, `old-g`, `old-3` (`before` has no `greedy.json`) |
| `tf-1.2.3/` | this repository's build (TensorFold 1.2.3, no patches): `new-final-1` and `new-final-2` started through the box's menu; `stock123` and `stock123-g` are the same engine from a test image built before the recipe was changed (identical build) |
| `tf-1.2.3-main/` | TensorFold `main` at `c70d67bb` (6 November, with #1942), unpatched, two starts |
| `settings/` | TensorFold 1.2.3 with no engine settings (`stock123-noenv`) and with only the 4-bit draft head (`stock123-affine4only`); `old-rounding-off` is the earlier build with `TF_FLASHNEXT_HEAD=bf16 TF_FLASHNEXT_MOE_B16=0 TF_FLASHNEXT_GDN_FMA=0 TF_FLASHNEXT_PROMPT_MX8=0`, whose `ident.json` equals 1.2.3's |

Files: `decode.json` (mixed load, 1 and 3 streams, 3 reps each), `greedy.json` (the same greedy request once and 3
at once, code and chat), `ident.json` (sha of the greedy reply to 8 prompts alone, and to 6 of them 3 at a time),
`pptg.json` (prompt tokens, time to the first token, prompt tok/s). A first rep after a start can be slow: the
`old-g` start's first one-stream rep read 56.9 tok/s. The README's numbers are the median of every rep over the
starts of a build.
