# Changelog

## 0.3.0 (unreleased)

- TensorFold 1.2.3 (`6ec5fa4c`), no patches. It carries our pull requests #1368, #1590, #1763, #1764, #1786 and #2160;
  #1942 is on `main` only, and a `main` build of 6 November measured no different at 3 streams. Against the 0.2.0 build
  (1.1.5 plus four patches) the numbers are within a few percent either way, the server is ready in 55 s instead of
  70 s, and with the old build's four rounding switches off its replies equal 1.2.3's token for token.
- Dropped: patch 0001 (admission of requests sent together: identical requests now decode as fast without it), 0002
  (GB10 tuning: its prompt and decode changes are in 1.2.3, its rounding changes no longer pay), 0003 and 0004 (they
  only serve 0002's rounding changes). The FP8 KV cache (`KV=fp8`, 5 streams) went with 0002; 1.2.3 has none for
  Flash Next, and `start.sh` refuses `KV=fp8`. The 0.2.0 build stays in commit `3bfc9fc`.
- The image is tagged with the TensorFold version and labelled `tf.build` (a hash of the Dockerfile, the pins and any
  `patches/*.patch`); `patches/` may be empty.
- `tools/bench.py`: mixed decode, greedy requests and reply identity.

## 0.2.0 (unreleased)

- TensorFold 1.1.5 (`84e0d560`), which carries @jschmied's FP4-Local pull requests, plus 4 patches of ours: admission
  of requests sent together, our GB10 tuning, prompt rounding independent of how a fill is split, and the first token
  of a long `"draft": false` prompt.
- Serving defaults: 262,144-token windows, 3 streams with the bf16 cache (the most the engine fits) and 5 with
  `KV=fp8`; thinking on, MTP drafts, `TF_FLASHNEXT_DRAFT_HEAD=affine4`, `TF_FLASHNEXT_CHUNK=4096`.
- Checks at those settings:
  - task suite 123 / 130 (bf16) and 124 (FP8), against stock 1.1.5's 122
  - concurrent == alone 28 / 28
  - drafted == `"draft": false`
  - reply hashes equal with the prompt cache on and off
- Numbers against stock TensorFold 1.1.5 in the README; `tools/cache_bench.py` measures the prompt cache.

## 0.1.0

- The first version, on TensorFold 1.1.4 with @jschmied's pull requests carried as patches.
