# Changelog

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
