# Changelog

## 0.1.0 (unreleased)

- Qwen3.8-Flash-Next (NVIDIA FP4-Local) on one DGX Spark: TensorFold 1.1.4 (`eab85828`) plus 11 patches (@jschmied's
  #1143 / #1171 / #1164, burst admission, our GB10 tuning, prompt rounding independent of how a fill is split, the
  first token of a long undrafted prompt), built with Zig 0.17.0 and CUDA 13.3 in Docker, with
  libtfgrammar from xgrammar v0.2.8.
- Serving defaults: bf16 KV cache, 4 streams of 65,536 tokens, thinking on, MTP drafts, `TF_FLASHNEXT_DRAFT_HEAD=affine4`,
  `TF_FLASHNEXT_CHUNK=4096`; `KV=fp8` for 5 streams of 262,144.
- Checks: task suite 123 / 130 (bf16) and 124 (FP8); concurrent == alone 28 / 28; drafted == `"draft": false` (short
  and long prompts); exact changes checked with reply hashes (`tools/same.py`); the same hashes with the prompt cache
  on and off. Numbers in the README.
