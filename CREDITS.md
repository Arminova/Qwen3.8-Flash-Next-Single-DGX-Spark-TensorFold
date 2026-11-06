# Credits

- **The Qwen team** for Qwen3.8-Flash-Next.
- **NVIDIA** for the checkpoint this serves: the FP4 QAD export with NVFP4 experts and MXFP8 dense layers
  ([nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)).
- **Ash Hart and the TensorFold contributors** for [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0),
  the engine, server and kernels this builds on, including its Flash-Next family.
- **[@jschmied](https://github.com/jschmied)** for the FP4-Local pull requests this build stands on:
  [#1143](https://github.com/ashhart/TensorFold/pull/1143) (NVFP4 experts on prompt tiles),
  [#1171](https://github.com/ashhart/TensorFold/pull/1171) (decode half tiles) and
  [#1164](https://github.com/ashhart/TensorFold/pull/1164) (hyper-connection normalize and mix), in TensorFold since
  1.1.5 (the first version of this recipe carried them as patches on 1.1.4, with his authorship).
- **[Mia (MiaAI-Lab)](https://github.com/MiaAI-Lab)** for the Flash-Next TensorFold build and recipe that set the bar
  here ([Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)):
  the earlier build of this repository (commit `3bfc9fc`) used its FP8 KV row format (e4m3 codes, one power-of-two
  scale per position and head) and followed its burst admission.
- **[xgrammar](https://github.com/mlc-ai/xgrammar)** (Apache-2.0), built into the image as `libtfgrammar` for
  structured output and tool calls.
- The settings, the benchmark tools and the measurements are by [Arminova](https://github.com/Arminova), who also wrote
  the TensorFold pull requests [#1368](https://github.com/ashhart/TensorFold/pull/1368),
  [#1590](https://github.com/ashhart/TensorFold/pull/1590), [#1763](https://github.com/ashhart/TensorFold/pull/1763),
  [#1764](https://github.com/ashhart/TensorFold/pull/1764), [#1786](https://github.com/ashhart/TensorFold/pull/1786)
  and [#2160](https://github.com/ashhart/TensorFold/pull/2160) that the earlier four patches became.
