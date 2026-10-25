# Credits

- **The Qwen team** for Qwen3.8-Flash-Next.
- **NVIDIA** for the checkpoint this serves: the FP4 QAD export with NVFP4 experts and MXFP8 dense layers
  ([nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)).
- **Ash Hart and the TensorFold contributors** for [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0),
  the engine, server and kernels this builds on, including its Flash-Next family.
- **[@jschmied](https://github.com/jschmied)** for the FP4-Local pull requests carried in `patches/0001-0007` with his
  authorship: [#1143](https://github.com/ashhart/TensorFold/pull/1143) (NVFP4 experts on prompt tiles),
  [#1171](https://github.com/ashhart/TensorFold/pull/1171) (decode half tiles) and
  [#1164](https://github.com/ashhart/TensorFold/pull/1164) (hyper-connection normalize and mix), rebased onto 1.1.4.
- **[Mia (MiaAI-Lab)](https://github.com/MiaAI-Lab)** for the Flash-Next TensorFold build and recipe that set the bar
  here ([Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)):
  our FP8 KV cache uses its row format (e4m3 codes, one power-of-two scale per position and head), and our admission
  of requests sent together follows its burst admission.
- **[xgrammar](https://github.com/mlc-ai/xgrammar)** (Apache-2.0), built into the image as `libtfgrammar` for
  structured output and tool calls.
- The patches 0008 and 0009, the tests and the measurements are by [Arminova](https://github.com/Arminova).
