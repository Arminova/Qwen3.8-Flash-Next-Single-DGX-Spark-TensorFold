# config.sh: every pin and default in one place. prepare.sh, start.sh and stop.sh read it; override any value
# with an environment variable of the same name (or put it in a .env file next to this one).
[ -f "$(dirname "${BASH_SOURCE[0]}")/.env" ] && . "$(dirname "${BASH_SOURCE[0]}")/.env"

# TensorFold, the engine our patches apply to (release 1.1.5)
TF_REPO=${TF_REPO:-https://github.com/ashhart/TensorFold.git}
TF_REF=${TF_REF:-v1.1.5}
TF_COMMIT=${TF_COMMIT:-84e0d560303d85c50becb730720c04a56e38abe5}
# libtfgrammar (structured output and required tool calls) is built against this xgrammar release
XGRAMMAR_REPO=${XGRAMMAR_REPO:-https://github.com/mlc-ai/xgrammar.git}
XGRAMMAR_REF=${XGRAMMAR_REF:-v0.2.8}

# the checkpoint, exactly as published (NVFP4 experts, MXFP8 dense layers, CSF-compressed expert scales)
CHECKPOINT=${CHECKPOINT:-nvidia/Qwen3.8-Flash-Next-FP4-Local}
CHECKPOINT_REVISION=${CHECKPOINT_REVISION:-b9ce18a6887e18c089fdea8509a33e63429a3300}
# keep it on the internal NVMe: the model reads its n-gram table from these files while it serves
MODELS_DIR=${MODELS_DIR:-$HOME/models}
MODEL_DIR=${MODEL_DIR:-$MODELS_DIR/Qwen3.8-Flash-Next-FP4-Local}

# the image start.sh runs, built by prepare.sh from the Dockerfile (TensorFold at TF_COMMIT plus patches/)
IMAGE=${IMAGE:-qwen38-flash-next-spark-tensorfold:local}
BUILD_IMAGE=${BUILD_IMAGE:-nvidia/cuda:13.3.1-devel-ubuntu24.04}
RUN_IMAGE=${RUN_IMAGE:-nvidia/cuda:13.3.1-base-ubuntu24.04}
ZIG_VERSION=${ZIG_VERSION:-0.17.0}
ZIG_SHA256_AARCH64=${ZIG_SHA256_AARCH64:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}
ZIG_SHA256_X86_64=${ZIG_SHA256_X86_64:-1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026}
HF_HUB_VERSION=${HF_HUB_VERSION:-2.2.0}

# serving
NAME=${NAME:-qwen38-flash-next-tf}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3.8-flash-next}
BIND=${BIND:-127.0.0.1}
PORT=${PORT:-8888}
# KV cache: bf16 (the default: fastest on short and mixed loads) or fp8 (the long-window mode: about 42 % fewer cache
# bytes a token, faster prompts from 16K tokens up, room for 262,144 tokens x 5 streams)
KV=${KV:-bf16}
if [ "$KV" = fp8 ]; then CONTEXT=${CONTEXT:-262144}; PARALLEL=${PARALLEL:-5}; else CONTEXT=${CONTEXT:-65536}; PARALLEL=${PARALLEL:-4}; fi
MAX_TOKENS=${MAX_TOKENS:-32768}
# the settings every number in the README was measured with (4-bit draft head, 4,096-row prompt chunks)
ENGINE_ENV=${ENGINE_ENV:-TF_FLASHNEXT_DRAFT_HEAD=affine4 TF_FLASHNEXT_CHUNK=4096}
# more container environment, space-separated KEY=VALUE (switches: README "Settings")
EXTRA_ENV=${EXTRA_ENV:-}
# MemAvailable the start needs (GiB): the weights take ~77 GB, the windows and the prompt cache the rest
MIN_FREE_GIB=${MIN_FREE_GIB:-108}
