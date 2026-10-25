#!/bin/bash
# prepare.sh: everything start.sh needs, once. Checks the machine, downloads the checkpoint as published (~96 GiB,
# resumable; nothing is converted), then builds the image (TensorFold at TF_COMMIT plus patches/, and libtfgrammar).
# Safe to run again: finished steps are skipped. --rebuild builds the image again even when it is current.
#   ./prepare.sh [--no-download] [--rebuild]
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
. "$D/config.sh"
download=1 rebuild=0
for a in "$@"; do
  case "$a" in
    --no-download) download=0 ;;
    --rebuild) rebuild=1 ;;
    *) echo "usage: ./prepare.sh [--no-download] [--rebuild]"; exit 2 ;;
  esac
done
say() { printf '\033[1;36m→\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m✗\033[0m %s\n' "$*"; exit 1; }

say "checking the machine"
command -v docker > /dev/null || die "docker is not installed"
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | head -1 || true)
[ -n "$gpu" ] || die "nvidia-smi found no GPU"
case "$gpu" in *GB10*) ;; *) echo "  note: built and tested on a DGX Spark (GB10); this GPU is: $gpu" ;; esac
total=$(awk '/^MemTotal:/ {printf "%d", $2 / 1048576}' /proc/meminfo)
[ "$total" -ge 115 ] || die "this needs a 128 GB machine (MemTotal ${total} GiB)"

if [ $download = 1 ]; then
  mkdir -p "$MODEL_DIR"
  need=$((100 - $(du -s --block-size=1G "$MODEL_DIR" | cut -f1)))
  free=$(df --output=avail --block-size=1G "$MODEL_DIR" | tail -1 | tr -d ' ')
  [ "$need" -le 0 ] || [ "$free" -ge "$need" ] || die "$MODEL_DIR needs ${need} GB more free disk (has ${free} GB)"
  say "downloading $CHECKPOINT @ ${CHECKPOINT_REVISION:0:8} into $MODEL_DIR (~96 GiB, resumable)"
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/tmp/hf -e HF_HUB_DISABLE_TELEMETRY=1 \
    ${HF_TOKEN:+-e HF_TOKEN} -v "$MODEL_DIR":/out python:3.12-slim sh -c \
    "pip install -q --no-cache-dir --disable-pip-version-check --no-warn-script-location huggingface_hub==$HF_HUB_VERSION &&
     PATH=/tmp/.local/bin:\$PATH hf download '$CHECKPOINT' --revision '$CHECKPOINT_REVISION' --local-dir /out"
fi
[ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/model.safetensors.index.json" ] || die "$MODEL_DIR: no config.json or index: run ./prepare.sh without --no-download"
n=$(ls "$MODEL_DIR/tensors" 2> /dev/null | grep -c '^model-[0-9]*-of-00021\.safetensors$' || true)
[ "$n" = 21 ] || die "$MODEL_DIR/tensors: expected 21 weight files, found $n"
say "checkpoint complete: 21 weight files"

hash=$(cat "$D"/patches/*.patch "$D"/Dockerfile | sha256sum | cut -c1-12)
have=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2> /dev/null || true)
if [ $rebuild = 0 ] && [ "$have" = "$hash" ]; then
  say "image $IMAGE is current (patches $hash)"
  exit 0
fi
avail=$(awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo)
[ "$avail" -ge 24 ] || die "the build needs ~24 GB free memory (MemAvailable ${avail} GiB): stop a running model first"
say "building $IMAGE: TensorFold ${TF_COMMIT:0:8} + $(ls "$D"/patches/*.patch | wc -l) patches (about 15 min on a DGX Spark)"
docker build ${DOCKER_BUILD_ARGS:-} -t "$IMAGE" \
  --build-arg BUILD_IMAGE="$BUILD_IMAGE" --build-arg RUN_IMAGE="$RUN_IMAGE" \
  --build-arg TF_REPO="$TF_REPO" --build-arg TF_COMMIT="$TF_COMMIT" \
  --build-arg XGRAMMAR_REPO="$XGRAMMAR_REPO" --build-arg XGRAMMAR_REF="$XGRAMMAR_REF" --build-arg ZIG_VERSION="$ZIG_VERSION" \
  --build-arg ZIG_SHA256_AARCH64="$ZIG_SHA256_AARCH64" --build-arg ZIG_SHA256_X86_64="$ZIG_SHA256_X86_64" \
  --build-arg PATCHES_HASH="$hash" "$D"
say "done: ./start.sh serves it"
