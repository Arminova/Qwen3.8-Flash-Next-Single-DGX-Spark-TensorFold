#!/bin/bash
# start.sh: serves Qwen3.8-Flash-Next (NVIDIA FP4-Local) on one DGX Spark through TensorFold with our patches: an
# OpenAI-compatible API on BIND:PORT (127.0.0.1:8888 by default), thinking on, MTP drafts. KV=bf16 (default): 4
# streams of 65,536 tokens; KV=fp8: 5 streams of 262,144. Runs detached as container NAME and waits until the model
# answers (~75 s from a warm disk). Settings: config.sh (or environment variables / .env).
#   ./start.sh      KV=fp8 ./start.sh      ./stop.sh      docker logs -f qwen38-flash-next-tf
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
. "$D/config.sh"
say() { printf '\033[1;36m→\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m✗\033[0m %s\n' "$*"; exit 1; }

case "$KV" in bf16 | fp8) ;; *) die "KV must be bf16 or fp8 (got $KV)" ;; esac
docker image inspect "$IMAGE" > /dev/null 2>&1 || die "no image $IMAGE: run ./prepare.sh first"
[ -f "$MODEL_DIR/config.json" ] || die "no checkpoint in $MODEL_DIR: run ./prepare.sh first"
docker rm -f "$NAME" > /dev/null 2>&1 || true
avail=$(awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo)
if [ "${MEM_CHECK:-1}" = 1 ] && [ "$avail" -lt "$MIN_FREE_GIB" ]; then
  die "only ${avail} GiB of memory is free; this model needs ${MIN_FREE_GIB}. Stop other models (or MEM_CHECK=0)"
fi
host=$BIND
[ "$host" = 0.0.0.0 ] && host=127.0.0.1
[ "$BIND" = 0.0.0.0 ] && [ -z "${API_KEY:-}" ] && echo "  note: BIND=0.0.0.0 serves the whole network with no API key"
envs=(-e TF_FLASHNEXT_KV="$KV" -e TENSORFOLD_NO_LIVE=1)
for kv in $ENGINE_ENV $EXTRA_ENV; do envs+=(-e "$kv"); done
say "starting $SERVED_MODEL_NAME on $BIND:$PORT ($KV KV, $PARALLEL x $CONTEXT tokens, image $IMAGE)"
docker run -d --name "$NAME" --gpus all --ipc=host -p "$BIND:$PORT:$PORT" -v "$MODEL_DIR":/model:ro \
  "${envs[@]}" ${API_KEY:+-e TENSORFOLD_API_KEY="$API_KEY"} \
  "$IMAGE" serve /model --host 0.0.0.0 --port "$PORT" --name "$SERVED_MODEL_NAME" --context "$CONTEXT" \
  --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS" --thinking --temperature 1.0 --top-p 0.95 --top-k 20 \
  --no-update-check > /dev/null
t=0
until curl -sf "http://$host:$PORT/v1/models" ${API_KEY:+-H "Authorization: Bearer $API_KEY"} > /dev/null 2>&1; do
  sleep 5
  t=$((t + 5))
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || { docker logs --tail 20 "$NAME" 2>&1; die "the server exited"; }
  [ $t -ge 900 ] && die "not ready after ${t}s (still running: docker logs $NAME)"
done
say "ready after ${t}s: http://$BIND:$PORT/v1 (model $SERVED_MODEL_NAME). Stop with ./stop.sh"
