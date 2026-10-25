#!/bin/bash
# stop.sh: stops and removes the server container start.sh made.
D=$(cd "$(dirname "$0")" && pwd)
. "$D/config.sh"
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker stop "$NAME" > /dev/null && docker rm -f "$NAME" > /dev/null 2>&1
  echo "stopped $NAME"
else
  echo "$NAME is not running"
fi
