#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_NAME=fc-agent          # default file name
OUT_PATH="$SCRIPT_DIR/$DEFAULT_NAME"

if [[ $# -gt 0 ]]; then
  case "$1" in
    */*) # caller supplied a path (e.g. ./bin/fc, /tmp/fc)
      OUT_PATH="$(realpath -m -- "$1")"
      ;;
    *)   # caller supplied just a new file name (e.g. fc-dev)
      OUT_PATH="$SCRIPT_DIR/$1"
      ;;
  esac
fi

# Build inside a disposable Alpine container bound to $SCRIPT_DIR
docker run --rm \
  -v "$SCRIPT_DIR":/src \
  -w /src \
  alpine:3 \
  sh -exc '
    apk add --no-cache build-base musl-dev linux-headers
    gcc -Os -static -s -o fc-agent fc_agent.c
    strip fc-agent || true
  '

# Move/rename if the target name or location differs
if [[ "$OUT_PATH" != "$SCRIPT_DIR/fc-agent" ]]; then
  mv -f -- "$SCRIPT_DIR/fc-agent" "$OUT_PATH"
fi

echo "[+] Built $(basename "$OUT_PATH") at $(dirname "$OUT_PATH")"
file "$OUT_PATH" || true
