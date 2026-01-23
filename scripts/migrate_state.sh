#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="./state"
mkdir -p "$STATE_DIR"

for f in symbolic_memory.msgpack symbolic_memory.msgpack.bak game_state.json; do
  if [ -f "./$f" ]; then
    mv -f "./$f" "$STATE_DIR/$f"
    echo "Moved $f -> state/"
  fi
done

echo "Done. You can now commit the state/ folder to snapshot your current run."
