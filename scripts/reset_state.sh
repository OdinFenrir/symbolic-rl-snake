#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="./state"
rm -f "$STATE_DIR/symbolic_memory.msgpack" "$STATE_DIR/symbolic_memory.msgpack.bak" "$STATE_DIR/game_state.json" || true
echo "State reset complete."
