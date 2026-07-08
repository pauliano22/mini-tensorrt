#!/usr/bin/env bash
# Build the browser demo. Requires emscripten (https://emscripten.org).
# Output: engine.js — a single self-contained file (WASM embedded) exposing
# MiniTRT() -> Module with _predict(pixels_ptr, logits_ptr).
set -euo pipefail
cd "$(dirname "$0")"

emcc -O2 -I../include \
    ../src/ir.cpp ../src/optimizer.cpp ../src/backend.cpp demo.cpp \
    -o engine.js \
    -sEXPORTED_FUNCTIONS=_predict,_malloc,_free \
    -sEXPORTED_RUNTIME_METHODS=HEAPF32 \
    -sMODULARIZE=1 -sEXPORT_NAME=MiniTRT \
    -sSINGLE_FILE=1 -sALLOW_MEMORY_GROWTH=1

echo "Built engine.js ($(wc -c < engine.js) bytes)"
