#!/usr/bin/env bash
set -euo pipefail

# Generate compile_commands.json for the C++ extension and expose it at repo root.
# Usage:
#   ./gen_compile_commands.sh [additional cmake args]
# Env vars:
#   BUILD_DIR   - build directory (default: "$ROOT_DIR/build")
#   GENERATOR   - cmake generator to use if available (default: Ninja)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-"$ROOT_DIR/build"}"
GENERATOR="${GENERATOR:-Ninja}"

mkdir -p "$BUILD_DIR"

cmake_args=(
  -S "$ROOT_DIR/torch_ttnn/cpp_extension"
  -B "$BUILD_DIR"
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)

# Prefer Ninja if available; otherwise let CMake pick a default
if command -v ninja >/dev/null 2>&1; then
  cmake_args+=( -G "$GENERATOR" )
fi

# Pass through any extra arguments to CMake
cmake "${cmake_args[@]}" "$@"

if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
  echo "ERROR: compile_commands.json was not generated in '$BUILD_DIR'" >&2
  exit 1
fi

target_link="$ROOT_DIR/compile_commands.json"
rm -f "$target_link"

# Prefer a symlink for tools that watch for changes; fallback to copy if symlink fails
if ln -s "$BUILD_DIR/compile_commands.json" "$target_link" 2>/dev/null; then
  echo "Symlinked compile_commands.json -> $BUILD_DIR/compile_commands.json"
else
  cp "$BUILD_DIR/compile_commands.json" "$target_link"
  echo "Copied compile_commands.json to project root"
fi

echo "Done: $target_link"


