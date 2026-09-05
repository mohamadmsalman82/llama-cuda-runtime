#!/usr/bin/env bash
# Compile-checks every CUDA source with an ordinary C++ compiler.
#
# The project is written on a machine with no NVIDIA toolkit, so without this
# the kernels would first meet a compiler on the GPU box. This strips the
# <<<...>>> launch syntax, which is the only part of CUDA C++ a host compiler
# cannot parse, and builds each translation unit against the stand-in headers in
# tools/hostcheck/. Every kernel body, every template instantiation, and every
# cuBLAS call signature gets type-checked.
#
# It proves the code compiles. It proves nothing about what it computes.
#
#   ./tools/hostcheck.sh [--keep]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1
cleanup() { [ "$KEEP" -eq 0 ] && rm -rf "$WORK"; }
trap cleanup EXIT

CXX="${CXX:-c++}"
FLAGS=(-std=c++17 -fsyntax-only -Wall -Wextra -Wno-unused-parameter
       -Wno-unused-function
       "-I$ROOT/tools/hostcheck" "-I$ROOT/src" "-I$ROOT/tests"
       "-I$ROOT/third_party")

if [ ! -f "$ROOT/third_party/nlohmann/json.hpp" ]; then
  echo "third_party/nlohmann/json.hpp is missing; configure the build once first" >&2
  exit 2
fi

# bash 3.2, which is what macOS ships, has no mapfile.
SOURCES=$(cd "$ROOT" && find src tools tests -name '*.cu' 2>/dev/null | sort)

status=0
for source in $SOURCES; do
  target="$WORK/$(echo "$source" | tr '/' '_')"
  mkdir -p "$(dirname "$target")"
  # A kernel launch is `name<<<grid, block, shared, stream>>>(args)`. The
  # configuration never contains a '>' of its own, so removing it leaves an
  # ordinary function call that still type-checks its arguments.
  sed -E 's/<<<[^>]*>>>//g' "$ROOT/$source" > "$target"

  if "$CXX" "${FLAGS[@]}" -x c++ "$target" 2>"$WORK/err"; then
    printf '  ok    %s\n' "$source"
  else
    printf '  FAIL  %s\n' "$source"
    sed "s|$target|$source|g" "$WORK/err" >&2
    status=1
  fi
done

if [ "$status" -eq 0 ]; then
  echo "all CUDA sources compile as host C++"
else
  echo "host compile check failed" >&2
fi
exit "$status"
