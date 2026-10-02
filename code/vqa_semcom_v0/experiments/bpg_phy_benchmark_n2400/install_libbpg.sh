#!/usr/bin/env bash
# Isolated dependency build; no production repo edits or system install.
set -euo pipefail
BPG_PREFIX=${1:?Usage: install_libbpg.sh PREFIX [BUILD_DIR]}
BPG_BUILD_DIR=${2:-"$BPG_PREFIX/build"}
mkdir -p "$BPG_PREFIX/bin" "$BPG_BUILD_DIR"
for BPG_REQUIRED in gcc g++ make python3 curl tar patch; do
    command -v "$BPG_REQUIRED" >/dev/null || { echo "Missing $BPG_REQUIRED; install build-essential libpng-dev libjpeg-dev python3-venv" >&2; exit 2; }
done
curl -fL --retry 2 https://bellard.org/bpg/libbpg-0.9.8.tar.gz -o "$BPG_BUILD_DIR/libbpg-0.9.8.tar.gz"
echo "c0788e23bdf1a7d36cb4424ccb2fae4c7789ac94949563c4ad0e2569d3bf0095  $BPG_BUILD_DIR/libbpg-0.9.8.tar.gz" | sha256sum -c -
tar -xzf "$BPG_BUILD_DIR/libbpg-0.9.8.tar.gz" -C "$BPG_BUILD_DIR"
python3 -m venv "$BPG_BUILD_DIR/build-venv"
"$BPG_BUILD_DIR/build-venv/bin/pip" install cmake==3.27.9
# Compatible scalar build matches the locally tested tool family; freeze binary hashes per VM.
"$BPG_BUILD_DIR/build-venv/bin/python" - "$BPG_BUILD_DIR/libbpg-0.9.8" <<'PY'
from pathlib import Path
import sys
r=Path(sys.argv[1])
for name in ['bpgenc.c','bpgdec.c']:
    p=r/name;p.write_text('#ifndef _GNU_SOURCE\n#define _GNU_SOURCE\n#endif\n#include <string.h>\n#include <strings.h>\n'+p.read_text())
p=r/'x265/source/CMakeLists.txt';s=p.read_text();s=s.replace('message(FATAL_ERROR "Yasm 1.2.0 or later must be installed")','message(WARNING "Assembly disabled for isolated GAP01 portable build")');p.write_text(s)
PY
export PATH="$BPG_BUILD_DIR/build-venv/bin:$PATH"
make -C "$BPG_BUILD_DIR/libbpg-0.9.8" x265.out USE_BPGVIEW= CMAKE_OPTS=-DENABLE_ASSEMBLY=OFF
make -C "$BPG_BUILD_DIR/libbpg-0.9.8" -j4 bpgenc bpgdec USE_BPGVIEW=
install -m755 "$BPG_BUILD_DIR/libbpg-0.9.8/bpgenc" "$BPG_PREFIX/bin/bpgenc"
install -m755 "$BPG_BUILD_DIR/libbpg-0.9.8/bpgdec" "$BPG_PREFIX/bin/bpgdec"
sha256sum "$BPG_PREFIX/bin/bpgenc" "$BPG_PREFIX/bin/bpgdec" > "$BPG_PREFIX/binary_sha256.txt"
