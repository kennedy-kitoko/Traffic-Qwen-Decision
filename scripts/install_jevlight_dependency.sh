#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMIT="d92caa655e5a7487a49361626673df809eee9129"
DEST="$ROOT/.deps/JevLight"
if [[ -e "$DEST/.git" ]]; then
  test "$(git -C "$DEST" rev-parse HEAD)" = "$COMMIT" || { echo "JevLight checkout is not pinned commit $COMMIT" >&2; exit 2; }
else
  mkdir -p "$(dirname "$DEST")"
  git clone --filter=blob:none --no-checkout https://github.com/usail-hkust/JevLight.git "$DEST"
  git -C "$DEST" sparse-checkout init --cone
  git -C "$DEST" sparse-checkout set models utils results
  git -C "$DEST" checkout --detach "$COMMIT"
fi
if git -C "$DEST" apply --reverse --check "$ROOT/third_party/jevlight-runtime.patch" >/dev/null 2>&1; then
  echo "TrafficQwen runtime patch already applied."
else
  git -C "$DEST" apply --check "$ROOT/third_party/jevlight-runtime.patch"
  git -C "$DEST" apply "$ROOT/third_party/jevlight-runtime.patch"
fi
echo "Pinned JevLight code is ready. Traffic data are not downloaded."