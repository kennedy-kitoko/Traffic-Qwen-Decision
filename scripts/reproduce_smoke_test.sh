#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
test -s data/traffic_qwen/v1_dual_choice/train.jsonl || { echo "Prepare authorized V1 data first; see data/README.md" >&2; exit 2; }
python scripts/check_environment.py
python scripts/train_traffic_qwen.py --config configs/smoke_test.json --smoke
if [[ "$#" -gt 0 ]]; then
  if [[ "$1" == "cityflow" ]]; then bash scripts/reproduce_jinan.sh smoke; fi
fi