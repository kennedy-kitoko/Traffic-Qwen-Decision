#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JEVDIR="$ROOT/.deps/JevLight"
MODE="no-guardrail"
if [[ "$#" -gt 0 ]]; then MODE="$1"; fi
TRAFFIC="anon_3_4_jinan_real.json"
if [[ -v TRAFFIC_FILE ]]; then TRAFFIC="$TRAFFIC_FILE"; fi
MODEL="$ROOT/checkpoints/traffic-qwen-v1-dual-choice"
[[ -e "$JEVDIR/.git" ]] || { echo "Run scripts/install_jevlight_dependency.sh first" >&2; exit 2; }
[[ -f "$MODEL/adapter_model.safetensors" && -f "$MODEL/joint_head.safetensors" ]] || { echo "Local adapter missing. Train or obtain matching checkpoint." >&2; exit 2; }
[[ -f "$JEVDIR/data/Jinan/3_4/$TRAFFIC" && -f "$JEVDIR/data/Jinan/3_4/roadnet_3_4.json" ]] || { echo "Authorized Jinan input files missing under $JEVDIR/data/Jinan/3_4" >&2; exit 2; }
export PYTHONPATH="$ROOT:$JEVDIR"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
run_case() {
  local label="$1" seconds="$2" guardrail="$3" fallback="$4"
  local stamp="$(date -u +%Y%m%dT%H%M%SZ%N)"
  local out="$ROOT/results/cityflow/$label-$stamp"
  local guardarg="--no_guardrail"; local fallarg="--no_fallback"
  if [[ "$guardrail" == true ]]; then guardarg="--guardrail"; fi
  if [[ "$fallback" == true ]]; then fallarg="--fallback"; fi
  (cd "$JEVDIR" && python "$ROOT/scripts/evaluate_cityflow.py" --dataset jinan --traffic_file "$TRAFFIC" --run_counts "$seconds" --model_path "$MODEL" --device cuda --seed 3407 --proj_name "Traffic-Qwen-V1-$label" --output_dir "$out" "$guardarg" "$fallarg")
  if [[ "$seconds" == 3600 && -f "$JEVDIR/results/benchmark_jinan_with_traffic_qwen_v1.json" ]]; then
    cp "$JEVDIR/results/benchmark_jinan_with_traffic_qwen_v1.json" "$ROOT/results/benchmark_jinan_with_traffic_qwen_v1.json"
  fi
}
case "$MODE" in
  smoke) run_case smoke300 300 false false ;;
  no-guardrail) run_case no_guardrail 3600 false false ;;
  guardrail) run_case guardrail 3600 true true ;;
  all) run_case no_guardrail 3600 false false; run_case guardrail 3600 true true ;;
  *) echo "Usage: $0 {smoke|no-guardrail|guardrail|all}" >&2; exit 2 ;;
esac