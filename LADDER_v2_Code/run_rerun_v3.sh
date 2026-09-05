#!/usr/bin/env bash
# LADDER six-dim re-run (v3) — AFTER Bug#2 fix (Agent B string-invoke).
# Same config as rerun_sixdim_v2 but with fixed nodes_v2.py.
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

MODEL=deepseek-chat
Q=../LADDER_Code/data/questions.jsonl
ADJ=../LADDER_Code/data/adjacency_matrix_knowledge_graph.csv
OUT=output/rerun_sixdim_v3
START=0
END=30
STUDENTS=30
SEED=42
CONC=10

echo "### RERUN V3 (Agent B fixed) — $(date)"

echo ""
echo "=== PHASE 1: A2 + A3 baseline ==="
python -u pipeline/run_full_experiment.py \
  --questions "$Q" --adjacency "$ADJ" --model "$MODEL" \
  --arms A2,A3 --attack baseline \
  --start "$START" --end "$END" --students "$STUDENTS" \
  --seed "$SEED" --concurrent "$CONC" --output "$OUT"

echo ""
echo "=== PHASE 2: A3 refusal_suppression ==="
python -u pipeline/run_full_experiment.py \
  --questions "$Q" --adjacency "$ADJ" --model "$MODEL" \
  --arms A3 --attack refusal_suppression \
  --start "$START" --end "$END" --students "$STUDENTS" \
  --seed "$SEED" --concurrent "$CONC" --output "$OUT"

echo ""
echo "=== PHASE 3: A3 role_play_en ==="
python -u pipeline/run_full_experiment.py \
  --questions "$Q" --adjacency "$ADJ" --model "$MODEL" \
  --arms A3 --attack role_play_en \
  --start "$START" --end "$END" --students "$STUDENTS" \
  --seed "$SEED" --concurrent "$CONC" --output "$OUT"

echo ""
echo "RERUN_V3_ALL_DONE — $(date)"
