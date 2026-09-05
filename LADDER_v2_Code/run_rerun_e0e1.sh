#!/usr/bin/env bash
# E0 (Vanilla) + E1 (SHaPE) re-run — after fixing:
#   E0: string→message list (was char-iterating) + now actually feeds the question
#   E1: SHaPE trace now records binary disclosure_level (was defaulting intended=0)
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

MODEL=deepseek-chat
Q=../LADDER_Code/data/questions.jsonl
ADJ=../LADDER_Code/data/adjacency_matrix_knowledge_graph.csv
OUT=output/rerun_e0e1_v1
START=0
END=30
STUDENTS=30
SEED=42
CONC=10

echo "### RERUN E0+E1 (baseline) — $(date)"

python -u pipeline/run_full_experiment.py \
  --questions "$Q" --adjacency "$ADJ" --model "$MODEL" \
  --arms E0,E1 --attack baseline \
  --start "$START" --end "$END" --students "$STUDENTS" \
  --seed "$SEED" --concurrent "$CONC" --output "$OUT"

echo "RERUN_E0E1_DONE — $(date)"
