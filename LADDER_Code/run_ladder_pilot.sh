#!/usr/bin/env bash
# LADDER pilot: A1/A2/A3 x {baseline, refusal_suppression, role_play_en}, seed 42.
# 50 questions x 2 samples = 100 tests per run x 9 runs = 900 generations.
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

MODEL=deepseek-chat
Q=data/questions.jsonl
ADJ=data/adjacency_matrix_knowledge_graph.csv
OUT=output/ladder_pilot_seed42
END=50
SAMPLES=2
SEED=42
CONC=10

ARMS="binary graded_ticket graded_prompt"
ATTACKS="baseline refusal_suppression role_play_en"

echo "############ LADDER PILOT START ############"
for atk in $ATTACKS; do
  for arm in $ARMS; do
    echo "===== arm=$arm attack=$atk ====="
    python pipeline/run_ladder_test.py \
      --questions "$Q" --adjacency "$ADJ" --model "$MODEL" \
      --arm "$arm" --attack "$atk" --end "$END" --samples "$SAMPLES" \
      --seed "$SEED" --concurrent "$CONC" --output "$OUT" \
      2>&1 | grep -E "arm=|Intended disclosure|Results:|saved to" || true
  done
done

echo "############ DISCLOSURE EVAL ############"
python evaluation/disclosure_evaluation.py \
  --input "$OUT" \
  --output evaluation/output/ladder_pilot_disclosure \
  --model gpt-5.4-mini --provider apinebula --concurrent 10 \
  2>&1 | grep -E "over=|Evaluating|✅" || true

echo "############ LADDER PILOT DONE ############"
