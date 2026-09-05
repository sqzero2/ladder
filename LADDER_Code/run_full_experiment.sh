#!/usr/bin/env bash
# LADDER definitive run: E0/E1/A1/A2/A3 x {baseline, refusal_suppression, role_play_en}
# 100 questions x 2 samples x seed 42 → 200 tests per run, 3000 total generations.
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

MODEL=deepseek-chat
Q=data/questions.jsonl
ADJ=data/adjacency_matrix_knowledge_graph.csv
OUT=output/ladder_full_seed42
END=100
SAMPLES=2
SEED=42
CONC=10

ATTACKS="baseline refusal_suppression role_play_en"

echo "############################################################"
echo "# LADDER FULL EXPERIMENT — 5 arms x 3 attacks x 100Q x 2s #"
echo "############################################################"
echo "start: $(date)"

# --- Phase 1: all generations (5 arms x 3 attacks) ---

for atk in $ATTACKS; do
  echo ""
  echo "===== PHASE1: attack=$atk ====="

  echo "  [E0] adaptive baseline..."
  python pipeline/run_adaptive_test.py \
    --questions "$Q" --adjacency "$ADJ" --model "$MODEL" \
    --attack "$atk" --end "$END" --samples "$SAMPLES" \
    --seed "$SEED" --concurrent "$CONC" --output "$OUT" \
    2>&1 | grep -E "Results:|saved to:" || true

  echo "  [E1] SHaPE Pipeline..."
  python pipeline/run_shape_test.py \
    --questions "$Q" --adjacency "$ADJ" --model "$MODEL" \
    --attack "$atk" --end "$END" --samples "$SAMPLES" \
    --seed "$SEED" --concurrent "$CONC" --output "$OUT" \
    2>&1 | grep -E "Results:|saved to:" || true

  for arm in binary graded_ticket graded_prompt; do
    echo "  [LADDER $arm]..."
    python pipeline/run_ladder_test.py \
      --questions "$Q" --adjacency "$ADJ" --model "$MODEL" \
      --arm "$arm" --attack "$atk" --end "$END" --samples "$SAMPLES" \
      --seed "$SEED" --concurrent "$CONC" --output "$OUT" \
      2>&1 | grep -E "Results:|saved to:" || true
  done

  echo "  [DONE] attack=$atk at $(date)"
done

echo ""
echo "===== PHASE2: old-4-metric eval ====="
python evaluation/batch_evaluate.py \
  --input-dir "$OUT" \
  --output-dir evaluation/output/ladder_full_old4 \
  --evaluator-model gpt-5.4-mini --evaluator-provider apinebula \
  --max-concurrent-files 5 --max-concurrent-evaluations 10 \
  2>&1 | grep -E "✅|❌|处理|完成|saved" || true

echo ""
echo "===== PHASE3: disclosure-axis eval (canonical intended) ====="
python evaluation/disclosure_evaluation.py \
  --input "$OUT" \
  --output evaluation/output/ladder_full_disclosure \
  --adjacency "$ADJ" \
  --model gpt-5.4-mini --provider apinebula --concurrent 10 \
  2>&1 | grep -E "over=|Evaluating|canonical|✅|Error" || true

echo ""
echo "===== PHASE4: analyze ====="
python analyze_ladder.py evaluation/output/ladder_full_disclosure

echo "############################################################"
echo "# DONE — $(date)"
echo "############################################################"
