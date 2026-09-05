#!/usr/bin/env bash
# Disclosure eval on rerun_sixdim_v3 (Agent B fixed), via apinebula + proxy.
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
export HTTPS_PROXY=http://127.0.0.1:52106
export HTTP_PROXY=http://127.0.0.1:52106

echo "### EVAL V3 — $(date)"
python -u pipeline/run_disclosure_eval.py \
  --input output/rerun_sixdim_v3 \
  --output output/rerun_sixdim_v3_disclosure \
  --model gpt-5.4-mini --provider apinebula --concurrent 10

echo "EVAL_V3_ALL_DONE — $(date)"
