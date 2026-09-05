#!/usr/bin/env bash
# Disclosure eval on rerun_e0e1_v1 (E0+E1 fixed), via apinebula + proxy.
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
export HTTPS_PROXY=http://127.0.0.1:52106
export HTTP_PROXY=http://127.0.0.1:52106

echo "### EVAL E0+E1 — $(date)"
python -u pipeline/run_disclosure_eval.py \
  --input output/rerun_e0e1_v1 \
  --output output/rerun_e0e1_v1_disclosure \
  --model gpt-5.4-mini --provider apinebula --concurrent 10

echo "EVAL_E0E1_DONE — $(date)"
