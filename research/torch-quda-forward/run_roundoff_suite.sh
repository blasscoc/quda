#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
./run_fp4_roundoff_tests.sh > benchmark-results/roundoff-tests.log 2>&1
for gauge in cold warm; do
 args=(); if [ "$gauge" = warm ]; then args+=(--warm); fi
 for mode in baseline sr dual center center_dual; do
  ./run_fp4_roundoff.sh --mode "$mode" "${args[@]}" --output "/work/benchmark-results/fp4-roundoff/${gauge}-${mode}.json" > "benchmark-results/roundoff-${gauge}-${mode}.log" 2>&1
 done
done
./run_roundoff_followups.sh
./run_roundoff_confirm.sh
python3 analyze_fp4_roundoff.py
