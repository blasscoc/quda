#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
for seed in 45 46 47; do
 ./run_fp4_roundoff.sh --trajectory --mode center_dual_mom --scalar64 --seed "$seed" --output "/work/benchmark-results/fp4-roundoff/compact-momentum-${seed}.json" > "benchmark-results/roundoff-compact-momentum-${seed}.log" 2>&1
 if [ "$seed" != 45 ]; then
 ./run_fp4_roundoff.sh --trajectory --mode baseline --seed "$seed" --output "/work/benchmark-results/fp4-roundoff/baseline-seed-${seed}.json" > "benchmark-results/roundoff-baseline-seed-${seed}.log" 2>&1
 fi
done
./run_fp4_roundoff.sh --mode center_dual --iters 24 --output /work/benchmark-results/fp4-roundoff/cold-24-no-refresh.json > benchmark-results/roundoff-cold-24-no-refresh.log 2>&1
