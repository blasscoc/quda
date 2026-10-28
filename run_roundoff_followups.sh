#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
for warm in cold warm; do
 args=(); if [ "$warm" = warm ]; then args+=(--warm); fi
 ./run_fp4_roundoff.sh --mode center_dual --iters 24 --restart-every 4 "${args[@]}" --output "/work/benchmark-results/fp4-roundoff/${warm}-refresh.json" > "benchmark-results/roundoff-${warm}-refresh.log" 2>&1
done
for gain in 1 65536; do
 ./run_fp4_roundoff.sh --mode center_dual --iters 12 --rhs-scale .00001 --gain "$gain" --output "/work/benchmark-results/fp4-roundoff/tiny-${gain}.json" > "benchmark-results/roundoff-tiny-${gain}.log" 2>&1
done
for mode in baseline center_dual center_dual_mom; do
 ./run_fp4_roundoff.sh --trajectory --padded-momentum --mode "$mode" --output "/work/benchmark-results/fp4-roundoff/trajectory-${mode}.json" > "benchmark-results/roundoff-trajectory-${mode}.log" 2>&1
done

./run_fp4_roundoff.sh --trajectory --padded-momentum --mode center_dual_mom --scalar64 --output /work/benchmark-results/fp4-roundoff/trajectory-stable.json > benchmark-results/roundoff-trajectory-stable.log 2>&1
