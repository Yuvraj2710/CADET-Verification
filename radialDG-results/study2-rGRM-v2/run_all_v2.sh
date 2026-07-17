#!/bin/bash
# Run all rGRM v2 simulations in parallel on HPC (ibt067)
# PAR_POLYDEG = POLYDEG (Breuer thesis format)
#
# Usage: nohup bash run_all_v2.sh > run_all_v2.log 2>&1 &
#
# Expects:
#   - h5 configs in ~/CADET-Verification/output/test_cadet-core/radialDG/v2/
#   - cadet-cli at ~/local-fix/bin/cadet-cli
#   - LD_LIBRARY_PATH=~/local-fix/lib

set -euo pipefail

export LD_LIBRARY_PATH="$HOME/local-fix/lib"
CADET="$HOME/local-fix/bin/cadet-cli"
BASE="$HOME/CADET-Verification/output/test_cadet-core/radialDG/v2"

echo "=== rGRM v2 parallel run ==="
echo "Started: $(date)"
echo "Total configs: $(ls "$BASE"/*.h5 2>/dev/null | wc -l)"
echo ""

# Run a single sim with timing and error handling
run_sim() {
    local h5="$1"
    local name=$(basename "$h5" .h5)
    local t0=$(date +%s)
    echo "[START] $name at $(date +%H:%M:%S)"
    if "$CADET" "$h5" > /dev/null 2>&1; then
        local t1=$(date +%s)
        echo "[DONE]  $name  ($(( t1 - t0 ))s)"
    else
        local rc=$?
        local t1=$(date +%s)
        echo "[FAIL]  $name  rc=$rc  ($(( t1 - t0 ))s)"
    fi
}
export -f run_sim
export CADET LD_LIBRARY_PATH

# All DG sims (lin + SMA) in parallel — 48 configs, run 24 at a time
echo "--- Phase 1: All DG sims (48 configs, 24 parallel) ---"
ls "$BASE"/radGRM_DG_*.h5 | xargs -P 24 -I {} bash -c 'run_sim "$@"' _ {}
echo ""

# All FV sims in parallel — 12 configs
# Small FV (Z128-Z1024) are fast; large FV (Z2048+) are slow
# Run all together, 12 parallel
echo "--- Phase 2: All FV sims (12 configs, all parallel) ---"
ls "$BASE"/radGRM_FV_*.h5 | xargs -P 12 -I {} bash -c 'run_sim "$@"' _ {}
echo ""

echo "=== All done ==="
echo "Finished: $(date)"
