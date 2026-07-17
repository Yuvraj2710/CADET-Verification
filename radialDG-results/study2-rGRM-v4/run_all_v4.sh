#!/bin/bash
# Run all rGRM v4 simulations on HPC — very low dispersion
# Usage: nohup bash run_all_v4.sh > run_all_v4.log 2>&1 &

set -euo pipefail

export LD_LIBRARY_PATH="$HOME/local-fix/lib"
CADET="$HOME/local-fix/bin/cadet-cli"
BASE="$HOME/CADET-Verification/output/test_cadet-core/radialDG/v4"

echo "=== rGRM v4 parallel run (very low dispersion) ==="
echo "Started: $(date)"
echo "Total configs: $(ls "$BASE"/*.h5 2>/dev/null | wc -l)"
echo ""

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

# All 33 configs, 8 parallel
echo "--- All 33 sims, 8 parallel ---"
ls "$BASE"/*.h5 | xargs -P 8 -I {} bash -c 'run_sim "$@"' _ {}
echo ""

echo "=== All done ==="
echo "Finished: $(date)"
