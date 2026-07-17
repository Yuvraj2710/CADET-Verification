#!/usr/bin/env python3
"""Rerun rGRM DG sims with PAR_NELEM co-refinement (Jan Breuer's strategy).

Instead of fixed PAR_NELEM=1, scale particle elements with bulk:
    PAR_NELEM = max(1, nElem // 4)

This ensures particle error doesn't bottleneck DG bulk convergence.

Usage:
    python rerun_rGRM_DG_corefine.py [--results-dir DIR] [--cadet-path PATH] [--workers N] [--dry-run]
"""

import os
import sys
import h5py
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# Default paths (HPC)
DEFAULT_RESULTS_DIR = os.path.expanduser(
    "~/CADET-Verification/output/test_cadet-core/radialDG")
DEFAULT_CADET_PATH = os.path.expanduser("~/local-fix/bin/cadet-cli")

# rGRM DG configs — lin P1/P2 have 7 levels (up to Z256), rest have 6 (up to Z128)
CONFIGS = [
    {
        'prefix': 'radGRM_DG_lin_1comp_varCoeff',
        'poly_degs': [1, 2],
        'nelem_start': 4,
        'n_disc': 7,  # 4, 8, 16, 32, 64, 128, 256
    },
    {
        'prefix': 'radGRM_DG_lin_1comp_varCoeff',
        'poly_degs': [3, 4],
        'nelem_start': 4,
        'n_disc': 6,  # 4, 8, 16, 32, 64, 128
    },
    {
        'prefix': 'radGRM_DG_SMA_4comp_varCoeff',
        'poly_degs': [1, 2, 3, 4],
        'nelem_start': 4,
        'n_disc': 6,  # 4, 8, 16, 32, 64, 128
    },
]


def get_filename(prefix, p, nElem):
    return f"{prefix}_P{p}_DG_P{p}Z{nElem}.h5"


def modify_and_run(h5_path, cadet_path, nElem, dry_run=False):
    """Modify PAR_NELEM in h5 file and rerun."""
    par_nelem = max(1, nElem // 4)

    # Read current PAR_NELEM
    try:
        with h5py.File(h5_path, 'r') as f:
            unit_keys = [k for k in f['input/model'].keys() if k.startswith('unit_')]
            for uk in unit_keys:
                pt_path = f'input/model/{uk}/particle_type_000/discretization'
                if pt_path in f:
                    old_nelem = int(f[pt_path]['PAR_NELEM'][()]) if 'PAR_NELEM' in f[pt_path] else 'N/A'
                    old_ncells = int(f[pt_path]['NCELLS'][()]) if 'NCELLS' in f[pt_path] else 'N/A'
                    break
            else:
                return h5_path, False, "No particle_type_000 found"
    except Exception as e:
        return h5_path, False, f"Read error: {e}"

    if old_nelem == par_nelem:
        return h5_path, True, f"Already PAR_NELEM={par_nelem}, skipping"

    basename = os.path.basename(h5_path)
    info = f"Z{nElem}: PAR_NELEM {old_nelem} -> {par_nelem}, NCELLS {old_ncells} -> {par_nelem}"

    if dry_run:
        return basename, True, f"DRY RUN: would set {info}"

    # Modify h5 — handle both scalar and array datasets
    try:
        with h5py.File(h5_path, 'a') as f:
            for uk in [k for k in f['input/model'].keys() if k.startswith('unit_')]:
                pt_path = f'input/model/{uk}/particle_type_000/discretization'
                if pt_path in f:
                    for key in ['PAR_NELEM', 'NCELLS']:
                        if key in f[pt_path]:
                            del f[pt_path][key]
                            f[pt_path].create_dataset(key, data=par_nelem)
    except Exception as e:
        return basename, False, f"Write error: {e}"

    # Delete existing output
    try:
        with h5py.File(h5_path, 'a') as f:
            if 'output' in f:
                del f['output']
    except:
        pass

    # Run CADET
    try:
        result = subprocess.run(
            [cadet_path, h5_path],
            capture_output=True, text=True, timeout=36000  # 10h timeout
        )
        if result.returncode != 0:
            return basename, False, f"CADET failed (rc={result.returncode}): {result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return basename, False, "CADET timed out (2h)"
    except Exception as e:
        return basename, False, f"Run error: {e}"

    return basename, True, f"OK ({info})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--cadet-path', default=DEFAULT_CADET_PATH)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    # Build list of (h5_path, nElem) pairs
    tasks = []
    for cfg in CONFIGS:
        prefix = cfg['prefix']
        for p in cfg['poly_degs']:
            for disc_idx in range(cfg['n_disc']):
                nElem = cfg['nelem_start'] * 2**disc_idx
                fname = get_filename(prefix, p, nElem)
                fpath = os.path.join(args.results_dir, fname)
                if os.path.exists(fpath):
                    tasks.append((fpath, nElem))
                else:
                    print(f"MISSING: {fname}")

    print(f"\nFound {len(tasks)} h5 files to process")
    print(f"CADET: {args.cadet_path}")
    print(f"Workers: {args.workers}")
    if args.dry_run:
        print("*** DRY RUN — no modifications ***")
    print()

    # Run in parallel
    done = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(modify_and_run, path, args.cadet_path, nElem, args.dry_run): path
            for path, nElem in tasks
        }
        for future in as_completed(futures):
            name, ok, msg = future.result()
            done += 1
            if not ok:
                failed += 1
            status = "OK" if ok else "FAIL"
            print(f"[{done}/{len(tasks)}] {status}: {name} — {msg}")

    print(f"\nDone: {done}, Failed: {failed}")


if __name__ == '__main__':
    main()
