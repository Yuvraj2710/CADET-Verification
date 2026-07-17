#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare CASEMA references with Python-generated (CADET-Verification) references.

Validates that the Python analytical solutions agree with the verified CASEMA
semi-analytic references to within a tight tolerance, establishing trust in
the Python-generated references for convergence tests.

The test extracts model parameters from the CASEMA reference files and
computes the Python analytical solution with matching parameters.

Usage
-----
    python scripts/test_reference_comparison.py

"""

import sys
from pathlib import Path

import numpy as np
import h5py

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utility.analytical_solutions import compute_analytical_outlet


def load_casema_reference(filepath):
    """Load solution and model parameters from a CASEMA reference file."""
    with h5py.File(filepath, 'r') as f:
        times = f['/output/solution/SOLUTION_TIMES'][()]
        outlet = f['/output/solution/unit_001/SOLUTION_OUTLET'][()]

        unit = f['/input/model/unit_001']
        unit_type = unit['UNIT_TYPE'][()].decode()
        section_times = f['/input/solver/sections/SECTION_TIMES'][()]

        params = {
            'velocity': float(np.atleast_1d(unit['VELOCITY'][()])[0]),
            'col_dispersion': float(np.atleast_1d(unit['COL_DISPERSION'][()])[0]),
            'col_length': float(unit['COL_LENGTH'][()]),
            'ka': float(unit['adsorption/LIN_KA'][()][0]),
            'kd': float(unit['adsorption/LIN_KD'][()][0]),
            'c_in': float(f['/input/model/unit_000/sec_000/CONST_COEFF'][()][0]),
            't_inj': float(section_times[1]),
        }

        if unit_type == 'LUMPED_RATE_MODEL_WITHOUT_PORES':
            model_type = 'LRM'
            params['total_porosity'] = float(unit['TOTAL_POROSITY'][()])

        elif unit_type == 'LUMPED_RATE_MODEL_WITH_PORES':
            model_type = 'LRMP'
            params['col_porosity'] = float(np.atleast_1d(unit['COL_POROSITY'][()])[0])
            params['par_porosity'] = float(np.atleast_1d(unit['PAR_POROSITY'][()])[0])
            params['par_radius'] = float(np.atleast_1d(unit['PAR_RADIUS'][()])[0])
            params['film_diffusion'] = float(np.atleast_1d(unit['FILM_DIFFUSION'][()])[0])

        elif unit_type == 'GENERAL_RATE_MODEL':
            model_type = 'GRM'
            params['col_porosity'] = float(np.atleast_1d(unit['COL_POROSITY'][()])[0])
            params['par_porosity'] = float(np.atleast_1d(unit['PAR_POROSITY'][()])[0])
            params['par_radius'] = float(np.atleast_1d(unit['PAR_RADIUS'][()])[0])
            params['film_diffusion'] = float(np.atleast_1d(unit['FILM_DIFFUSION'][()])[0])
            params['pore_diffusion'] = float(np.atleast_1d(unit['PAR_DIFFUSION'][()])[0])
            if 'PAR_SURFDIFFUSION' in unit:
                params['surface_diffusion'] = float(np.atleast_1d(unit['PAR_SURFDIFFUSION'][()])[0])

        else:
            raise ValueError(f"Unknown unit type: {unit_type}")

    return times, outlet, model_type, params


def compare_reference(casema_path, model_name, dps=30, atol=1e-10):
    """Compare CASEMA reference with Python analytical solution.

    Extracts parameters from the CASEMA file, computes the Python analytical
    solution at the same time points, and compares.
    """
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    casema_times, casema_outlet, model_type, params = load_casema_reference(casema_path)

    print(f"  Model type: {model_type}")
    print(f"  CASEMA: {len(casema_times)} time points, t=[{casema_times[0]:.1f}, {casema_times[-1]:.1f}]")
    print(f"  Parameters extracted from CASEMA file:")
    for k, v in sorted(params.items()):
        print(f"    {k} = {v}")

    print(f"\n  Computing Python analytical solution ({dps} dps)...")

    python_outlet = compute_analytical_outlet(
        model_type, params, casema_times, dps=dps
    )

    # Compute errors
    abs_diff = np.abs(casema_outlet - python_outlet)
    max_abs_err = np.max(abs_diff)
    mean_abs_err = np.mean(abs_diff)

    max_val = max(np.max(np.abs(casema_outlet)), np.max(np.abs(python_outlet)))
    max_rel_err = max_abs_err / max_val if max_val > 0 else 0.0

    print(f"\n  Max absolute error:  {max_abs_err:.6e}")
    print(f"  Mean absolute error: {mean_abs_err:.6e}")
    print(f"  Max relative error:  {max_rel_err:.6e}")
    print(f"  Max CASEMA value:    {np.max(np.abs(casema_outlet)):.6e}")
    print(f"  Max Python value:    {np.max(np.abs(python_outlet)):.6e}")

    passed = max_abs_err < atol
    status = "PASS" if passed else "FAIL"
    print(f"  Tolerance: {atol:.0e}")
    print(f"  Result: {status}")

    return passed, max_abs_err, max_rel_err


def main():
    casema_dir = project_root / 'data' / 'CASEMA_reference'

    # 1D chromatography models with CASEMA references
    models = [
        ('ref_LRM_dynLin_1comp_benchmark1.h5', 'LRM (dynLin, 1 comp)'),
        ('ref_LRMP_dynLin_1comp_benchmark1.h5', 'LRMP (dynLin, 1 comp)'),
        ('ref_GRM_dynLin_1comp_benchmark1.h5', 'GRM (dynLin, 1 comp)'),
    ]

    dps = 30
    atol = 1e-10
    results = []

    print("Reference Comparison: CASEMA vs Python Analytical Solutions")
    print(f"Parameters are extracted from CASEMA files to ensure identical models")
    print(f"Tolerance: {atol:.0e}, Precision: {dps} dps")

    for filename, model_name in models:
        casema_path = casema_dir / filename

        if not casema_path.exists():
            print(f"\n  SKIP {model_name}: CASEMA reference not found")
            continue

        passed, abs_err, rel_err = compare_reference(
            casema_path, model_name, dps=dps, atol=atol
        )
        results.append((model_name, passed, abs_err, rel_err))

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for model_name, passed, abs_err, rel_err in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {model_name}: max_abs={abs_err:.2e}, max_rel={rel_err:.2e}")
        if not passed:
            all_passed = False

    if not results:
        print("  No comparisons were run (missing reference files).")
        return 1

    if all_passed:
        print(f"\n  All {len(results)} comparisons passed.")
        return 0
    else:
        n_fail = sum(1 for _, p, _, _ in results if not p)
        print(f"\n  {n_fail}/{len(results)} comparisons FAILED.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
