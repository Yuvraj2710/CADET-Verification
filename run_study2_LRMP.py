#!/usr/bin/env python3
"""Run Study 2 for rLRMP: lin 1-comp + SMA 4-comp.

Parameters from the verification table:
  r_in=0.01, r_out=0.26, velocity_coeff=4.5e-3, col_dispersion=1e-5
  col_porosity=0.6, par_porosity=0.2, total_porosity=0.68
  film_diffusion=3.3e-3, par_radius=1e-4
"""

import sys
import os

# Ensure cadet-cli is on PATH for autodetection in convergence analysis
_cadet_dir = os.path.expanduser('~/local-fix/bin')
if os.path.isdir(_cadet_dir) and _cadet_dir not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _cadet_dir + ':' + os.environ.get('PATH', '')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.radialDG as radialDG

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/test_cadet-core/radialDG/LRMP')
    parser.add_argument('--cadet', default=os.path.expanduser('~/local-fix'))
    parser.add_argument('--njobs', type=int, default=1)
    parser.add_argument('--config', choices=['lin', 'sma', 'both'], default='both')
    parser.add_argument('--skip-fv', action='store_true')
    parser.add_argument('--fv-start', type=int, default=None,
                        help='Start FV sweep from this cell count (e.g. 8192)')
    parser.add_argument('--polydegs', type=int, nargs='+', default=None,
                        help='DG polynomial degrees to run (e.g. 4 5)')
    parser.add_argument('--dg-n-disc', type=int, default=None,
                        help='Number of DG refinement levels (default: 10)')
    args = parser.parse_args()

    print("Study 2 rLRMP: lin 1-comp + SMA 4-comp")
    print(f"Output: {args.output}")
    print(f"CADET: {args.cadet}")
    print(f"Config: {args.config}")

    if args.config in ('lin', 'both'):
        print("\n=== LRMP lin (FV ref=Z65536 + DG) ===")
        radialDG.radialDG_tests(
            n_jobs=args.njobs,
            small_test=False,
            output_path=args.output,
            cadet_path=args.cadet,
            studies=[2],
            study2_configs=[3],
            fv_n_disc=14,  # 8*2^13 = Z65536 as finest FV ref
            study2_skip_fv=args.skip_fv,
        )

    if args.config in ('sma', 'both'):
        print("\n=== LRMP SMA (FV ref=Z16384 + DG) ===")
        _fv_start = args.fv_start if args.fv_start else 8
        # Compute fv_n_disc so finest = 16384
        import math
        _fv_n_disc = int(math.log2(16384 / _fv_start)) + 1
        radialDG.radialDG_tests(
            n_jobs=args.njobs,
            small_test=False,
            output_path=args.output,
            cadet_path=args.cadet,
            studies=[2],
            study2_configs=[4],
            fv_n_disc=_fv_n_disc,
            fv_start_ncol=_fv_start,
            study2_skip_fv=args.skip_fv,
            study2_polydegs=args.polydegs,
            dg_n_disc=args.dg_n_disc,
        )
