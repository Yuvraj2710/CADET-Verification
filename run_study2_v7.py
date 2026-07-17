#!/usr/bin/env python3
"""Run Study 2 with velocity_coeff = 9.775e-6 (residence-time matched to axial v=5.75e-4).

All other parameters (dispersion, film diffusion, etc.) stay at v2 values.
Only runs GRM configs (lin 1comp + SMA 4comp).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.radialDG as radialDG
import src.benchmark_models.setting_radCol1D_DG_GRM_lin_1comp as setting_GRM_lin
import src.benchmark_models.setting_radCol1D_DG_GRM_SMA_4comp as setting_GRM_SMA

# Monkey-patch the setting modules to override velocity_coeff
_orig_get_model_lin = setting_GRM_lin.get_model
_orig_get_model_sma = setting_GRM_SMA.get_model

NEW_VELOCITY_COEFF = 9.775e-6


def patched_get_model_lin():
    m = _orig_get_model_lin()
    m['input']['model']['unit_001']['velocity_coeff'] = NEW_VELOCITY_COEFF
    return m


def patched_get_model_sma():
    m = _orig_get_model_sma()
    m['input']['model']['unit_001']['velocity_coeff'] = NEW_VELOCITY_COEFF
    return m


setting_GRM_lin.get_model = patched_get_model_lin
setting_GRM_SMA.get_model = patched_get_model_sma

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/test_cadet-core/radialDG/v7')
    parser.add_argument('--cadet', default=os.path.expanduser('~/local-fix'))
    parser.add_argument('--njobs', type=int, default=1)
    parser.add_argument('--config', choices=['lin', 'sma', 'both'], default='both')
    parser.add_argument('--skip-fv', action='store_true')
    args = parser.parse_args()

    print(f"Study 2 v7: velocity_coeff = {NEW_VELOCITY_COEFF}")
    print(f"Output: {args.output}")
    print(f"CADET: {args.cadet}")
    print(f"Config: {args.config}")

    if args.config in ('lin', 'both'):
        # GRM lin: FV ref = Z16384 (8 levels from 128)
        print("\n=== GRM lin (FV ref=Z16384 + DG) ===")
        radialDG.radialDG_tests(
            n_jobs=args.njobs,
            small_test=False,
            output_path=args.output,
            cadet_path=args.cadet,
            studies=[2],
            study2_configs=[0],
            study2_skip_fv=args.skip_fv,
        )

    if args.config in ('sma', 'both'):
        # GRM SMA: FV ref = Z2048 (5 levels from 128)
        print("\n=== GRM SMA (FV ref=Z2048 + DG) ===")
        radialDG.radialDG_tests(
            n_jobs=args.njobs,
            small_test=False,
            output_path=args.output,
            cadet_path=args.cadet,
            studies=[2],
            study2_configs=[1],
            fv_n_disc=5,  # 128*2^4 = Z2048 as finest ref
            study2_skip_fv=args.skip_fv,
        )
