#!/usr/bin/env python3
"""Run FV SMA Z16, Z32, Z64 for v7 (to fill in coarse FV levels)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.radialDG as radialDG
import src.benchmark_models.setting_radCol1D_DG_GRM_SMA_4comp as setting_GRM_SMA

NEW_VELOCITY_COEFF = 9.775e-6

_orig = setting_GRM_SMA.get_model
def patched():
    m = _orig()
    m['input']['model']['unit_001']['velocity_coeff'] = NEW_VELOCITY_COEFF
    return m
setting_GRM_SMA.get_model = patched

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/test_cadet-core/radialDG/v7')
    parser.add_argument('--cadet', default=os.path.expanduser('~/local-fix'))
    parser.add_argument('--njobs', type=int, default=-1)
    args = parser.parse_args()

    radialDG.radialDG_tests(
        n_jobs=args.njobs,
        small_test=False,
        output_path=args.output,
        cadet_path=args.cadet,
        studies=[2],
        study2_configs=[1],
        study2_skip_fv=False,
        fv_start_ncol=16,
        fv_n_disc=3,  # 16, 32, 64
        study2_skip_dg_rerun=True,  # skip DG, only FV
    )
