#!/usr/bin/env python3
"""
Runner for Study 1 GRM SMA with corrected velocity (residence-time matched).

velocity_coeff = 9.775e-6 (equivalent to axial v=5.75e-4).
FV reference = Z2048 from study2 v7 (must already exist in output dir).

Usage:
    python run_study1_GRM_SMA_v7.py [--node-type CGL|LGL] [--polydeg N] [--n-repeats N] [--small]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.benchmark_models.setting_radCol1D_DG_GRM_SMA_4comp as setting_GRM_SMA

# Monkey-patch velocity_coeff to residence-time matched value (same as study2 v7)
NEW_VELOCITY_COEFF = 9.775e-6

_orig_get_model = setting_GRM_SMA.get_model


def patched_get_model():
    m = _orig_get_model()
    m['input']['model']['unit_001']['velocity_coeff'] = NEW_VELOCITY_COEFF
    return m


setting_GRM_SMA.get_model = patched_get_model

from src.study1_cgl_vs_lgl_GRM_SMA import study1_GRM_SMA_tests


def main():
    parser = argparse.ArgumentParser(description='Study 1 GRM SMA v7: corrected velocity')
    parser.add_argument('--node-type', type=str, default=None,
                        help='Node type: CGL or LGL (default: both)')
    parser.add_argument('--polydeg', type=int, default=None,
                        help='Polynomial degree (default: 1-5)')
    parser.add_argument('--n-repeats', type=int, default=3,
                        help='Number of reruns for min wall time (default: 3)')
    parser.add_argument('--small', action='store_true',
                        help='Use reduced discretization levels')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--output-path', type=str,
                        default='output/test_cadet-core/radialDG/v7',
                        help='Output directory')
    parser.add_argument('--cadet-path', type=str,
                        default=os.path.expanduser('~/local-fix'),
                        help='Path to CADET install')
    args = parser.parse_args()

    node_types = [args.node_type] if args.node_type else None
    polydegs = [args.polydeg] if args.polydeg else None

    print(f"=== Study 1 GRM SMA v7 (velocity_coeff={NEW_VELOCITY_COEFF}): "
          f"{args.node_type or 'CGL+LGL'} P{args.polydeg or '1-5'} ===")
    print(f"  FV reference: Z2048")
    print(f"  n_repeats: {args.n_repeats}")
    print(f"  n_jobs: {args.n_jobs}")
    print(f"  output: {args.output_path}")
    print(f"  cadet: {args.cadet_path}")
    print()

    study1_GRM_SMA_tests(
        n_jobs=args.n_jobs,
        small_test=args.small,
        output_path=args.output_path,
        cadet_path=args.cadet_path,
        polydegs=polydegs,
        node_types=node_types,
        n_repeats=args.n_repeats,
        fv_ncol=2048,
    )

    print(f"\n=== Done: {args.node_type or 'CGL+LGL'} "
          f"P{args.polydeg or '1-5'} ===")


if __name__ == '__main__':
    main()
