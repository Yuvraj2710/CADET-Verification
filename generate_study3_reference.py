#!/usr/bin/env python3
"""Generate high-resolution reference solutions for Study 3 (radial LRM, Langmuir 2-comp).

DG P10, Ne=300, for both dispersion settings (D1e4 and D1e5).

Usage on HPC:
  python3 generate_study3_reference.py --cadet /path/to/cadet-cli --output radialDG-results/study3/h5
"""

import os
import sys
import copy
import subprocess
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.benchmark_models.setting_radCol1D_DG_LRM_lang_2comp as setting_DG_LRM_lang
from cadet import Cadet

parser = argparse.ArgumentParser()
parser.add_argument('--cadet', default=os.path.expanduser('~/local-fix'),
                    help='Path to cadet-cli or CADET install root')
parser.add_argument('--output', default='radialDG-results/study3/h5')
args = parser.parse_args()

OUTPUT_DIR = args.output
os.makedirs(OUTPUT_DIR, exist_ok=True)

POLYDEG = 10
NELEM = 300

time_integrator = {
    'ABSTOL': 1e-12, 'RELTOL': 1e-10, 'ALGTOL': 1e-10,
    'USE_MODIFIED_NEWTON': False,
    'INIT_STEP_SIZE': 1e-6,
    'MAX_STEPS': 5000000
}

for D0, label in [(1e-4, 'D1e4'), (1e-5, 'D1e5')]:
    print(f"\n=== Generating reference: {label}, DG P{POLYDEG}, Ne={NELEM} ===")

    base = setting_DG_LRM_lang.get_model(D0=D0)
    config = copy.deepcopy(dict(base))

    config['input']['solver']['time_integrator'] = time_integrator

    unit = config['input']['model']['unit_001']
    disc = unit['discretization']
    disc['SPATIAL_METHOD'] = 'DG'
    disc['POLYDEG'] = POLYDEG
    disc['NELEM'] = NELEM
    unit['POLYNOMIAL_INTERPOLATION_NODES'] = 'CGL'

    fname = f'radLRM_DG_lang_2comp_{label}_P{POLYDEG}_DG_P{POLYDEG}Z{NELEM}.h5'
    fpath = os.path.join(OUTPUT_DIR, fname)

    model = Cadet(install_path=args.cadet)
    model.root.input = config['input']
    model.filename = fpath
    model.save()

    print(f"  Running {fname} ...")
    data = model.run_simulation()
    if data.return_code == 0:
        model.load()
        t = model.root.output.solution.solution_times
        c = model.root.output.solution.unit_001.solution_outlet
        print(f"  Solution shape: t={t.shape}, c={c.shape}")
        print(f"  Saved: {fpath}")
    else:
        print(f"  FAILED! {data.error_message}")

print("\nDone.")
