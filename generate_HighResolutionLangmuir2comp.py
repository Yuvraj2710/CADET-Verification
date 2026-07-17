#!/usr/bin/env python3
"""
Generate HighResolutionLangmuir2comp.png — reference outlet chromatograms
for the two-component rLRM with Langmuir binding at two dispersion levels.

Uses DG with CGL nodes at N_e = 300, N_d = 10 (matching thesis caption).
IDAS abstol = 1e-12 to ensure spatial error dominates.
"""

import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from cadet import Cadet

import src.benchmark_models.setting_radCol1D_DG_LRM_lang_2comp as setting_lang

CADET_PATH = '/home/IBT/rathore/local-fix'
OUTPUT_DIR = Path('/home/IBT/rathore/CADET-Verification')

N_ELEM = 300
POLY_DEG = 5


def setup_and_run(D0, label):
    """Configure and run a high-resolution DG simulation."""
    base = setting_lang.get_model(D0=D0)
    config = copy.deepcopy(base)

    # DG discretization
    unit = config['input']['model']['unit_001']
    # RADIAL_COLUMN_MODEL_1D is only for pure transport (npartype=0).
    # With particles/binding, must use the dedicated radial unit type.
    unit['unit_type'] = 'RADIAL_LUMPED_RATE_MODEL_WITHOUT_PORES'
    disc = unit['discretization']
    disc['SPATIAL_METHOD'] = 'DG'
    disc['POLYDEG'] = POLY_DEG
    disc['NELEM'] = N_ELEM
    unit['POLYNOMIAL_INTERPOLATION_NODES'] = 'CGL'

    # Tight time integration
    ti = config['input']['solver']['time_integrator']
    ti['abstol'] = 1e-12
    ti['reltol'] = 1e-10
    ti['init_step_size'] = 1e-6
    ti['max_steps'] = 5_000_000

    model = Cadet(install_path=CADET_PATH)
    model.root.input = config['input']
    model.filename = str(OUTPUT_DIR / f'highres_lang2comp_{label}.h5')
    model.save()

    print(f"Running {label} (D0={D0}, N_e={N_ELEM}, N_d={POLY_DEG}) ...")
    data = model.run_simulation()
    if data.return_code != 0:
        raise RuntimeError(f"Simulation failed for {label}: {data.error_message}")
    model.load()
    print(f"  {label} done.")
    return model


def main():
    cases = [
        (1e-4, 'D1e4', r'$D_0 = 10^{-4}\;\mathrm{m^2/s}$ (rLRM1)'),
        (1e-5, 'D1e5', r'$D_0 = 10^{-5}\;\mathrm{m^2/s}$ (rLRM2)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, (D0, label, title) in zip(axes, cases):
        model = setup_and_run(D0, label)

        time = model.root.output.solution.solution_times
        outlet = model.root.output.solution.unit_001.solution_outlet

        ax.plot(time, outlet[:, 0], label='Component 1', linewidth=1.2)
        ax.plot(time, outlet[:, 1], label='Component 2', linewidth=1.2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Concentration [mol/L]')
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(0, 40)

    fig.tight_layout()
    out_path = OUTPUT_DIR / 'HighResolutionLangmuir2comp.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
