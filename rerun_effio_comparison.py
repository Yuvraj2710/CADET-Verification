#!/usr/bin/env python3
"""
Rerun the two finest DG P5 membrane simulations with Effio's exact
geometry (r_a = 11.25 mm) and regenerate effio_comparison.png.

Does NOT touch the convergence study H5 files.
"""

import os
import sys
import copy
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.benchmark_models.setting_radCol1D_DG_membrane_transport_1comp as setting_transport
import src.benchmark_models.setting_radCol1D_DG_membrane_LRMP_1comp as setting_LRMP

from cadet import Cadet

CADET_PATH = '/Users/yuvj/CADET/build/src/cadet-cli/cadet-cli'
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'radialDG-results-membrane')

R_A_EFFIO = 11.25e-3  # Effio's reported outer radius [m]


def run_single(setting_module, prefix, polyDeg, nElem):
    """Run a single DG simulation with corrected r_a."""
    config = setting_module.get_model()

    # Patch r_a to Effio's value
    unit = config['input']['model']['unit_001']
    unit['col_radius_outer'] = R_A_EFFIO

    # DG discretization
    disc = unit['discretization']
    disc['SPATIAL_METHOD'] = 'DG'
    disc['POLYDEG'] = polyDeg
    disc['NELEM'] = nElem
    unit['POLYNOMIAL_INTERPOLATION_NODES'] = 'CGL'

    # For LRMP, need the correct unit type
    if unit.get('npartype', 0) > 0:
        unit['unit_type'] = 'RADIAL_LUMPED_RATE_MODEL_WITH_PORES'

    fname = os.path.join(OUTPUT_DIR, f'{prefix}_effio_geom_P{polyDeg}Z{nElem}.h5')
    model = Cadet(install_path=CADET_PATH)
    model.root.input = config['input']
    model.filename = fname
    model.save()
    model.run()
    model.load()

    times = model.root.output.solution.solution_times
    outlet = model.root.output.solution.unit_001.solution_outlet
    if outlet.ndim == 2:
        outlet = outlet[:, 0]

    print(f"  {fname}: peak={outlet.max():.6e} at t={times[np.argmax(outlet)]:.2f}s")
    return times, outlet


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    F_mL_per_s = 0.05  # 3 mL/min

    # ---- Load existing simulations (already run with Effio's r_a) ----
    fname_a = os.path.join(OUTPUT_DIR, 'radMembrane_transport_1comp_effio_geom_P5Z512.h5')
    fname_b = os.path.join(OUTPUT_DIR, 'radMembrane_LRMP_1comp_effio_geom_P5Z256.h5')

    if not os.path.exists(fname_a) or not os.path.exists(fname_b):
        print("H5 files not found, running simulations...")
        t_a, sol_a = run_single(setting_transport, 'radMembrane_transport_1comp', 5, 512)
        t_b, sol_b = run_single(setting_LRMP, 'radMembrane_LRMP_1comp', 5, 256)
    else:
        print("Loading existing simulations with r_a = 11.25 mm...")
        with h5py.File(fname_a, 'r') as f:
            t_a = f['output']['solution']['SOLUTION_TIMES'][:]
            sol_a = f['output']['solution']['unit_001']['SOLUTION_OUTLET'][:, 0]
        with h5py.File(fname_b, 'r') as f:
            t_b = f['output']['solution']['SOLUTION_TIMES'][:]
            sol_b = f['output']['solution']['unit_001']['SOLUTION_OUTLET'][:, 0]
        print(f"  Config A: peak={sol_a.max():.6e} at t={t_a[np.argmax(sol_a)]:.2f}s")
        print(f"  Config B: peak={sol_b.max():.6e} at t={t_b[np.argmax(sol_b)]:.2f}s")

    # ---- Load WPD-digitized Effio et al. Fig. 3 data ----
    def load_wpd_csv(filename):
        data = np.loadtxt(os.path.join(OUTPUT_DIR, filename), delimiter=',')
        idx = np.argsort(data[:, 0])
        return data[idx, 0], data[idx, 1]

    effio_dex_sim_V, effio_dex_sim_y = load_wpd_csv('effio_fig3a_sim_wpd.csv')
    effio_dex_meas_V, effio_dex_meas_y = load_wpd_csv('effio_fig3a_meas_wpd.csv')
    effio_nacl_sim_V, effio_nacl_sim_y = load_wpd_csv('effio_fig3b_sim_wpd.csv')
    effio_nacl_meas_V, effio_nacl_meas_y = load_wpd_csv('effio_fig3b_meas_wpd.csv')

    # Normalize all to peak = 1
    effio_dex_meas_n = effio_dex_meas_y / effio_dex_meas_y.max()
    effio_dex_sim_n = effio_dex_sim_y / effio_dex_sim_y.max()
    effio_nacl_meas_n = effio_nacl_meas_y / effio_nacl_meas_y.max()
    effio_nacl_sim_n = effio_nacl_sim_y / effio_nacl_sim_y.max()

    V_a = t_a * F_mL_per_s
    sol_a_n = sol_a / sol_a.max()
    V_b = t_b * F_mL_per_s
    sol_b_n = sol_b / sol_b.max()

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(V_a, sol_a_n, '-', color='#d62728', linewidth=1.8, label=r'DG $N_d = 5$', zorder=3)
    ax1.plot(effio_dex_sim_V, effio_dex_sim_n, '-', color='#006400',
             linewidth=1.5, label='Effio et al. (sim.)', zorder=2)
    ax1.plot(effio_dex_meas_V, effio_dex_meas_n, '--', color='#006400',
             linewidth=1.5, label='Effio et al. (exp.)', zorder=4)
    ax1.set_xlabel('Volume [mL]', fontsize=14)
    ax1.set_ylabel('Normalized outlet signal', fontsize=14)
    ax1.set_title('')
    ax1.set_xlim(0, 3.5)
    ax1.tick_params(labelsize=12)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, which='major', alpha=0.3)

    ax2.plot(V_b, sol_b_n, '-', color='#d62728', linewidth=1.8, label=r'DG $N_d = 5$', zorder=3)
    ax2.plot(effio_nacl_sim_V, effio_nacl_sim_n, '-', color='#006400',
             linewidth=1.5, label='Effio et al. (sim.)', zorder=2)
    ax2.plot(effio_nacl_meas_V, effio_nacl_meas_n, '--', color='#006400',
             linewidth=1.5, label='Effio et al. (exp.)', zorder=4)
    ax2.set_xlabel('Volume [mL]', fontsize=14)
    ax2.set_ylabel('Normalized outlet signal', fontsize=14)
    ax2.set_title('')
    ax2.set_xlim(0, 6)
    ax2.tick_params(labelsize=12)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, which='major', alpha=0.3)

    out_path = os.path.join(OUTPUT_DIR, 'effio_comparison.png')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
