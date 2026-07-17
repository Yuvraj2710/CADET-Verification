#!/usr/bin/env python3
"""
Compare CADET DG and FV solutions against Effio et al. (2016) Fig. 3
experimental data.

Effio et al., J. Chromatogr. A 1429 (2016) 142-154.
  Fig. 3a: Dextran T2000 pulse (UV 215 nm)
  Fig. 3b: NaCl pulse (conductivity)

CADET time -> volume: V [mL] = t [s] * F, F = 0.05 mL/s (3 mL/min).
All curves normalized to peak = 1 for shape comparison.

Notes on peak position mismatch:
  - CADET geometric volume = pi*(r_out^2 - r_in^2)*H = 2.915 mL,
    ~3% less than manufacturer VM = 3 mL (spiral-wound geometry).
  - Effio subtracted system + capsule void volumes (~1.3 mL) from
    their chromatograms, but capsule void adds extra dispersion
    that CADET does not model -> experiment peaks are broader/later.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import h5py
except ImportError:
    print("ERROR: h5py required. Install with: pip install h5py")
    sys.exit(1)

H5_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'output', 'test_cadet-core', 'radialDG', 'membrane')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'radialDG-results-membrane')

F_mL_per_s = 0.05  # 3 mL/min = 0.05 mL/s


def read_outlet(h5_path):
    with h5py.File(h5_path, 'r') as f:
        times = f['output']['solution']['SOLUTION_TIMES'][:]
        outlet = f['output']['solution']['unit_001']['SOLUTION_OUTLET'][:]
    if outlet.ndim == 2:
        outlet = outlet[:, 0]
    return times, outlet


# =========================================================================
# Hand-digitized experimental data from Effio et al. Fig. 3
# Volume [mL] vs signal (UV or conductivity), void-subtracted
# =========================================================================

effio_dex_meas_V = np.array([
    0.0, 0.5, 0.8, 1.0, 1.1, 1.2, 1.3, 1.35, 1.4, 1.45,
    1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95,
    2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8, 3.0, 3.5])
effio_dex_meas_y = np.array([
    0.0, 0.0, 0.0, 1.0, 3.0, 7.0, 12.0, 15.0, 18.0, 21.0,
    24.0, 26.5, 28.5, 29.5, 30.0, 29.0, 27.0, 24.5, 21.5, 18.5,
    15.5, 10.5, 7.0, 4.5, 2.8, 1.5, 0.8, 0.2, 0.0, 0.0])

effio_dex_sim_V = np.array([
    0.0, 0.5, 0.8, 1.0, 1.1, 1.2, 1.3, 1.35, 1.4, 1.45,
    1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95,
    2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8, 3.0, 3.5])
effio_dex_sim_y = np.array([
    0.0, 0.0, 0.0, 1.5, 4.0, 8.5, 14.0, 17.0, 20.0, 23.0,
    25.5, 27.5, 29.0, 30.0, 30.0, 29.0, 27.0, 24.0, 20.0, 16.5,
    13.5, 8.5, 5.0, 2.8, 1.5, 0.7, 0.3, 0.0, 0.0, 0.0])

effio_nacl_meas_V = np.array([
    0.0, 0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.2,
    2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3.0, 3.2, 3.4, 3.6,
    3.8, 4.0, 4.2, 4.5, 5.0, 5.5, 6.0])
effio_nacl_meas_y = np.array([
    0.0, 0.0, 0.0, 0.1, 0.3, 0.8, 1.8, 3.0, 3.6, 4.2,
    4.6, 4.85, 4.85, 4.6, 4.3, 3.9, 3.2, 2.5, 1.9, 1.4,
    1.0, 0.7, 0.5, 0.3, 0.1, 0.0, 0.0])

effio_nacl_sim_V = np.array([
    0.0, 0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.2,
    2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3.0, 3.2, 3.4, 3.6,
    3.8, 4.0, 4.5, 5.0, 6.0])
effio_nacl_sim_y = np.array([
    0.0, 0.0, 0.0, 0.1, 0.5, 1.2, 2.5, 3.8, 4.3, 4.7,
    5.0, 5.0, 4.7, 4.2, 3.7, 3.1, 2.2, 1.4, 0.9, 0.5,
    0.3, 0.1, 0.0, 0.0, 0.0])

# Normalize to peak = 1
effio_dex_meas_n = effio_dex_meas_y / effio_dex_meas_y.max()
effio_dex_sim_n = effio_dex_sim_y / effio_dex_sim_y.max()
effio_nacl_meas_n = effio_nacl_meas_y / effio_nacl_meas_y.max()
effio_nacl_sim_n = effio_nacl_sim_y / effio_nacl_sim_y.max()


# =========================================================================
# Load CADET solutions
# =========================================================================

# Finest DG: P5 Z512 (transport), P5 Z256 (LRMP)
t_dg_a, sol_dg_a = read_outlet(os.path.join(H5_DIR, 'radMembrane_transport_1comp_P5_DG_P5Z512.h5'))
t_dg_b, sol_dg_b = read_outlet(os.path.join(H5_DIR, 'radMembrane_LRMP_1comp_P5_DG_P5Z256.h5'))

# FV WENO3: finest converged runs
t_fv_a, sol_fv_a = read_outlet(os.path.join(H5_DIR, 'radMembrane_transport_1comp_FV_Z16384.h5'))
t_fv_b, sol_fv_b = read_outlet(os.path.join(H5_DIR, 'radMembrane_LRMP_1comp_FV_Z16384.h5'))

# Convert CADET time [s] -> volume [mL]
V_dg_a = t_dg_a * F_mL_per_s
V_dg_b = t_dg_b * F_mL_per_s
V_fv_a = t_fv_a * F_mL_per_s
V_fv_b = t_fv_b * F_mL_per_s

# Normalize CADET solutions
sol_dg_a_n = sol_dg_a / sol_dg_a.max()
sol_dg_b_n = sol_dg_b / sol_dg_b.max()
sol_fv_a_n = sol_fv_a / sol_fv_a.max()
sol_fv_b_n = sol_fv_b / sol_fv_b.max()

# Print peak info
for label, V, sol_n in [
    ('DG Dextran', V_dg_a, sol_dg_a_n),
    ('DG NaCl', V_dg_b, sol_dg_b_n),
    ('FV Dextran', V_fv_a, sol_fv_a_n),
    ('FV NaCl', V_fv_b, sol_fv_b_n)]:
    i = np.argmax(sol_n)
    print(f'{label}: peak at V = {V[i]:.3f} mL (t = {V[i]/F_mL_per_s:.1f} s)')
print(f'Effio exp dextran peak: ~1.7 mL')
print(f'Effio exp NaCl peak:    ~2.4 mL')


# =========================================================================
# Thesis figure: 1x2 — CADET DG vs Effio sim + exp
# =========================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# (a) Dextran T2000
ax1.plot(effio_dex_meas_V, effio_dex_meas_n, 'o', color='#d62728',
         markersize=5, markerfacecolor='none', markeredgewidth=1.2,
         label='Effio et al. (exp.)', zorder=4)
ax1.plot(effio_dex_sim_V, effio_dex_sim_n, '--', color='#2ca02c',
         linewidth=1.5, label='Effio et al. (sim.)', zorder=2)
ax1.plot(V_dg_a, sol_dg_a_n, 'k-', linewidth=1.8, label='CADET DG P5', zorder=3)
ax1.plot(V_fv_a, sol_fv_a_n, '-', color='#1f77b4', linewidth=1.5,
         label='CADET FV', zorder=3)
ax1.set_xlabel('Volume [mL]', fontsize=14)
ax1.set_ylabel('Normalized outlet signal', fontsize=14)
ax1.set_xlim(0, 4)
ax1.set_ylim(-0.05, 1.1)
ax1.tick_params(labelsize=12)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, which='major', alpha=0.3)

# (b) NaCl
ax2.plot(effio_nacl_meas_V, effio_nacl_meas_n, 'o', color='#d62728',
         markersize=5, markerfacecolor='none', markeredgewidth=1.2,
         label='Effio et al. (exp.)', zorder=4)
ax2.plot(effio_nacl_sim_V, effio_nacl_sim_n, '--', color='#2ca02c',
         linewidth=1.5, label='Effio et al. (sim.)', zorder=2)
ax2.plot(V_dg_b, sol_dg_b_n, 'k-', linewidth=1.8, label='CADET DG P5', zorder=3)
ax2.plot(V_fv_b, sol_fv_b_n, '-', color='#1f77b4', linewidth=1.5,
         label='CADET FV', zorder=3)
ax2.set_xlabel('Volume [mL]', fontsize=14)
ax2.set_ylabel('Normalized outlet signal', fontsize=14)
ax2.set_xlim(0, 6)
ax2.set_ylim(-0.05, 1.1)
ax2.tick_params(labelsize=12)
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(True, which='major', alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'effio_DG_vs_experiment.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_path}")
