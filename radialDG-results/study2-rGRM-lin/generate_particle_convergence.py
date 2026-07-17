#!/usr/bin/env python3
"""Particle domain convergence for rGRM lin 1comp.

FV WENO3 self-convergence over particle+solid domain (Z128-Z512, ref=Z1024).
Uses piecewise-constant interpolation to map coarser FV solutions onto the
Z1024 grid before computing errors.

Memory-efficient: processes time steps in chunks to avoid loading full arrays.

Produces:
  - rGRM_lin_FV_particle.csv
  - Linf_vs_DoF_particle.png
"""

import os
import csv
import numpy as np
import h5py
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'test_cadet-core', 'radialDG')

FV_LEVELS = [128, 256, 512, 1024]
FV_REF = 1024

NCOMP = 1
SUM_NBOUND = 1
CHUNK_SIZE = 50  # time steps per chunk


def open_fv(z):
    """Open FV h5 file and return handle + metadata."""
    path = os.path.join(BASE, f'radGRM_FV_WENO3_lin_1comp_FV_Z{z}.h5')
    f = h5py.File(path, 'r')
    sim_time = f['meta']['TIME_SIM'][()]
    ncol = int(f['input']['model']['unit_001']['discretization']['NCOL'][()])
    ncells = int(f['input']['model']['unit_001']['particle_type_000']['discretization']['NCELLS'][()])
    total_dof = ncol * (NCOMP + ncells * (NCOMP + SUM_NBOUND))
    return f, sim_time, ncol, ncells, total_dof


def compute_errors_chunked(f_coarse, f_ref, ncol_coarse, ncol_ref, field):
    """Compute L2 and Linf errors using chunked time processing.

    Interpolates coarser FV to reference grid via piecewise constant extension.
    """
    ds_coarse = f_coarse['output']['solution']['unit_001'][field]
    ds_ref = f_ref['output']['solution']['unit_001'][field]
    ntime = ds_ref.shape[0]

    ratio_ax = ncol_ref // ncol_coarse
    ncells_coarse = ncol_coarse // 4
    ncells_ref = ncol_ref // 4
    ratio_par = ncells_ref // ncells_coarse

    l2_sum = 0.0
    linf_max = 0.0

    for t0 in range(0, ntime, CHUNK_SIZE):
        t1 = min(t0 + CHUNK_SIZE, ntime)
        chunk_coarse = ds_coarse[t0:t1]
        chunk_ref = ds_ref[t0:t1]

        # Piecewise constant interpolation
        chunk_interp = np.repeat(chunk_coarse, ratio_ax, axis=1)
        chunk_interp = np.repeat(chunk_interp, ratio_par, axis=2)

        diff = chunk_interp - chunk_ref
        l2_sum += np.sum(diff**2)
        linf_max = max(linf_max, np.max(np.abs(diff)))

        del chunk_coarse, chunk_ref, chunk_interp, diff

    return np.sqrt(l2_sum), linf_max


def main():
    # Open reference
    f_ref, _, ref_ncol, ref_ncells, _ = open_fv(FV_REF)
    print(f"Reference: FV Z{FV_REF}")

    rows_outlet = []
    rows_particle = []
    rows_solid = []
    rows_combined = []

    for z in FV_LEVELS:
        f_coarse, sim_time, ncol, ncells, total_dof = open_fv(z)
        print(f"Processing FV Z{z}: ncol={ncol}, ncells={ncells}, DoF={total_dof}")

        if z == FV_REF:
            for rows in [rows_outlet, rows_particle, rows_solid, rows_combined]:
                rows.append({'Ne': ncol, 'Np': ncells, 'DoF': total_dof,
                             'L2error': 0.0, 'L2EOC': 0.0,
                             'Linferror': 0.0, 'LinfEOC': 0.0,
                             'Simtime': sim_time})
            f_coarse.close()
            continue

        # Outlet errors (small, load directly)
        sol_out = f_coarse['output']['solution']['unit_001']['SOLUTION_OUTLET'][:]
        ref_out = f_ref['output']['solution']['unit_001']['SOLUTION_OUTLET'][:]
        diff = sol_out - ref_out
        l2_out = np.sqrt(np.sum(diff**2))
        linf_out = np.max(np.abs(diff))
        rows_outlet.append({'Ne': ncol, 'Np': ncells, 'DoF': total_dof,
                            'L2error': l2_out, 'L2EOC': 0.0,
                            'Linferror': linf_out, 'LinfEOC': 0.0,
                            'Simtime': sim_time})

        # Particle domain errors (chunked)
        l2_par, linf_par = compute_errors_chunked(f_coarse, f_ref, ncol, ref_ncol, 'SOLUTION_PARTICLE')
        rows_particle.append({'Ne': ncol, 'Np': ncells, 'DoF': total_dof,
                              'L2error': l2_par, 'L2EOC': 0.0,
                              'Linferror': linf_par, 'LinfEOC': 0.0,
                              'Simtime': sim_time})
        print(f"  Particle: L2={l2_par:.6e}, Linf={linf_par:.6e}")

        # Solid domain errors (chunked)
        l2_sol, linf_sol = compute_errors_chunked(f_coarse, f_ref, ncol, ref_ncol, 'SOLUTION_SOLID')
        rows_solid.append({'Ne': ncol, 'Np': ncells, 'DoF': total_dof,
                           'L2error': l2_sol, 'L2EOC': 0.0,
                           'Linferror': linf_sol, 'LinfEOC': 0.0,
                           'Simtime': sim_time})
        print(f"  Solid:    L2={l2_sol:.6e}, Linf={linf_sol:.6e}")

        # Combined: max of Linf, sqrt(sum of L2^2)
        linf_comb = max(linf_par, linf_sol)
        l2_comb = np.sqrt(l2_par**2 + l2_sol**2)
        rows_combined.append({'Ne': ncol, 'Np': ncells, 'DoF': total_dof,
                              'L2error': l2_comb, 'L2EOC': 0.0,
                              'Linferror': linf_comb, 'LinfEOC': 0.0,
                              'Simtime': sim_time})

        f_coarse.close()

    f_ref.close()

    # Compute EOC for all
    for rows in [rows_outlet, rows_particle, rows_solid, rows_combined]:
        for i in range(1, len(rows)):
            if rows[i]['L2error'] > 0 and rows[i-1]['L2error'] > 0:
                rows[i]['L2EOC'] = np.log(rows[i-1]['L2error'] / rows[i]['L2error']) / np.log(2)
            if rows[i]['Linferror'] > 0 and rows[i-1]['Linferror'] > 0:
                rows[i]['LinfEOC'] = np.log(rows[i-1]['Linferror'] / rows[i]['Linferror']) / np.log(2)

    # --- Print tables ---
    for name, rows in [('Outlet', rows_outlet), ('Particle (liquid)', rows_particle),
                       ('Solid (bound)', rows_solid), ('Combined (par+sol)', rows_combined)]:
        print(f"\nFV WENO3 self-convergence — {name} (ref = Z{FV_REF})")
        print(f"{'Ne':>6} {'Np':>6} {'DoF':>10} {'L2error':>12} {'L2EOC':>8} {'Linferror':>12} {'LinfEOC':>8} {'Simtime':>10}")
        for r in rows:
            print(f"{r['Ne']:>6} {r['Np']:>6} {r['DoF']:>10} {r['L2error']:>12.6e} {r['L2EOC']:>8.4f} {r['Linferror']:>12.6e} {r['LinfEOC']:>8.4f} {r['Simtime']:>10.3f}")

    # --- Write CSV ---
    header = ['Ne', 'Np', 'DoF',
              'L2error_outlet', 'L2EOC_outlet', 'Linferror_outlet', 'LinfEOC_outlet',
              'L2error_particle', 'L2EOC_particle', 'Linferror_particle', 'LinfEOC_particle',
              'L2error_solid', 'L2EOC_solid', 'Linferror_solid', 'LinfEOC_solid',
              'L2error_combined', 'L2EOC_combined', 'Linferror_combined', 'LinfEOC_combined',
              'Simtime']
    csv_path = os.path.join(OUT_DIR, 'rGRM_lin_FV_particle.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(rows_outlet) - 1):  # exclude reference
            ro, rp, rs, rc = rows_outlet[i], rows_particle[i], rows_solid[i], rows_combined[i]
            w.writerow([
                int(ro['Ne']), int(ro['Np']), int(ro['DoF']),
                f"{ro['L2error']:.6e}", f"{ro['L2EOC']:.4f}",
                f"{ro['Linferror']:.6e}", f"{ro['LinfEOC']:.4f}",
                f"{rp['L2error']:.6e}", f"{rp['L2EOC']:.4f}",
                f"{rp['Linferror']:.6e}", f"{rp['LinfEOC']:.4f}",
                f"{rs['L2error']:.6e}", f"{rs['L2EOC']:.4f}",
                f"{rs['Linferror']:.6e}", f"{rs['LinfEOC']:.4f}",
                f"{rc['L2error']:.6e}", f"{rc['L2EOC']:.4f}",
                f"{rc['Linferror']:.6e}", f"{rc['LinfEOC']:.4f}",
                f"{ro['Simtime']:.3f}",
            ])
    print(f"\nSaved: {csv_path}")

    # --- Plot: Linf vs DoF for all domains ---
    fig, ax = plt.subplots(figsize=(7, 5.5))

    labels = [('Outlet', rows_outlet, 'ks--'),
              ('Particle (liquid)', rows_particle, 'bo-'),
              ('Solid (bound)', rows_solid, 'r^-'),
              ('Combined', rows_combined, 'gD-')]

    for label, rows, fmt in labels:
        dofs = [r['DoF'] for r in rows[:-1]]
        errs = [r['Linferror'] for r in rows[:-1]]
        ax.loglog(dofs, errs, fmt, markersize=6, linewidth=1.2,
                  markerfacecolor='white', markeredgewidth=1.5,
                  label=label, zorder=3)

    ax.set_xlabel('Degrees of freedom', fontsize=11)
    ax.set_ylabel('$L^\\infty$ error', fontsize=11)
    ax.set_title(f'FV WENO3 self-convergence (ref Z{FV_REF})', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='major', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Linf_vs_DoF_particle.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: Linf_vs_DoF_particle.png")

    print("\nDone!")


if __name__ == '__main__':
    main()
