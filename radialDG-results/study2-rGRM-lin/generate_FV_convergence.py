#!/usr/bin/env python3
"""Compute FV WENO3 self-convergence for rGRM lin 1comp.

Uses Z8192 as finest reference. Produces CSV and plots.
"""

import os
import csv
import numpy as np
import h5py
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'test_cadet-core', 'radialDG')

LEVELS = [128, 256, 512, 1024, 2048, 4096, 8192]
REF = 8192


def load_outlet(z):
    path = os.path.join(BASE, f'radGRM_FV_WENO3_lin_1comp_FV_Z{z}.h5')
    f = h5py.File(path, 'r')
    sol = f['output']['solution']['unit_001']['SOLUTION_OUTLET'][:]
    sim_time = f['meta']['TIME_SIM'][()]
    ncol = f['input']['model']['unit_001']['discretization']['NCOL'][()]
    f.close()
    return sol, sim_time, int(ncol)


def main():
    # Load reference
    ref_sol, _, _ = load_outlet(REF)

    rows = []
    for z in LEVELS:
        sol, sim_time, ncol = load_outlet(z)
        bulk_dof = ncol

        if z == REF:
            # No error for reference level
            rows.append({
                'Ne': ncol, 'DoF': bulk_dof,
                'L2error': 0.0, 'L2EOC': 0.0,
                'Linferror': 0.0, 'LinfEOC': 0.0,
                'Simtime': sim_time
            })
            continue

        diff = sol - ref_sol
        # L2 error: sqrt(sum of squares) over all time points and components
        l2_err = np.sqrt(np.sum(diff**2))
        # Linf error: max absolute error over all time points and components
        linf_err = np.max(np.abs(diff))

        rows.append({
            'Ne': ncol, 'DoF': bulk_dof,
            'L2error': l2_err, 'L2EOC': 0.0,
            'Linferror': linf_err, 'LinfEOC': 0.0,
            'Simtime': sim_time
        })

    # Compute EOC
    for i in range(1, len(rows)):
        if rows[i]['L2error'] > 0 and rows[i-1]['L2error'] > 0:
            rows[i]['L2EOC'] = np.log(rows[i-1]['L2error'] / rows[i]['L2error']) / np.log(2)
        if rows[i]['Linferror'] > 0 and rows[i-1]['Linferror'] > 0:
            rows[i]['LinfEOC'] = np.log(rows[i-1]['Linferror'] / rows[i]['Linferror']) / np.log(2)

    # Print table
    print(f"\nFV WENO3 self-convergence for rGRM lin 1comp (ref = Z{REF})")
    print(f"{'Ne':>6} {'DoF':>6} {'L2error':>12} {'L2EOC':>8} {'Linferror':>12} {'LinfEOC':>8} {'Simtime':>10}")
    for r in rows:
        print(f"{r['Ne']:>6} {r['DoF']:>6} {r['L2error']:>12.6e} {r['L2EOC']:>8.4f} {r['Linferror']:>12.6e} {r['LinfEOC']:>8.4f} {r['Simtime']:>10.3f}")

    # Write CSV (exclude reference row)
    csv_path = os.path.join(OUT_DIR, 'rGRM_lin_FV.csv')
    header = ['Ne', 'DoF', 'L2error', 'L2EOC', 'Linferror', 'LinfEOC', 'Simtime']
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows[:-1]:  # exclude reference
            w.writerow([
                int(r['Ne']), int(r['DoF']),
                f"{r['L2error']:.6e}", f"{r['L2EOC']:.4f}",
                f"{r['Linferror']:.6e}", f"{r['LinfEOC']:.4f}",
                f"{r['Simtime']:.3f}",
            ])
    print(f"\nSaved: {csv_path}")

    # Plots
    dofs = [r['DoF'] for r in rows[:-1]]
    linf = [r['Linferror'] for r in rows[:-1]]
    times = [r['Simtime'] for r in rows[:-1]]

    # Plot 1: Linf vs DoF
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.loglog(dofs, linf, 'ks--', markersize=5, linewidth=1.2, label='FV WENO3')
    ax.set_xlabel('Degrees of freedom', fontsize=11)
    ax.set_ylabel('$L^\\infty$ error', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which='major', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Linf_vs_DoF.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: Linf_vs_DoF.png")

    # Plot 2: Linf vs compute time
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.loglog(times, linf, 'ks--', markersize=5, linewidth=1.2, label='FV WENO3')
    ax.set_xlabel('Compute time [s]', fontsize=11)
    ax.set_ylabel('$L^\\infty$ error', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which='major', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Linf_vs_compute.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: Linf_vs_compute.png")


if __name__ == '__main__':
    main()
