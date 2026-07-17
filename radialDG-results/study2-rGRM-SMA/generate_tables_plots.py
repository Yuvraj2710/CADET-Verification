#!/usr/bin/env python3
"""Generate CSV tables and convergence plots for rGRM SMA 4comp (Study 2, Config 1).

FV WENO3 self-convergence (Z128–Z2048, ref=Z2048)
DG P1–P4 convergence against FV Z2048 reference

Note: P4 Z64 and all Z128 sims are missing (crashed on HPC).

Produces:
  - rGRM_SMA_FV.csv, rGRM_SMA_DG.csv
  - Linf_vs_DoF.png, Linf_vs_compute.png
"""

import os
import csv
import numpy as np
import h5py
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'test_cadet-core', 'radialDG')

FV_LEVELS = [128, 256, 512, 1024, 2048]
FV_REF = 2048
DG_POLY_DEGS = [1, 2, 3, 4]
DG_NELEM = [4, 8, 16, 32, 64]

# GRM SMA 4comp: nComp=4, nBound=[1,1,1,1]
NCOMP = 4
SUM_NBOUND = 4


def load_fv_outlet(z):
    path = os.path.join(BASE, f'radGRM_FV_WENO3_SMA_4comp_FV_Z{z}.h5')
    f = h5py.File(path, 'r')
    sol = f['output']['solution']['unit_001']['SOLUTION_OUTLET'][:]
    sim_time = f['meta']['TIME_SIM'][()]
    ncol = int(f['input']['model']['unit_001']['discretization']['NCOL'][()])
    ncells = int(f['input']['model']['unit_001']['particle_type_000']['discretization']['NCELLS'][()])
    f.close()
    bulk_dof = ncol * NCOMP
    total_dof = ncol * (NCOMP + ncells * (NCOMP + SUM_NBOUND))
    return sol, sim_time, ncol, ncells, bulk_dof, total_dof


def load_dg_outlet(p, z):
    path = os.path.join(BASE, f'radGRM_DG_SMA_4comp_P{p}_DG_P{p}Z{z}.h5')
    if not os.path.exists(path):
        return None
    f = h5py.File(path, 'r')
    if 'output' not in f:
        f.close()
        return None
    sol = f['output']['solution']['unit_001']['SOLUTION_OUTLET'][:]
    sim_time = f['meta']['TIME_SIM'][()]
    nelem = int(f['input']['model']['unit_001']['discretization']['NELEM'][()])
    polydeg = int(f['input']['model']['unit_001']['discretization']['POLYDEG'][()])
    par_nelem = int(f['input']['model']['unit_001']['particle_type_000']['discretization']['PAR_NELEM'][()])
    par_polydeg = int(f['input']['model']['unit_001']['particle_type_000']['discretization']['PAR_POLYDEG'][()])
    f.close()
    bulk_dof = nelem * (polydeg + 1) * NCOMP
    total_dof = nelem * ((polydeg + 1) * NCOMP + par_nelem * (par_polydeg + 1) * (NCOMP + SUM_NBOUND))
    return sol, sim_time, nelem, par_nelem, bulk_dof, total_dof


def compute_errors(sol, ref_sol):
    diff = sol - ref_sol
    l2_err = np.sqrt(np.sum(diff**2))
    linf_err = np.max(np.abs(diff))
    return l2_err, linf_err


def main():
    # Load FV reference
    ref_sol, _, _, _, _, _ = load_fv_outlet(FV_REF)

    # --- FV self-convergence ---
    fv_rows = []
    for z in FV_LEVELS:
        sol, sim_time, ncol, ncells, bulk_dof, total_dof = load_fv_outlet(z)
        if z == FV_REF:
            fv_rows.append({'Ne': ncol, 'Np': ncells, 'DoF': total_dof,
                            'L2error': 0.0, 'L2EOC': 0.0,
                            'Linferror': 0.0, 'LinfEOC': 0.0,
                            'Simtime': sim_time})
            continue
        l2_err, linf_err = compute_errors(sol, ref_sol)
        fv_rows.append({'Ne': ncol, 'Np': ncells, 'DoF': total_dof,
                        'L2error': l2_err, 'L2EOC': 0.0,
                        'Linferror': linf_err, 'LinfEOC': 0.0,
                        'Simtime': sim_time})

    for i in range(1, len(fv_rows)):
        if fv_rows[i]['L2error'] > 0 and fv_rows[i-1]['L2error'] > 0:
            fv_rows[i]['L2EOC'] = np.log(fv_rows[i-1]['L2error'] / fv_rows[i]['L2error']) / np.log(2)
        if fv_rows[i]['Linferror'] > 0 and fv_rows[i-1]['Linferror'] > 0:
            fv_rows[i]['LinfEOC'] = np.log(fv_rows[i-1]['Linferror'] / fv_rows[i]['Linferror']) / np.log(2)

    # Write FV CSV (exclude reference)
    header = ['Ne', 'Np', 'DoF', 'L2error', 'L2EOC', 'Linferror', 'LinfEOC', 'Simtime']
    fv_csv = os.path.join(OUT_DIR, 'rGRM_SMA_FV.csv')
    with open(fv_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in fv_rows[:-1]:
            w.writerow([
                int(r['Ne']), int(r['Np']), int(r['DoF']),
                f"{r['L2error']:.6e}", f"{r['L2EOC']:.4f}",
                f"{r['Linferror']:.6e}", f"{r['LinfEOC']:.4f}",
                f"{r['Simtime']:.3f}",
            ])
    print(f"Saved: {fv_csv}")

    # --- DG convergence against FV reference ---
    dg_rows = {}
    for p in DG_POLY_DEGS:
        dg_rows[p] = []
        for ne in DG_NELEM:
            result = load_dg_outlet(p, ne)
            if result is None:
                print(f"  Skipping DG P{p} Z{ne} (no output)")
                continue
            sol, sim_time, nelem, par_nelem, bulk_dof, total_dof = result
            l2_err, linf_err = compute_errors(sol, ref_sol)
            dg_rows[p].append({
                'Method': f'DG P{p}', 'Ne': ne, 'Np': par_nelem,
                'DoF': total_dof,
                'L2error': l2_err, 'L2EOC': 0.0,
                'Linferror': linf_err, 'LinfEOC': 0.0,
                'Simtime': sim_time
            })
        # Compute EOC
        for i in range(1, len(dg_rows[p])):
            if dg_rows[p][i]['L2error'] > 0 and dg_rows[p][i-1]['L2error'] > 0:
                dg_rows[p][i]['L2EOC'] = np.log(dg_rows[p][i-1]['L2error'] / dg_rows[p][i]['L2error']) / np.log(2)
            if dg_rows[p][i]['Linferror'] > 0 and dg_rows[p][i-1]['Linferror'] > 0:
                dg_rows[p][i]['LinfEOC'] = np.log(dg_rows[p][i-1]['Linferror'] / dg_rows[p][i]['Linferror']) / np.log(2)

    # Write DG CSV
    dg_header = ['Method', 'Ne', 'Np', 'DoF', 'L2error', 'L2EOC', 'Linferror', 'LinfEOC', 'Simtime']
    dg_csv = os.path.join(OUT_DIR, 'rGRM_SMA_DG.csv')
    with open(dg_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(dg_header)
        for p in DG_POLY_DEGS:
            for r in dg_rows[p]:
                w.writerow([
                    r['Method'], int(r['Ne']), int(r['Np']),
                    int(r['DoF']),
                    f"{r['L2error']:.6e}", f"{r['L2EOC']:.4f}",
                    f"{r['Linferror']:.6e}", f"{r['LinfEOC']:.4f}",
                    f"{r['Simtime']:.3f}",
                ])
    print(f"Saved: {dg_csv}")

    # --- Print tables ---
    print(f"\nFV WENO3 self-convergence (ref = Z{FV_REF})")
    print(f"{'Ne':>6} {'Np':>6} {'DoF':>10} {'L2error':>12} {'L2EOC':>8} {'Linferror':>12} {'LinfEOC':>8} {'Simtime':>10}")
    for r in fv_rows:
        print(f"{r['Ne']:>6} {r['Np']:>6} {r['DoF']:>10} {r['L2error']:>12.6e} {r['L2EOC']:>8.4f} {r['Linferror']:>12.6e} {r['LinfEOC']:>8.4f} {r['Simtime']:>10.3f}")

    for p in DG_POLY_DEGS:
        print(f"\nDG P{p} convergence vs FV Z{FV_REF}")
        print(f"{'Ne':>6} {'Np':>6} {'DoF':>10} {'L2error':>12} {'L2EOC':>8} {'Linferror':>12} {'LinfEOC':>8} {'Simtime':>10}")
        for r in dg_rows[p]:
            print(f"{r['Ne']:>6} {r['Np']:>6} {r['DoF']:>10} {r['L2error']:>12.6e} {r['L2EOC']:>8.4f} {r['Linferror']:>12.6e} {r['LinfEOC']:>8.4f} {r['Simtime']:>10.3f}")

    # --- Plots ---
    colors_dg = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728'}

    def get_valid(rows, poly_deg):
        """Show all points."""
        return np.ones(len(rows), dtype=bool)

    # Plot 1: Linf vs DoF
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # FV (exclude reference row)
    fv_dofs = [r['DoF'] for r in fv_rows[:-1]]
    fv_linf = [r['Linferror'] for r in fv_rows[:-1]]
    ax.loglog(fv_dofs, fv_linf, 'ks--', markersize=5, linewidth=1.2,
              label='FV WENO3', zorder=2)

    # DG
    for p in DG_POLY_DEGS:
        if not dg_rows[p]:
            continue
        dofs = np.array([r['DoF'] for r in dg_rows[p]])
        errs = np.array([r['Linferror'] for r in dg_rows[p]])
        valid = get_valid(dg_rows[p], p)
        ax.loglog(dofs[valid], errs[valid], 'o-', color=colors_dg[p],
                  markersize=6, linewidth=1.5, markerfacecolor='white',
                  markeredgewidth=1.5, label=f'DG P{p}', zorder=3)

    ax.set_xlabel('Degrees of freedom', fontsize=11)
    ax.set_ylabel('$L^\\infty$ error', fontsize=11)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, which='major', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Linf_vs_DoF.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("\nSaved: Linf_vs_DoF.png")

    # Plot 2: Linf vs compute time
    fig, ax = plt.subplots(figsize=(7, 5.5))

    fv_times = [r['Simtime'] for r in fv_rows[:-1]]
    ax.loglog(fv_times, fv_linf, 'ks--', markersize=5, linewidth=1.2,
              label='FV WENO3', zorder=2)

    for p in DG_POLY_DEGS:
        if not dg_rows[p]:
            continue
        times = np.array([r['Simtime'] for r in dg_rows[p]])
        errs = np.array([r['Linferror'] for r in dg_rows[p]])
        valid = get_valid(dg_rows[p], p)
        ax.loglog(times[valid], errs[valid], 'o-', color=colors_dg[p],
                  markersize=6, linewidth=1.5, markerfacecolor='white',
                  markeredgewidth=1.5, label=f'DG P{p}', zorder=3)

    ax.set_xlabel('Compute time [s]', fontsize=11)
    ax.set_ylabel('$L^\\infty$ error', fontsize=11)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, which='major', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Linf_vs_compute.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: Linf_vs_compute.png")

    print("\nDone!")


if __name__ == '__main__':
    main()
