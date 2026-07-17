#!/usr/bin/env python3
"""rLRMP lin 1comp -- FV vs DG convergence.

Reference: FV Z65536.
Style: matches rGRM v7 generate_convergence.py
"""

import os
import csv
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'radialDG-results-LRMP')
BASE = os.path.expanduser("~/CADET-Verification/output/test_cadet-core/radialDG/LRMP")

NCOMP = 1
SUM_NBOUND = 1

FV_LEVELS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
FV_REF = 65536

DG_POLY_DEGS = [1, 2, 3, 4, 5]
DG_NELEM = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def open_fv(z):
    path = os.path.join(BASE, f'radLRMP_FV_WENO3_lin_1comp_FV_Z{z}.h5')
    f = h5py.File(path, 'r')
    sim_time = f['meta']['TIME_SIM'][()]
    ncol = int(f['input']['model']['unit_001']['discretization']['NCOL'][()])
    sol_out = f['output']['solution']['unit_001']['SOLUTION_OUTLET'][:]
    f.close()
    # LRMP: DoF = ncol * (ncomp + ncomp + sum_nbound) per partype
    dof_bulk = ncol * NCOMP
    dof_total = ncol * (NCOMP + NCOMP + SUM_NBOUND)
    return sol_out, sim_time, ncol, dof_bulk, dof_total


def open_dg(p, ne):
    path = os.path.join(BASE, f'radLRMP_DG_lin_1comp_P{p}_DG_P{p}Z{ne}.h5')
    if not os.path.exists(path):
        return None
    f = h5py.File(path, 'r')
    if 'output' not in f:
        f.close()
        return None
    sol_out = f['output']['solution']['unit_001']['SOLUTION_OUTLET'][:]
    sim_time = f['meta']['TIME_SIM'][()]
    polydeg = int(f['input']['model']['unit_001']['discretization']['POLYDEG'][()])
    nelem = int(f['input']['model']['unit_001']['discretization']['NELEM'][()])
    f.close()
    dof_bulk = nelem * (polydeg + 1) * NCOMP
    dof_total = nelem * (polydeg + 1) * (NCOMP + NCOMP + SUM_NBOUND)
    return sol_out, sim_time, polydeg, nelem, dof_bulk, dof_total


def compute_errors(sol, ref):
    diff = sol - ref
    return np.sqrt(np.sum(diff**2)), np.max(np.abs(diff))


def compute_eoc(rows, err_key, eoc_key):
    for i in range(1, len(rows)):
        if rows[i][err_key] > 0 and rows[i-1][err_key] > 0:
            rows[i][eoc_key] = np.log(rows[i-1][err_key] / rows[i][err_key]) / np.log(2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ref_out, _, _, _, _ = open_fv(FV_REF)
    print(f"Reference: FV Z{FV_REF}")

    # -- FV self-convergence --
    fv_rows = []
    for z in FV_LEVELS:
        sol_out, sim_time, ncol, dof_bulk, dof_total = open_fv(z)
        row = {'NCOL': ncol,
               'DoF_bulk': dof_bulk, 'DoF_total': dof_total,
               'L2_outlet': 0.0, 'L2EOC_outlet': 0.0,
               'Linf_outlet': 0.0, 'LinfEOC_outlet': 0.0,
               'Simtime': sim_time}
        if z != FV_REF:
            row['L2_outlet'], row['Linf_outlet'] = compute_errors(sol_out, ref_out)
            print(f"  FV Z{z}: Linf={row['Linf_outlet']:.4e}")
        fv_rows.append(row)

    compute_eoc(fv_rows, 'L2_outlet', 'L2EOC_outlet')
    compute_eoc(fv_rows, 'Linf_outlet', 'LinfEOC_outlet')

    # -- DG convergence --
    dg_rows = {}
    for p in DG_POLY_DEGS:
        dg_rows[p] = []
        for ne in DG_NELEM:
            result = open_dg(p, ne)
            if result is None:
                continue
            sol_out, sim_time, polydeg, nelem, dof_bulk, dof_total = result
            l2, linf = compute_errors(sol_out, ref_out)
            row = {'POLYDEG': polydeg, 'NELEM': nelem,
                   'DoF_bulk': dof_bulk, 'DoF_total': dof_total,
                   'L2_outlet': l2, 'L2EOC_outlet': 0.0,
                   'Linf_outlet': linf, 'LinfEOC_outlet': 0.0,
                   'Simtime': sim_time}
            print(f"  DG P{p} Z{ne}: Linf={linf:.4e}, DoF={dof_total}")
            dg_rows[p].append(row)

        compute_eoc(dg_rows[p], 'L2_outlet', 'L2EOC_outlet')
        compute_eoc(dg_rows[p], 'Linf_outlet', 'LinfEOC_outlet')

    # -- CSVs --
    fv_csv = os.path.join(OUT_DIR, 'rLRMP_lin_FV_outlet.csv')
    with open(fv_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['NCOL', 'DoF_bulk', 'DoF_total',
                     'L2_outlet', 'L2EOC_outlet', 'Linf_outlet', 'LinfEOC_outlet', 'Simtime'])
        for r in fv_rows[:-1]:
            w.writerow([r['NCOL'], r['DoF_bulk'], r['DoF_total'],
                        f"{r['L2_outlet']:.6e}", f"{r['L2EOC_outlet']:.4f}",
                        f"{r['Linf_outlet']:.6e}", f"{r['LinfEOC_outlet']:.4f}",
                        f"{r['Simtime']:.3f}"])
    print(f"\nSaved: {fv_csv}")

    dg_csv = os.path.join(OUT_DIR, 'rLRMP_lin_DG_outlet.csv')
    with open(dg_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Method', 'POLYDEG', 'NELEM',
                     'DoF_bulk', 'DoF_total',
                     'L2_outlet', 'L2EOC_outlet', 'Linf_outlet', 'LinfEOC_outlet', 'Simtime'])
        for p in DG_POLY_DEGS:
            for r in dg_rows[p]:
                w.writerow([f'DG P{p}', r['POLYDEG'], r['NELEM'],
                            r['DoF_bulk'], r['DoF_total'],
                            f"{r['L2_outlet']:.6e}", f"{r['L2EOC_outlet']:.4f}",
                            f"{r['Linf_outlet']:.6e}", f"{r['LinfEOC_outlet']:.4f}",
                            f"{r['Simtime']:.3f}"])
    print(f"Saved: {dg_csv}")

    # -- Tables --
    print(f"\n{'='*80}")
    print(f"FV WENO3 self-convergence -- Outlet (ref = Z{FV_REF})")
    print(f"{'='*80}")
    print(f"{'NCOL':>6} {'DoF':>10} {'Linf':>12} {'LinfEOC':>8} {'L2':>12} {'L2EOC':>8} {'Time[s]':>10}")
    for r in fv_rows:
        print(f"{r['NCOL']:>6} {r['DoF_total']:>10} "
              f"{r['Linf_outlet']:>12.4e} {r['LinfEOC_outlet']:>8.2f} "
              f"{r['L2_outlet']:>12.4e} {r['L2EOC_outlet']:>8.2f} "
              f"{r['Simtime']:>10.1f}")

    for p in DG_POLY_DEGS:
        print(f"\n{'='*80}")
        print(f"DG P{p} convergence vs FV Z{FV_REF} -- Outlet")
        print(f"{'='*80}")
        print(f"{'Np':>4} {'Ne':>4} {'DoF':>10} "
              f"{'Linf':>12} {'LinfEOC':>8} {'L2':>12} {'L2EOC':>8} {'Time[s]':>10}")
        for r in dg_rows[p]:
            print(f"{r['POLYDEG']:>4} {r['NELEM']:>4} "
                  f"{r['DoF_total']:>10} "
                  f"{r['Linf_outlet']:>12.4e} {r['LinfEOC_outlet']:>8.2f} "
                  f"{r['L2_outlet']:>12.4e} {r['L2EOC_outlet']:>8.2f} "
                  f"{r['Simtime']:>10.1f}")

    # -- Plots --
    colors_dg = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd'}

    def make_plot(fv_x, fv_y, dg_x_y, xlabel, ylabel, title, fname):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.loglog(fv_x, fv_y, 'ks--', markersize=5, linewidth=1.2,
                  label='FV WENO3', zorder=2)
        for p in DG_POLY_DEGS:
            if p not in dg_x_y or len(dg_x_y[p][0]) == 0:
                continue
            ax.loglog(dg_x_y[p][0], dg_x_y[p][1], 'o-', color=colors_dg[p],
                      markersize=6, linewidth=1.5, markerfacecolor='white',
                      markeredgewidth=1.5, label=f'DG P{p}', zorder=3)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        if title:
            ax.set_title(title, fontsize=13)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        ax.grid(True, which='major', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")

    fv_dofs = [r['DoF_total'] for r in fv_rows[:-1]]
    fv_linf = [r['Linf_outlet'] for r in fv_rows[:-1]]
    fv_times = [r['Simtime'] for r in fv_rows[:-1]]

    # Drop plateau points where DG error hits FV reference accuracy floor
    def trim_plateau(rows):
        if len(rows) <= 1:
            return rows
        keep = [rows[0]]
        for i in range(1, len(rows)):
            if rows[i]['Linf_outlet'] < rows[i-1]['Linf_outlet'] * 0.7:
                keep.append(rows[i])
            else:
                break
        return keep

    dg_plot = {p: trim_plateau(dg_rows[p]) for p in DG_POLY_DEGS}

    dg_dof = {p: (np.array([r['DoF_total'] for r in dg_plot[p]]),
                  np.array([r['Linf_outlet'] for r in dg_plot[p]])) for p in DG_POLY_DEGS}
    dg_time = {p: (np.array([r['Simtime'] for r in dg_plot[p]]),
                   np.array([r['Linf_outlet'] for r in dg_plot[p]])) for p in DG_POLY_DEGS}

    make_plot(fv_dofs, fv_linf, dg_dof,
              'Degrees of freedom', '$L^\\infty$ error (outlet)',
              '',
              'Linf_vs_DoF_outlet.png')

    make_plot(fv_times, fv_linf, dg_time,
              'Compute time [s]', '$L^\\infty$ error (outlet)',
              '',
              'Linf_vs_compute_outlet.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
