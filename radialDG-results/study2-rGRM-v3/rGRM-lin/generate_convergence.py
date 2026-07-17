#!/usr/bin/env python3
"""rGRM lin 1comp — FV vs DG convergence (less dispersive v3 setting).

Less dispersive parameters to show proper DG P+1 convergence rates:
  COL_DISPERSION = 5.75e-10, FILM_DIFFUSION = 6.9e-7, PAR_DIFFUSION = 6.07e-11

Outlet + particle + solid convergence: FV Z4096 reference.

Produces:
  - rGRM_lin_FV_outlet.csv
  - rGRM_lin_DG_outlet.csv
  - Linf_vs_DoF_outlet.png
  - Linf_vs_compute_outlet.png
  - Linf_vs_DoF_particle.png
  - Linf_vs_compute_particle.png
"""

import os
import csv
import numpy as np
import h5py
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'test_cadet-core', 'radialDG', 'v3-results')

# --- Configuration ---
NCOMP = 1
SUM_NBOUND = 1

FV_LEVELS = [16, 32, 64, 128, 256, 512, 1024]
FV_REF = 1024

DG_POLY_DEGS = [1, 2, 3, 4]
DG_NELEM = [4, 8, 16, 32, 64, 128]


# ─── Loaders ────────────────────────────────────────────────────────

def open_fv(z):
    path = os.path.join(BASE, f'radGRM_FV_WENO3_lin_1comp_FV_Z{z}.h5')
    f = h5py.File(path, 'r')
    sim_time = f['meta']['TIME_SIM'][()]
    ncol = int(f['input']['model']['unit_001']['discretization']['NCOL'][()])
    ncells = int(f['input']['model']['unit_001']['particle_type_000']['discretization']['NCELLS'][()])
    sol_out = f['output']['solution']['unit_001']['SOLUTION_OUTLET'][:]
    has_particle = 'SOLUTION_PARTICLE' in f['output']['solution']['unit_001']
    sol_par = f['output']['solution']['unit_001']['SOLUTION_PARTICLE'][:] if has_particle else None
    sol_sol = f['output']['solution']['unit_001']['SOLUTION_SOLID'][:] if has_particle else None
    f.close()
    dof_bulk = ncol * NCOMP
    dof_total = ncol * (NCOMP + ncells * (NCOMP + SUM_NBOUND))
    return sol_out, sol_par, sol_sol, sim_time, ncol, ncells, dof_bulk, dof_total


def open_dg(p, ne):
    path = os.path.join(BASE, f'radGRM_DG_lin_1comp_P{p}_DG_P{p}Z{ne}.h5')
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
    par_polydeg = int(f['input']['model']['unit_001']['particle_type_000']['discretization']['PAR_POLYDEG'][()])
    par_nelem = int(f['input']['model']['unit_001']['particle_type_000']['discretization']['PAR_NELEM'][()])
    f.close()
    dof_bulk = nelem * (polydeg + 1) * NCOMP
    dof_total = nelem * ((polydeg + 1) * NCOMP + par_nelem * (par_polydeg + 1) * (NCOMP + SUM_NBOUND))
    return sol_out, sim_time, polydeg, nelem, par_polydeg, par_nelem, dof_bulk, dof_total


def compute_errors(sol, ref):
    diff = sol - ref
    return np.sqrt(np.sum(diff**2)), np.max(np.abs(diff))


def compute_eoc(rows, err_key, eoc_key):
    for i in range(1, len(rows)):
        if rows[i][err_key] > 0 and rows[i-1][err_key] > 0:
            rows[i][eoc_key] = np.log(rows[i-1][err_key] / rows[i][err_key]) / np.log(2)


def main():
    # ── Load FV reference ──
    ref_out, ref_par, ref_sol, _, _, _, _, _ = open_fv(FV_REF)
    print(f"Reference: FV Z{FV_REF}")
    has_particle_ref = ref_par is not None

    # ── FV self-convergence ──
    fv_rows = []
    for z in FV_LEVELS:
        sol_out, sol_par, sol_sol, sim_time, ncol, ncells, dof_bulk, dof_total = open_fv(z)
        row = {'NCOL': ncol, 'NCELLS': ncells,
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

    # ── DG convergence ──
    dg_rows = {}
    for p in DG_POLY_DEGS:
        dg_rows[p] = []
        for ne in DG_NELEM:
            result = open_dg(p, ne)
            if result is None:
                print(f"  Skipping DG P{p} Z{ne} (missing/no output)")
                continue
            sol_out, sim_time, polydeg, nelem, par_polydeg, par_nelem, dof_bulk, dof_total = result
            l2, linf = compute_errors(sol_out, ref_out)
            row = {'POLYDEG': polydeg, 'NELEM': nelem,
                   'PAR_POLYDEG': par_polydeg, 'PAR_NELEM': par_nelem,
                   'DoF_bulk': dof_bulk, 'DoF_total': dof_total,
                   'L2_outlet': l2, 'L2EOC_outlet': 0.0,
                   'Linf_outlet': linf, 'LinfEOC_outlet': 0.0,
                   'Simtime': sim_time}
            print(f"  DG P{p} Z{ne}: Linf={linf:.4e}, DoF={dof_total}")
            dg_rows[p].append(row)

        compute_eoc(dg_rows[p], 'L2_outlet', 'L2EOC_outlet')
        compute_eoc(dg_rows[p], 'Linf_outlet', 'LinfEOC_outlet')

    # ── CSVs ──
    fv_csv = os.path.join(OUT_DIR, 'rGRM_lin_FV_outlet.csv')
    with open(fv_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['NCOL', 'NCELLS', 'DoF_bulk', 'DoF_total',
                     'L2_outlet', 'L2EOC_outlet', 'Linf_outlet', 'LinfEOC_outlet', 'Simtime'])
        for r in fv_rows[:-1]:
            w.writerow([r['NCOL'], r['NCELLS'], r['DoF_bulk'], r['DoF_total'],
                        f"{r['L2_outlet']:.6e}", f"{r['L2EOC_outlet']:.4f}",
                        f"{r['Linf_outlet']:.6e}", f"{r['LinfEOC_outlet']:.4f}",
                        f"{r['Simtime']:.3f}"])
    print(f"\nSaved: {fv_csv}")

    dg_csv = os.path.join(OUT_DIR, 'rGRM_lin_DG_outlet.csv')
    with open(dg_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Method', 'POLYDEG', 'NELEM', 'PAR_POLYDEG', 'PAR_NELEM',
                     'DoF_bulk', 'DoF_total',
                     'L2_outlet', 'L2EOC_outlet', 'Linf_outlet', 'LinfEOC_outlet', 'Simtime'])
        for p in DG_POLY_DEGS:
            for r in dg_rows[p]:
                w.writerow([f'DG P{p}', r['POLYDEG'], r['NELEM'],
                            r['PAR_POLYDEG'], r['PAR_NELEM'],
                            r['DoF_bulk'], r['DoF_total'],
                            f"{r['L2_outlet']:.6e}", f"{r['L2EOC_outlet']:.4f}",
                            f"{r['Linf_outlet']:.6e}", f"{r['LinfEOC_outlet']:.4f}",
                            f"{r['Simtime']:.3f}"])
    print(f"Saved: {dg_csv}")

    # ── Tables ──
    print(f"\n{'='*90}")
    print(f"FV WENO3 self-convergence — Outlet (ref = Z{FV_REF})")
    print(f"{'='*90}")
    print(f"{'NCOL':>6} {'NCELLS':>6} {'DoF':>10} {'Linf':>12} {'LinfEOC':>8} {'L2':>12} {'L2EOC':>8} {'Time[s]':>10}")
    for r in fv_rows:
        print(f"{r['NCOL']:>6} {r['NCELLS']:>6} {r['DoF_total']:>10} "
              f"{r['Linf_outlet']:>12.4e} {r['LinfEOC_outlet']:>8.2f} "
              f"{r['L2_outlet']:>12.4e} {r['L2EOC_outlet']:>8.2f} "
              f"{r['Simtime']:>10.1f}")

    for p in DG_POLY_DEGS:
        print(f"\n{'='*90}")
        print(f"DG P{p} convergence vs FV Z{FV_REF} — Outlet")
        print(f"{'='*90}")
        print(f"{'Np':>4} {'Ne':>4} {'Par_Np':>6} {'Par_Ne':>6} {'DoF':>10} "
              f"{'Linf':>12} {'LinfEOC':>8} {'L2':>12} {'L2EOC':>8} {'Time[s]':>10}")
        for r in dg_rows[p]:
            print(f"{r['POLYDEG']:>4} {r['NELEM']:>4} {r['PAR_POLYDEG']:>6} {r['PAR_NELEM']:>6} "
                  f"{r['DoF_total']:>10} "
                  f"{r['Linf_outlet']:>12.4e} {r['LinfEOC_outlet']:>8.2f} "
                  f"{r['L2_outlet']:>12.4e} {r['L2EOC_outlet']:>8.2f} "
                  f"{r['Simtime']:>10.1f}")

    # ── Plots ──
    colors_dg = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728'}

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
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, which='major', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")

    fv_dofs = [r['DoF_total'] for r in fv_rows[:-1]]
    fv_linf = [r['Linf_outlet'] for r in fv_rows[:-1]]
    fv_times = [r['Simtime'] for r in fv_rows[:-1]]

    dg_dof = {p: (np.array([r['DoF_total'] for r in dg_rows[p]]),
                  np.array([r['Linf_outlet'] for r in dg_rows[p]])) for p in DG_POLY_DEGS}
    dg_time = {p: (np.array([r['Simtime'] for r in dg_rows[p]]),
                   np.array([r['Linf_outlet'] for r in dg_rows[p]])) for p in DG_POLY_DEGS}

    make_plot(fv_dofs, fv_linf, dg_dof,
              'Degrees of freedom', '$L^\\infty$ error (outlet)',
              f'rGRM lin (less dispersive) — Outlet (ref FV Z{FV_REF})',
              'Linf_vs_DoF_outlet.png')

    make_plot(fv_times, fv_linf, dg_time,
              'Compute time [s]', '$L^\\infty$ error (outlet)',
              f'rGRM lin (less dispersive) — Outlet (ref FV Z{FV_REF})',
              'Linf_vs_compute_outlet.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
