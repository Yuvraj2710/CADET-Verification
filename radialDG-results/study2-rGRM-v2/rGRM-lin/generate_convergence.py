#!/usr/bin/env python3
"""rGRM lin 1comp — FV vs DG convergence (Breuer thesis format).

Outlet convergence: FV Z1024 reference (self-convergence for FV, cross-ref for DG)
Particle domain convergence: FV Z1024 reference (piecewise-constant interp for FV)

Produces:
  - rGRM_lin_FV_outlet.csv          (FV outlet self-convergence)
  - rGRM_lin_DG_outlet.csv          (DG outlet vs FV Z1024)
  - rGRM_lin_FV_particle.csv        (FV particle self-convergence)
  - rGRM_lin_DG_particle.csv        (DG particle vs FV Z1024)
  - Linf_vs_DoF_outlet.png          (FV + DG on one plot)
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
BASE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output', 'test_cadet-core', 'radialDG', 'v2')

# --- Configuration ---
NCOMP = 1
SUM_NBOUND = 1

FV_LEVELS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
FV_REF = 4096

DG_POLY_DEGS = [1, 2, 3, 4]
DG_NELEM = [4, 8, 16, 32, 64, 128]

CHUNK_SIZE = 50  # time steps per chunk for particle arrays


# ─── Loaders ────────────────────────────────────────────────────────

def open_fv(z):
    """Return (h5_handle, sim_time, NCOL, NCELLS, DoF_bulk, DoF_total)."""
    path = os.path.join(BASE, f'radGRM_FV_WENO3_lin_1comp_FV_Z{z}.h5')
    f = h5py.File(path, 'r')
    sim_time = f['meta']['TIME_SIM'][()]
    ncol = int(f['input']['model']['unit_001']['discretization']['NCOL'][()])
    ncells = int(f['input']['model']['unit_001']['particle_type_000']['discretization']['NCELLS'][()])
    dof_bulk = ncol * NCOMP
    dof_total = ncol * (NCOMP + ncells * (NCOMP + SUM_NBOUND))
    return f, sim_time, ncol, ncells, dof_bulk, dof_total


def open_dg(p, ne):
    """Return (h5_handle, sim_time, POLYDEG, NELEM, PAR_POLYDEG, PAR_NELEM, DoF_bulk, DoF_total) or None."""
    path = os.path.join(BASE, f'radGRM_DG_lin_1comp_P{p}_DG_P{p}Z{ne}.h5')
    if not os.path.exists(path):
        return None
    f = h5py.File(path, 'r')
    if 'output' not in f:
        f.close()
        return None
    sim_time = f['meta']['TIME_SIM'][()]
    polydeg = int(f['input']['model']['unit_001']['discretization']['POLYDEG'][()])
    nelem = int(f['input']['model']['unit_001']['discretization']['NELEM'][()])
    par_polydeg = int(f['input']['model']['unit_001']['particle_type_000']['discretization']['PAR_POLYDEG'][()])
    par_nelem = int(f['input']['model']['unit_001']['particle_type_000']['discretization']['PAR_NELEM'][()])
    dof_bulk = nelem * (polydeg + 1) * NCOMP
    dof_total = nelem * ((polydeg + 1) * NCOMP + par_nelem * (par_polydeg + 1) * (NCOMP + SUM_NBOUND))
    return f, sim_time, polydeg, nelem, par_polydeg, par_nelem, dof_bulk, dof_total


# ─── Error computation ──────────────────────────────────────────────

def outlet_errors(sol, ref):
    diff = sol - ref
    return np.sqrt(np.sum(diff**2)), np.max(np.abs(diff))


def particle_errors_fv_chunked(f_coarse, f_ref, ncol_c, ncol_r, field):
    """FV-to-FV particle error with piecewise-constant interpolation (chunked)."""
    ds_c = f_coarse['output']['solution']['unit_001'][field]
    ds_r = f_ref['output']['solution']['unit_001'][field]
    ntime = ds_r.shape[0]

    ratio_ax = ncol_r // ncol_c
    ncells_c = ncol_c // 4
    ncells_r = ncol_r // 4
    ratio_par = ncells_r // ncells_c

    l2_sum, linf_max = 0.0, 0.0
    for t0 in range(0, ntime, CHUNK_SIZE):
        t1 = min(t0 + CHUNK_SIZE, ntime)
        chunk_c = ds_c[t0:t1]
        chunk_r = ds_r[t0:t1]
        chunk_i = np.repeat(np.repeat(chunk_c, ratio_ax, axis=1), ratio_par, axis=2)
        diff = chunk_i - chunk_r
        l2_sum += np.sum(diff**2)
        linf_max = max(linf_max, np.max(np.abs(diff)))
    return np.sqrt(l2_sum), linf_max


def particle_errors_dg_chunked(f_dg, f_ref, field):
    """DG vs FV reference — direct comparison (same outlet time points, shapes must match after mapping).

    DG particle output shape: (nTime, nAxialNodes, nParNodes, nComp)
    FV particle output shape: (nTime, NCOL, NCELLS, nComp)

    These generally don't match in spatial dimensions, so we compute
    the DG L2/Linf over the DG grid directly, and compare outlet only.
    For a proper cross-method particle comparison we'd need projection.

    For now: skip DG particle errors (needs proper DG-to-FV projection).
    """
    return None, None


# ─── EOC computation ────────────────────────────────────────────────

def compute_eoc(rows, err_key, eoc_key):
    for i in range(1, len(rows)):
        if rows[i][err_key] > 0 and rows[i-1][err_key] > 0:
            rows[i][eoc_key] = np.log(rows[i-1][err_key] / rows[i][err_key]) / np.log(2)


# ─── Main ───────────────────────────────────────────────────────────

def main():
    # ── Open FV reference ──
    f_ref, _, ref_ncol, ref_ncells, _, _ = open_fv(FV_REF)
    ref_outlet = f_ref['output']['solution']['unit_001']['SOLUTION_OUTLET'][:]
    print(f"Reference: FV Z{FV_REF} (NCOL={ref_ncol}, NCELLS={ref_ncells})")

    # ════════════════════════════════════════════════════════════════
    # 1. FV self-convergence
    # ════════════════════════════════════════════════════════════════
    fv_rows = []
    for z in FV_LEVELS:
        f, sim_time, ncol, ncells, dof_bulk, dof_total = open_fv(z)
        row = {'NCOL': ncol, 'NCELLS': ncells,
               'DoF_bulk': dof_bulk, 'DoF_total': dof_total,
               'L2_outlet': 0.0, 'L2EOC_outlet': 0.0,
               'Linf_outlet': 0.0, 'LinfEOC_outlet': 0.0,
               'L2_particle': 0.0, 'L2EOC_particle': 0.0,
               'Linf_particle': 0.0, 'LinfEOC_particle': 0.0,
               'L2_solid': 0.0, 'L2EOC_solid': 0.0,
               'Linf_solid': 0.0, 'LinfEOC_solid': 0.0,
               'Simtime': sim_time}

        if z != FV_REF:
            # Outlet
            sol_out = f['output']['solution']['unit_001']['SOLUTION_OUTLET'][:]
            row['L2_outlet'], row['Linf_outlet'] = outlet_errors(sol_out, ref_outlet)

            # Particle (liquid) — only if particle output exists
            has_particle = 'SOLUTION_PARTICLE' in f['output']['solution']['unit_001']
            has_particle_ref = 'SOLUTION_PARTICLE' in f_ref['output']['solution']['unit_001']
            if has_particle and has_particle_ref:
                row['L2_particle'], row['Linf_particle'] = particle_errors_fv_chunked(
                    f, f_ref, ncol, ref_ncol, 'SOLUTION_PARTICLE')
                row['L2_solid'], row['Linf_solid'] = particle_errors_fv_chunked(
                    f, f_ref, ncol, ref_ncol, 'SOLUTION_SOLID')
                print(f"  FV Z{z}: outlet Linf={row['Linf_outlet']:.4e}, "
                      f"particle Linf={row['Linf_particle']:.4e}, solid Linf={row['Linf_solid']:.4e}")
            else:
                print(f"  FV Z{z}: outlet Linf={row['Linf_outlet']:.4e} (no particle data)")

        fv_rows.append(row)
        f.close()

    # EOC
    for key_err, key_eoc in [('L2_outlet', 'L2EOC_outlet'), ('Linf_outlet', 'LinfEOC_outlet'),
                              ('L2_particle', 'L2EOC_particle'), ('Linf_particle', 'LinfEOC_particle'),
                              ('L2_solid', 'L2EOC_solid'), ('Linf_solid', 'LinfEOC_solid')]:
        compute_eoc(fv_rows, key_err, key_eoc)

    # ════════════════════════════════════════════════════════════════
    # 2. DG convergence vs FV reference
    # ════════════════════════════════════════════════════════════════
    dg_rows = {}
    for p in DG_POLY_DEGS:
        dg_rows[p] = []
        for ne in DG_NELEM:
            result = open_dg(p, ne)
            if result is None:
                print(f"  Skipping DG P{p} Z{ne} (missing/no output)")
                continue
            f, sim_time, polydeg, nelem, par_polydeg, par_nelem, dof_bulk, dof_total = result

            sol_out = f['output']['solution']['unit_001']['SOLUTION_OUTLET'][:]
            l2_out, linf_out = outlet_errors(sol_out, ref_outlet)

            row = {'POLYDEG': polydeg, 'NELEM': nelem,
                   'PAR_POLYDEG': par_polydeg, 'PAR_NELEM': par_nelem,
                   'DoF_bulk': dof_bulk, 'DoF_total': dof_total,
                   'L2_outlet': l2_out, 'L2EOC_outlet': 0.0,
                   'Linf_outlet': linf_out, 'LinfEOC_outlet': 0.0,
                   'Simtime': sim_time}

            print(f"  DG P{p} Z{ne}: outlet Linf={linf_out:.4e}, DoF={dof_total}")
            dg_rows[p].append(row)
            f.close()

        # EOC
        compute_eoc(dg_rows[p], 'L2_outlet', 'L2EOC_outlet')
        compute_eoc(dg_rows[p], 'Linf_outlet', 'LinfEOC_outlet')

    f_ref.close()

    # ════════════════════════════════════════════════════════════════
    # 3. Write CSVs
    # ════════════════════════════════════════════════════════════════

    # FV outlet CSV
    fv_out_csv = os.path.join(OUT_DIR, 'rGRM_lin_FV_outlet.csv')
    with open(fv_out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['NCOL', 'NCELLS', 'DoF_bulk', 'DoF_total',
                     'L2_outlet', 'L2EOC_outlet', 'Linf_outlet', 'LinfEOC_outlet',
                     'Simtime'])
        for r in fv_rows[:-1]:  # exclude reference
            w.writerow([r['NCOL'], r['NCELLS'], r['DoF_bulk'], r['DoF_total'],
                        f"{r['L2_outlet']:.6e}", f"{r['L2EOC_outlet']:.4f}",
                        f"{r['Linf_outlet']:.6e}", f"{r['LinfEOC_outlet']:.4f}",
                        f"{r['Simtime']:.3f}"])
    print(f"\nSaved: {fv_out_csv}")

    # FV particle CSV
    fv_par_csv = os.path.join(OUT_DIR, 'rGRM_lin_FV_particle.csv')
    with open(fv_par_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['NCOL', 'NCELLS', 'DoF_bulk', 'DoF_total',
                     'L2_particle', 'L2EOC_particle', 'Linf_particle', 'LinfEOC_particle',
                     'L2_solid', 'L2EOC_solid', 'Linf_solid', 'LinfEOC_solid',
                     'Simtime'])
        for r in fv_rows[:-1]:
            w.writerow([r['NCOL'], r['NCELLS'], r['DoF_bulk'], r['DoF_total'],
                        f"{r['L2_particle']:.6e}", f"{r['L2EOC_particle']:.4f}",
                        f"{r['Linf_particle']:.6e}", f"{r['LinfEOC_particle']:.4f}",
                        f"{r['L2_solid']:.6e}", f"{r['L2EOC_solid']:.4f}",
                        f"{r['Linf_solid']:.6e}", f"{r['LinfEOC_solid']:.4f}",
                        f"{r['Simtime']:.3f}"])
    print(f"Saved: {fv_par_csv}")

    # DG outlet CSV
    dg_out_csv = os.path.join(OUT_DIR, 'rGRM_lin_DG_outlet.csv')
    with open(dg_out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Method', 'POLYDEG', 'NELEM', 'PAR_POLYDEG', 'PAR_NELEM',
                     'DoF_bulk', 'DoF_total',
                     'L2_outlet', 'L2EOC_outlet', 'Linf_outlet', 'LinfEOC_outlet',
                     'Simtime'])
        for p in DG_POLY_DEGS:
            for r in dg_rows[p]:
                w.writerow([f'DG P{p}', r['POLYDEG'], r['NELEM'],
                            r['PAR_POLYDEG'], r['PAR_NELEM'],
                            r['DoF_bulk'], r['DoF_total'],
                            f"{r['L2_outlet']:.6e}", f"{r['L2EOC_outlet']:.4f}",
                            f"{r['Linf_outlet']:.6e}", f"{r['LinfEOC_outlet']:.4f}",
                            f"{r['Simtime']:.3f}"])
    print(f"Saved: {dg_out_csv}")

    # ════════════════════════════════════════════════════════════════
    # 4. Print tables
    # ════════════════════════════════════════════════════════════════

    print(f"\n{'='*80}")
    print(f"FV WENO3 self-convergence — Outlet (ref = Z{FV_REF})")
    print(f"{'='*80}")
    print(f"{'NCOL':>6} {'NCELLS':>6} {'DoF':>10} {'Linf':>12} {'LinfEOC':>8} {'L2':>12} {'L2EOC':>8} {'Time[s]':>10}")
    for r in fv_rows:
        print(f"{r['NCOL']:>6} {r['NCELLS']:>6} {r['DoF_total']:>10} "
              f"{r['Linf_outlet']:>12.4e} {r['LinfEOC_outlet']:>8.2f} "
              f"{r['L2_outlet']:>12.4e} {r['L2EOC_outlet']:>8.2f} "
              f"{r['Simtime']:>10.1f}")

    print(f"\n{'='*80}")
    print(f"FV WENO3 self-convergence — Particle (liquid) (ref = Z{FV_REF})")
    print(f"{'='*80}")
    print(f"{'NCOL':>6} {'NCELLS':>6} {'DoF':>10} {'Linf':>12} {'LinfEOC':>8} {'L2':>12} {'L2EOC':>8}")
    for r in fv_rows:
        print(f"{r['NCOL']:>6} {r['NCELLS']:>6} {r['DoF_total']:>10} "
              f"{r['Linf_particle']:>12.4e} {r['LinfEOC_particle']:>8.2f} "
              f"{r['L2_particle']:>12.4e} {r['L2EOC_particle']:>8.2f}")

    print(f"\n{'='*80}")
    print(f"FV WENO3 self-convergence — Solid (bound) (ref = Z{FV_REF})")
    print(f"{'='*80}")
    print(f"{'NCOL':>6} {'NCELLS':>6} {'DoF':>10} {'Linf':>12} {'LinfEOC':>8} {'L2':>12} {'L2EOC':>8}")
    for r in fv_rows:
        print(f"{r['NCOL']:>6} {r['NCELLS']:>6} {r['DoF_total']:>10} "
              f"{r['Linf_solid']:>12.4e} {r['LinfEOC_solid']:>8.2f} "
              f"{r['L2_solid']:>12.4e} {r['L2EOC_solid']:>8.2f}")

    for p in DG_POLY_DEGS:
        print(f"\n{'='*80}")
        print(f"DG P{p} convergence vs FV Z{FV_REF} — Outlet")
        print(f"{'='*80}")
        print(f"{'Np':>4} {'Ne':>4} {'Par_Np':>6} {'Par_Ne':>6} {'DoF':>10} "
              f"{'Linf':>12} {'LinfEOC':>8} {'L2':>12} {'L2EOC':>8} {'Time[s]':>10}")
        for r in dg_rows[p]:
            print(f"{r['POLYDEG']:>4} {r['NELEM']:>4} {r['PAR_POLYDEG']:>6} {r['PAR_NELEM']:>6} "
                  f"{r['DoF_total']:>10} "
                  f"{r['Linf_outlet']:>12.4e} {r['LinfEOC_outlet']:>8.2f} "
                  f"{r['L2_outlet']:>12.4e} {r['L2EOC_outlet']:>8.2f} "
                  f"{r['Simtime']:>10.1f}")

    # ════════════════════════════════════════════════════════════════
    # 5. Plots
    # ════════════════════════════════════════════════════════════════
    colors_dg = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728'}

    def plot_convergence(fv_x, fv_y, dg_data, xlabel, ylabel, title, fname):
        """fv_x/fv_y: lists. dg_data: dict {p: (x_arr, y_arr)}."""
        fig, ax = plt.subplots(figsize=(7, 5.5))

        ax.loglog(fv_x, fv_y, 'ks--', markersize=5, linewidth=1.2,
                  label='FV WENO3', zorder=2)

        for p in DG_POLY_DEGS:
            if p not in dg_data or len(dg_data[p][0]) == 0:
                continue
            ax.loglog(dg_data[p][0], dg_data[p][1], 'o-', color=colors_dg[p],
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

    # ── Outlet: Linf vs DoF ──
    fv_dofs = [r['DoF_total'] for r in fv_rows[:-1]]
    fv_linf_out = [r['Linf_outlet'] for r in fv_rows[:-1]]
    fv_times = [r['Simtime'] for r in fv_rows[:-1]]

    dg_dof_out = {}
    dg_linf_out = {}
    dg_time_out = {}
    for p in DG_POLY_DEGS:
        dg_dof_out[p] = (np.array([r['DoF_total'] for r in dg_rows[p]]),
                         np.array([r['Linf_outlet'] for r in dg_rows[p]]))
        dg_time_out[p] = (np.array([r['Simtime'] for r in dg_rows[p]]),
                          np.array([r['Linf_outlet'] for r in dg_rows[p]]))

    plot_convergence(fv_dofs, fv_linf_out, dg_dof_out,
                     'Degrees of freedom', '$L^\\infty$ error (outlet)',
                     f'rGRM lin — Outlet convergence (ref FV Z{FV_REF})',
                     'Linf_vs_DoF_outlet.png')

    plot_convergence(fv_times, fv_linf_out, dg_time_out,
                     'Compute time [s]', '$L^\\infty$ error (outlet)',
                     f'rGRM lin — Outlet convergence (ref FV Z{FV_REF})',
                     'Linf_vs_compute_outlet.png')

    # ── Particle: Linf vs DoF (FV only for now) ──
    fv_linf_par = [r['Linf_particle'] for r in fv_rows[:-1]]
    fv_linf_sol = [r['Linf_solid'] for r in fv_rows[:-1]]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.loglog(fv_dofs, fv_linf_out, 'ks--', markersize=5, linewidth=1.2,
              label='FV outlet', zorder=2)
    ax.loglog(fv_dofs, fv_linf_par, 'bo-', markersize=6, linewidth=1.2,
              markerfacecolor='white', markeredgewidth=1.5,
              label='FV particle (liquid)', zorder=3)
    ax.loglog(fv_dofs, fv_linf_sol, 'r^-', markersize=6, linewidth=1.2,
              markerfacecolor='white', markeredgewidth=1.5,
              label='FV solid (bound)', zorder=3)
    ax.set_xlabel('Degrees of freedom', fontsize=11)
    ax.set_ylabel('$L^\\infty$ error', fontsize=11)
    ax.set_title(f'rGRM lin — FV WENO3 particle convergence (ref Z{FV_REF})', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='major', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Linf_vs_DoF_particle.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: Linf_vs_DoF_particle.png")

    print("\nDone!")


if __name__ == '__main__':
    main()
