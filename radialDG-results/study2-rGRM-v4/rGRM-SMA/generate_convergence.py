#!/usr/bin/env python3
"""rGRM SMA 4comp -- FV vs DG convergence (v4: very low dispersion).

COL_DISPERSION = 5.75e-12, FILM_DIFFUSION = 6.9e-9
PORE_DIFFUSION = [7e-13, 6.07e-14, 6.07e-14, 6.07e-14]
Reference: FV Z1024.

Computes convergence for both outlet and particle solutions.
"""

import os
import csv
import numpy as np
import h5py
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.expanduser("~/CADET-Verification/output/test_cadet-core/radialDG/v4-SMA")

NCOMP = 4
SUM_NBOUND = 4

FV_LEVELS = [16, 32, 64, 128, 256, 512, 1024]
FV_REF = 1024

DG_POLY_DEGS = [1, 2, 3, 4]
DG_NELEM = [4, 8, 16, 32, 64, 128]

REF_PATH = os.path.join(BASE, f'radGRM_FV_WENO3_SMA_4comp_FV_Z{FV_REF}.h5')

# Z1024 particle data is a single 12.6 GB gzip chunk and can't be read.
# Use Z512 as reference for particle/solid convergence.
PAR_REF = 512
PAR_REF_PATH = os.path.join(BASE, f'radGRM_FV_WENO3_SMA_4comp_FV_Z{PAR_REF}.h5')


# ── Grid position helpers ──

def equivolume_faces(r0, r1, n):
    return np.sqrt(np.linspace(r0**2, r1**2, n + 1))


def equivolume_centers(r0, r1, n):
    faces = equivolume_faces(r0, r1, n)
    return 0.5 * (faces[:-1] + faces[1:])


def equispaced_faces(rp, n):
    return np.linspace(0, rp, n + 1)


def equispaced_centers(rp, n):
    faces = equispaced_faces(rp, n)
    return 0.5 * (faces[:-1] + faces[1:])


def cgl_nodes_ref(p):
    """CGL nodes on [-1, 1]."""
    return -np.cos(np.arange(p + 1) * np.pi / p)


def dg_node_positions(faces, polydeg):
    """Physical positions of DG CGL nodes within elements."""
    ref = cgl_nodes_ref(polydeg)
    positions = []
    for i in range(len(faces) - 1):
        a, b = faces[i], faces[i + 1]
        positions.extend(0.5 * (a + b) + 0.5 * (b - a) * ref)
    return np.array(positions)


def interp_indices(ref_faces, test_positions):
    """Map test positions to reference cell indices (nearest cell / piecewise constant)."""
    idx = np.searchsorted(ref_faces, test_positions, side='right') - 1
    return np.clip(idx, 0, len(ref_faces) - 2)


# ── Read geometry from h5 ──

def read_geometry(h5path):
    f = h5py.File(h5path, 'r')
    u = f['input']['model']['unit_001']
    r0 = float(u['COL_RADIUS_INNER'][()])
    r1 = float(u['COL_RADIUS_OUTER'][()])
    rp = float(u['particle_type_000']['PAR_RADIUS'][()])
    f.close()
    return r0, r1, rp


# ── Compute particle/solid errors by interpolation ──

def load_particle_ref(ref_path):
    """Load reference particle and solid solutions into memory (once)."""
    print("  Loading reference particle/solid data into memory...")
    f = h5py.File(ref_path, 'r')
    ref_par = f['output']['solution']['unit_001']['SOLUTION_PARTICLE'][:]
    ref_sol = f['output']['solution']['unit_001']['SOLUTION_SOLID'][:]
    f.close()
    print(f"  Loaded: particle {ref_par.shape}, solid {ref_sol.shape}")
    return ref_par, ref_sol


def compute_particle_errors(test_path, ref_par_data, ref_sol_data,
                            test_ax_pos, ref_ax_faces, ref_par_faces):
    """Compute L2 and Linf errors for SOLUTION_PARTICLE and SOLUTION_SOLID.

    Uses pre-loaded reference arrays. Loads test data in one shot.
    """
    ax_idx = interp_indices(ref_ax_faces, test_ax_pos)

    test_f = h5py.File(test_path, 'r')

    # Compute test particle positions
    test_par_disc = test_f['input']['model']['unit_001']['particle_type_000']['discretization']
    spatial = test_f['input']['model']['unit_001']['discretization']
    rp = float(test_f['input']['model']['unit_001']['particle_type_000']['PAR_RADIUS'][()])

    is_dg = False
    if 'SPATIAL_METHOD' in spatial:
        method = spatial['SPATIAL_METHOD'][()]
        if isinstance(method, bytes):
            method = method.decode()
        is_dg = (method == 'DG')

    if is_dg:
        par_nelem = int(test_par_disc['PAR_NELEM'][()])
        par_polydeg = int(test_par_disc['PAR_POLYDEG'][()])
        par_faces = equispaced_faces(rp, par_nelem)
        test_par_pos = dg_node_positions(par_faces, par_polydeg)
    else:
        ncells = int(test_par_disc['NCELLS'][()])
        test_par_pos = equispaced_centers(rp, ncells)

    par_idx = interp_indices(ref_par_faces, test_par_pos)

    # Load test data in one shot
    test_par_data = test_f['output']['solution']['unit_001']['SOLUTION_PARTICLE'][:]
    test_sol_data = test_f['output']['solution']['unit_001']['SOLUTION_SOLID'][:]
    test_f.close()

    # Interpolate reference to test grid (vectorized over all time steps)
    ref_par_on_test = ref_par_data[:, ax_idx][:, :, par_idx]
    ref_sol_on_test = ref_sol_data[:, ax_idx][:, :, par_idx]

    diff_par = test_par_data - ref_par_on_test
    l2_par = np.sqrt(np.sum(diff_par**2))
    linf_par = np.max(np.abs(diff_par))

    diff_sol = test_sol_data - ref_sol_on_test
    l2_sol = np.sqrt(np.sum(diff_sol**2))
    linf_sol = np.max(np.abs(diff_sol))

    return l2_par, linf_par, l2_sol, linf_sol


# ── Standard helpers ──

def open_fv(z):
    path = os.path.join(BASE, f'radGRM_FV_WENO3_SMA_4comp_FV_Z{z}.h5')
    f = h5py.File(path, 'r')
    sim_time = f['meta']['TIME_SIM'][()]
    ncol = int(f['input']['model']['unit_001']['discretization']['NCOL'][()])
    ncells = int(f['input']['model']['unit_001']['particle_type_000']['discretization']['NCELLS'][()])
    sol_out = f['output']['solution']['unit_001']['SOLUTION_OUTLET'][:]
    f.close()
    dof_bulk = ncol * NCOMP
    dof_total = ncol * (NCOMP + ncells * (NCOMP + SUM_NBOUND))
    return sol_out, sim_time, ncol, ncells, dof_bulk, dof_total, path


def open_dg(p, ne):
    path = os.path.join(BASE, f'radGRM_DG_SMA_4comp_P{p}_DG_P{p}Z{ne}.h5')
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
    return sol_out, sim_time, polydeg, nelem, par_polydeg, par_nelem, dof_bulk, dof_total, path


def compute_errors(sol, ref):
    diff = sol - ref
    return np.sqrt(np.sum(diff**2)), np.max(np.abs(diff))


def compute_eoc(rows, err_key, eoc_key):
    for i in range(1, len(rows)):
        if rows[i][err_key] > 0 and rows[i-1][err_key] > 0:
            rows[i][eoc_key] = np.log(rows[i-1][err_key] / rows[i][err_key]) / np.log(2)


def main():
    r0, r1, rp = read_geometry(REF_PATH)
    print(f"Geometry: r0={r0}, r1={r1}, rp={rp}")

    # Particle reference grid faces (Z512 -- Z1024 particle chunk is unreadable)
    par_ref_f = h5py.File(PAR_REF_PATH, 'r')
    par_ref_ncol = int(par_ref_f['input']['model']['unit_001']['discretization']['NCOL'][()])
    par_ref_ncells = int(par_ref_f['input']['model']['unit_001']['particle_type_000']['discretization']['NCELLS'][()])
    par_ref_f.close()
    par_ref_ax_faces = equivolume_faces(r0, r1, par_ref_ncol)
    par_ref_par_faces = equispaced_faces(rp, par_ref_ncells)

    ref_out, _, _, _, _, _, _ = open_fv(FV_REF)
    print(f"Reference: outlet=FV Z{FV_REF}, particle/solid=FV Z{PAR_REF} (NCOL={par_ref_ncol}, NCELLS={par_ref_ncells})")

    # Load particle/solid reference into memory once
    ref_par_data, ref_sol_data = load_particle_ref(PAR_REF_PATH)

    # ── FV self-convergence ──
    print("\n--- FV convergence ---")
    fv_rows = []
    for z in FV_LEVELS:
        sol_out, sim_time, ncol, ncells, dof_bulk, dof_total, fv_path = open_fv(z)
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
            row['L2_outlet'], row['Linf_outlet'] = compute_errors(sol_out, ref_out)
        if z != FV_REF and z != PAR_REF:
            # Particle errors vs Z512 reference
            test_ax_pos = equivolume_centers(r0, r1, ncol)
            l2p, linfp, l2s, linfs = compute_particle_errors(
                fv_path, ref_par_data, ref_sol_data, test_ax_pos, par_ref_ax_faces, par_ref_par_faces)
            row['L2_particle'] = l2p
            row['Linf_particle'] = linfp
            row['L2_solid'] = l2s
            row['Linf_solid'] = linfs
        if z != FV_REF:
            print(f"  FV Z{z}: Linf_out={row['Linf_outlet']:.4e}  Linf_par={row['Linf_particle']:.4e}  Linf_sol={row['Linf_solid']:.4e}")
        fv_rows.append(row)

    for metric in ['outlet', 'particle', 'solid']:
        compute_eoc(fv_rows, f'L2_{metric}', f'L2EOC_{metric}')
        compute_eoc(fv_rows, f'Linf_{metric}', f'LinfEOC_{metric}')

    # ── DG convergence ──
    dg_rows = {}
    for p in DG_POLY_DEGS:
        dg_rows[p] = []
        print(f"\n--- DG P{p} convergence ---")
        for ne in DG_NELEM:
            result = open_dg(p, ne)
            if result is None:
                print(f"  Skipping DG P{p} Z{ne} (missing/no output)")
                continue
            sol_out, sim_time, polydeg, nelem, par_polydeg, par_nelem, dof_bulk, dof_total, dg_path = result
            l2, linf = compute_errors(sol_out, ref_out)

            # Particle errors vs Z512 reference
            ax_faces = equivolume_faces(r0, r1, nelem)
            test_ax_pos = dg_node_positions(ax_faces, polydeg)
            l2p, linfp, l2s, linfs = compute_particle_errors(
                dg_path, ref_par_data, ref_sol_data, test_ax_pos, par_ref_ax_faces, par_ref_par_faces)

            row = {'POLYDEG': polydeg, 'NELEM': nelem,
                   'PAR_POLYDEG': par_polydeg, 'PAR_NELEM': par_nelem,
                   'DoF_bulk': dof_bulk, 'DoF_total': dof_total,
                   'L2_outlet': l2, 'L2EOC_outlet': 0.0,
                   'Linf_outlet': linf, 'LinfEOC_outlet': 0.0,
                   'L2_particle': l2p, 'L2EOC_particle': 0.0,
                   'Linf_particle': linfp, 'LinfEOC_particle': 0.0,
                   'L2_solid': l2s, 'L2EOC_solid': 0.0,
                   'Linf_solid': linfs, 'LinfEOC_solid': 0.0,
                   'Simtime': sim_time}
            print(f"  DG P{p} Z{ne}: Linf_out={linf:.4e}  Linf_par={linfp:.4e}  Linf_sol={linfs:.4e}  DoF={dof_total}")
            dg_rows[p].append(row)

        for metric in ['outlet', 'particle', 'solid']:
            compute_eoc(dg_rows[p], f'L2_{metric}', f'L2EOC_{metric}')
            compute_eoc(dg_rows[p], f'Linf_{metric}', f'LinfEOC_{metric}')

    # ── CSVs ──
    fv_csv = os.path.join(OUT_DIR, 'rGRM_SMA_FV.csv')
    with open(fv_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['NCOL', 'NCELLS', 'DoF_bulk', 'DoF_total',
                     'L2_outlet', 'L2EOC_outlet', 'Linf_outlet', 'LinfEOC_outlet',
                     'L2_particle', 'L2EOC_particle', 'Linf_particle', 'LinfEOC_particle',
                     'L2_solid', 'L2EOC_solid', 'Linf_solid', 'LinfEOC_solid',
                     'Simtime'])
        for r in fv_rows[:-1]:
            w.writerow([r['NCOL'], r['NCELLS'], r['DoF_bulk'], r['DoF_total'],
                        f"{r['L2_outlet']:.6e}", f"{r['L2EOC_outlet']:.4f}",
                        f"{r['Linf_outlet']:.6e}", f"{r['LinfEOC_outlet']:.4f}",
                        f"{r['L2_particle']:.6e}", f"{r['L2EOC_particle']:.4f}",
                        f"{r['Linf_particle']:.6e}", f"{r['LinfEOC_particle']:.4f}",
                        f"{r['L2_solid']:.6e}", f"{r['L2EOC_solid']:.4f}",
                        f"{r['Linf_solid']:.6e}", f"{r['LinfEOC_solid']:.4f}",
                        f"{r['Simtime']:.3f}"])
    print(f"\nSaved: {fv_csv}")

    dg_csv = os.path.join(OUT_DIR, 'rGRM_SMA_DG.csv')
    with open(dg_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Method', 'POLYDEG', 'NELEM', 'PAR_POLYDEG', 'PAR_NELEM',
                     'DoF_bulk', 'DoF_total',
                     'L2_outlet', 'L2EOC_outlet', 'Linf_outlet', 'LinfEOC_outlet',
                     'L2_particle', 'L2EOC_particle', 'Linf_particle', 'LinfEOC_particle',
                     'L2_solid', 'L2EOC_solid', 'Linf_solid', 'LinfEOC_solid',
                     'Simtime'])
        for p in DG_POLY_DEGS:
            for r in dg_rows[p]:
                w.writerow([f'DG P{p}', r['POLYDEG'], r['NELEM'],
                            r['PAR_POLYDEG'], r['PAR_NELEM'],
                            r['DoF_bulk'], r['DoF_total'],
                            f"{r['L2_outlet']:.6e}", f"{r['L2EOC_outlet']:.4f}",
                            f"{r['Linf_outlet']:.6e}", f"{r['LinfEOC_outlet']:.4f}",
                            f"{r['L2_particle']:.6e}", f"{r['L2EOC_particle']:.4f}",
                            f"{r['Linf_particle']:.6e}", f"{r['LinfEOC_particle']:.4f}",
                            f"{r['L2_solid']:.6e}", f"{r['L2EOC_solid']:.4f}",
                            f"{r['Linf_solid']:.6e}", f"{r['LinfEOC_solid']:.4f}",
                            f"{r['Simtime']:.3f}"])
    print(f"Saved: {dg_csv}")

    # ── Tables ──
    for label, err_suffix in [('Outlet', 'outlet'), ('Particle', 'particle'), ('Solid', 'solid')]:
        print(f"\n{'='*100}")
        print(f"FV WENO3 self-convergence -- {label} (ref = Z{FV_REF})")
        print(f"{'='*100}")
        print(f"{'NCOL':>6} {'NCELLS':>6} {'DoF':>10} "
              f"{'Linf':>12} {'LinfEOC':>8} {'L2':>12} {'L2EOC':>8} {'Time[s]':>10}")
        for r in fv_rows:
            print(f"{r['NCOL']:>6} {r['NCELLS']:>6} {r['DoF_total']:>10} "
                  f"{r[f'Linf_{err_suffix}']:>12.4e} {r[f'LinfEOC_{err_suffix}']:>8.2f} "
                  f"{r[f'L2_{err_suffix}']:>12.4e} {r[f'L2EOC_{err_suffix}']:>8.2f} "
                  f"{r['Simtime']:>10.1f}")

        for p in DG_POLY_DEGS:
            print(f"\n{'='*100}")
            print(f"DG P{p} convergence vs FV Z{FV_REF} -- {label}")
            print(f"{'='*100}")
            print(f"{'Np':>4} {'Ne':>4} {'Par_Np':>6} {'Par_Ne':>6} {'DoF':>10} "
                  f"{'Linf':>12} {'LinfEOC':>8} {'L2':>12} {'L2EOC':>8} {'Time[s]':>10}")
            for r in dg_rows[p]:
                print(f"{r['POLYDEG']:>4} {r['NELEM']:>4} {r['PAR_POLYDEG']:>6} {r['PAR_NELEM']:>6} "
                      f"{r['DoF_total']:>10} "
                      f"{r[f'Linf_{err_suffix}']:>12.4e} {r[f'LinfEOC_{err_suffix}']:>8.2f} "
                      f"{r[f'L2_{err_suffix}']:>12.4e} {r[f'L2EOC_{err_suffix}']:>8.2f} "
                      f"{r['Simtime']:>10.1f}")

    # ── Plots ──
    colors_dg = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728'}

    def make_plot(fv_x, fv_y, dg_x_y, xlabel, ylabel, fname):
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
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        ax.grid(True, which='major', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")

    # Trim floor points from outlet plots: P4 last 2, P3 last 2, P2 last 1
    plot_trim = {1: 0, 2: 1, 3: 2, 4: 2}
    def trimmed(rows, p):
        n = plot_trim.get(p, 0)
        return rows[:len(rows) - n] if n > 0 else rows

    for label, err_suffix in [('Outlet', 'outlet'), ('Particle', 'particle'), ('Solid', 'solid')]:
        fv_plot = [r for r in fv_rows if r[f'Linf_{err_suffix}'] > 0]
        fv_dofs = [r['DoF_total'] for r in fv_plot]
        fv_linf = [r[f'Linf_{err_suffix}'] for r in fv_plot]
        fv_times = [r['Simtime'] for r in fv_plot]

        if err_suffix == 'outlet':
            dg_dof = {p: (np.array([r['DoF_total'] for r in trimmed(dg_rows[p], p)]),
                          np.array([r[f'Linf_{err_suffix}'] for r in trimmed(dg_rows[p], p)])) for p in DG_POLY_DEGS}
            dg_time = {p: (np.array([r['Simtime'] for r in trimmed(dg_rows[p], p)]),
                           np.array([r[f'Linf_{err_suffix}'] for r in trimmed(dg_rows[p], p)])) for p in DG_POLY_DEGS}
        else:
            dg_dof = {p: (np.array([r['DoF_total'] for r in dg_rows[p]]),
                          np.array([r[f'Linf_{err_suffix}'] for r in dg_rows[p]])) for p in DG_POLY_DEGS}
            dg_time = {p: (np.array([r['Simtime'] for r in dg_rows[p]]),
                           np.array([r[f'Linf_{err_suffix}'] for r in dg_rows[p]])) for p in DG_POLY_DEGS}

        tag = label.lower()
        make_plot(fv_dofs, fv_linf, dg_dof,
                  'Degrees of freedom', f'$L^\\infty$ error ({tag})',
                  f'Linf_vs_DoF_{tag}.png')

        make_plot(fv_times, fv_linf, dg_time,
                  'Compute time [s]', f'$L^\\infty$ error ({tag})',
                  f'Linf_vs_compute_{tag}.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
