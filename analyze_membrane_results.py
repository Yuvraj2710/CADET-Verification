#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze membrane chromatography convergence results.

Reads pre-computed h5 files from output/test_cadet-core/radialDG/membrane/
and produces convergence plots + tables for:
  Config A: Dextran T2000 pulse (pure radial transport)
  Config B: NaCl pulse (radial LRMP, no binding)

Uses finest FV WENO3 solution as reference for DG convergence.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import h5py
except ImportError:
    print("ERROR: h5py required. Install with: pip install h5py")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

H5_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'output', 'test_cadet-core', 'radialDG', 'membrane')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'radialDG-results-membrane')

# Config A: pure transport
TRANSPORT_PREFIX = 'radMembrane_transport_1comp'
TRANSPORT_DG_POLYDEGS = [1, 2, 3, 4, 5]
TRANSPORT_DG_NELEMS = [2, 4, 8, 16, 32, 64, 128, 256, 512]
TRANSPORT_FV_NCELLS = [4 * 2**i for i in range(16)]  # 4 to 131072

# Config B: LRMP
LRMP_PREFIX = 'radMembrane_LRMP_1comp'
LRMP_DG_POLYDEGS = [1, 2, 3, 4, 5]
LRMP_DG_NELEMS = [2, 4, 8, 16, 32, 64, 128, 256]
LRMP_FV_NCELLS = [4 * 2**i for i in range(16)]  # 4 to 131072


# ============================================================================
# I/O helpers
# ============================================================================

def read_outlet(h5_path):
    """Read outlet solution, times, and compute time from a CADET h5 file."""
    with h5py.File(h5_path, 'r') as f:
        times = f['output']['solution']['SOLUTION_TIMES'][:]
        outlet = f['output']['solution']['unit_001']['SOLUTION_OUTLET'][:]
        simtime = float(f['meta']['TIME_SIM'][()]) if 'meta' in f and 'TIME_SIM' in f['meta'] else None
    # outlet shape: (nTime, nComp) or (nTime,)
    if outlet.ndim == 2:
        outlet = outlet[:, 0]  # single component
    return times, outlet, simtime


def dg_filename(prefix, polyDeg, nElem):
    return os.path.join(H5_DIR,
                        f'{prefix}_P{polyDeg}_DG_P{polyDeg}Z{nElem}.h5')


def fv_filename(prefix, nCells):
    return os.path.join(H5_DIR, f'{prefix}_FV_Z{nCells}.h5')


# ============================================================================
# Error computation
# ============================================================================

def compute_errors(times, solution, ref_times, ref_solution):
    """Compute L1, L2, Linf errors against a reference solution.

    If time grids differ, interpolate reference to solution times.
    """
    if len(times) != len(ref_times) or not np.allclose(times, ref_times):
        ref_interp = np.interp(times, ref_times, ref_solution)
    else:
        ref_interp = ref_solution

    diff = np.abs(solution - ref_interp)
    dt = np.diff(times)
    # Midpoint values for integration
    diff_mid = 0.5 * (diff[:-1] + diff[1:])

    T = times[-1] - times[0]
    L1 = np.sum(diff_mid * dt) / T
    L2 = np.sqrt(np.sum(diff_mid**2 * dt) / T)
    Linf = np.max(diff)

    return {'L1': L1, 'L2': L2, 'Linf': Linf}


def compute_eoc(disc, errors):
    """Compute experimental order of convergence."""
    eoc = []
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i-1] > 0 and disc[i] != disc[i-1]:
            eoc.append(np.log(errors[i-1] / errors[i]) /
                       np.log(disc[i] / disc[i-1]))
        else:
            eoc.append(np.nan)
    return eoc


def dof_dg(polyDeg, nElem):
    return (polyDeg + 1) * nElem


# ============================================================================
# Analysis for one config
# ============================================================================

def analyze_config(prefix, dg_polydegs, dg_nelems, fv_ncells,
                   config_label, ref_type='fv_finest'):
    """Run convergence analysis for one configuration.

    Returns dict with all results for plotting.
    """
    print(f"\n{'='*60}")
    print(f"  {config_label}")
    print(f"{'='*60}")

    results = {
        'config_label': config_label,
        'dg': {},
        'fv': {},
    }

    # ------------------------------------------------------------------
    # Load FV solutions
    # ------------------------------------------------------------------
    print("\nLoading FV WENO3 solutions...")
    fv_data = {}
    for nc in fv_ncells:
        fname = fv_filename(prefix, nc)
        if not os.path.exists(fname):
            print(f"  WARNING: {os.path.basename(fname)} not found, skipping")
            continue
        try:
            t, sol, st = read_outlet(fname)
            fv_data[nc] = (t, sol, st)
            print(f"  Z{nc}: loaded ({len(t)} time points, simtime={st:.3f}s)")
        except Exception as e:
            print(f"  WARNING: Z{nc} failed: {e}")

    if not fv_data:
        print("  ERROR: No FV data loaded!")
        return results

    # Reference = finest FV
    finest_nc = max(fv_data.keys())
    ref_times, ref_solution, _ = fv_data[finest_nc]
    print(f"\n  Reference: FV Z{finest_nc} "
          f"(max={np.max(ref_solution):.6f}, "
          f"integral={getattr(np, 'trapezoid', np.trapz)(ref_solution, ref_times):.6f})")

    # ------------------------------------------------------------------
    # FV self-convergence (each vs finest)
    # ------------------------------------------------------------------
    print("\n--- FV WENO3 self-convergence (vs finest FV) ---")
    fv_ncells_used = sorted([nc for nc in fv_data.keys() if nc < finest_nc])
    fv_errors = {'L1': [], 'L2': [], 'Linf': [], 'dof': [], 'ncells': [],
                  'simtime': []}

    print(f"{'nCells':>8} {'DOF':>8} {'L1':>12} {'L2':>12} {'Linf':>12} {'Simtime':>10}")
    print("-" * 70)
    for nc in fv_ncells_used:
        t, sol, st = fv_data[nc]
        errs = compute_errors(t, sol, ref_times, ref_solution)
        fv_errors['L1'].append(errs['L1'])
        fv_errors['L2'].append(errs['L2'])
        fv_errors['Linf'].append(errs['Linf'])
        fv_errors['dof'].append(nc)
        fv_errors['ncells'].append(nc)
        fv_errors['simtime'].append(st)
        print(f"{nc:>8d} {nc:>8d} {errs['L1']:>12.4e} "
              f"{errs['L2']:>12.4e} {errs['Linf']:>12.4e} {st:>10.3f}")

    if len(fv_errors['L1']) > 1:
        eoc_L1 = compute_eoc(fv_errors['dof'], fv_errors['L1'])
        eoc_L2 = compute_eoc(fv_errors['dof'], fv_errors['L2'])
        print(f"\n  FV EOC (L1, last): {eoc_L1[-1]:.2f}")
        print(f"  FV EOC (L2, last): {eoc_L2[-1]:.2f}")

    results['fv'] = fv_errors

    # ------------------------------------------------------------------
    # DG convergence (each vs finest FV)
    # ------------------------------------------------------------------
    print(f"\n--- DG convergence (vs FV Z{finest_nc}) ---")

    for p in dg_polydegs:
        print(f"\n  Polynomial degree P{p}:")
        dg_err = {'L1': [], 'L2': [], 'Linf': [], 'dof': [], 'nelem': [],
                  'simtime': []}

        print(f"  {'nElem':>8} {'DOF':>8} {'L1':>12} {'L2':>12} {'Linf':>12} {'Simtime':>10}")
        print("  " + "-" * 70)

        for ne in dg_nelems:
            fname = dg_filename(prefix, p, ne)
            if not os.path.exists(fname):
                continue
            try:
                t, sol, st = read_outlet(fname)
                errs = compute_errors(t, sol, ref_times, ref_solution)
                dof = dof_dg(p, ne)
                dg_err['L1'].append(errs['L1'])
                dg_err['L2'].append(errs['L2'])
                dg_err['Linf'].append(errs['Linf'])
                dg_err['dof'].append(dof)
                dg_err['nelem'].append(ne)
                dg_err['simtime'].append(st)
                print(f"  {ne:>8d} {dof:>8d} {errs['L1']:>12.4e} "
                      f"{errs['L2']:>12.4e} {errs['Linf']:>12.4e} {st:>10.3f}")
            except Exception as e:
                print(f"  {ne:>8d} FAILED: {e}")

        if len(dg_err['L1']) > 1:
            eoc_L1 = compute_eoc(dg_err['dof'], dg_err['L1'])
            eoc_L2 = compute_eoc(dg_err['dof'], dg_err['L2'])
            # Filter out NaN for display
            valid_eoc = [e for e in eoc_L1 if not np.isnan(e)]
            if valid_eoc:
                print(f"  EOC (L1, last): {valid_eoc[-1]:.2f}")
            valid_eoc2 = [e for e in eoc_L2 if not np.isnan(e)]
            if valid_eoc2:
                print(f"  EOC (L2, last): {valid_eoc2[-1]:.2f}")

        results['dg'][p] = dg_err

    return results


# ============================================================================
# Plotting
# ============================================================================

DG_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd'}

ERROR_LABELS = {'L1': r'$L^1$ error (outlet)', 'L2': r'$L^2$ error (outlet)',
                'Linf': r'$L^\infty$ error (outlet)'}


def _trim_negative_eoc(dof, errors):
    """Trim points after the first negative EOC (error increase)."""
    if len(errors) < 2:
        return dof, errors
    for i in range(1, len(errors)):
        if errors[i] >= errors[i - 1]:
            return dof[:i], errors[:i]
    return dof, errors


def plot_convergence(results, error_type='L2', filename_prefix='convergence'):
    """Plot DG + FV convergence on log-log scale."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # FV WENO3
    fv = results['fv']
    if fv['dof']:
        dof_t, err_t = _trim_negative_eoc(fv['dof'], fv[error_type])
        ax.loglog(dof_t, err_t, 'ks--', markersize=5,
                  linewidth=1.2, label='FV WENO3', zorder=2)

    # DG curves
    for p, dg_err in sorted(results['dg'].items()):
        if not dg_err['dof']:
            continue
        color = DG_COLORS.get(p, f'C{p}')
        dof_t, err_t = _trim_negative_eoc(dg_err['dof'], dg_err[error_type])
        ax.loglog(dof_t, err_t, 'o-', color=color,
                  markersize=6, linewidth=1.5, markerfacecolor='white',
                  markeredgewidth=1.5, label=f'DG P{p}', zorder=3)

    ax.set_xlabel('Degrees of freedom', fontsize=14)
    ax.set_ylabel(ERROR_LABELS.get(error_type, f'{error_type} error'), fontsize=14)
    ax.set_title('', fontsize=13)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(True, which='major', alpha=0.3)

    out_path = os.path.join(OUTPUT_DIR, f'{filename_prefix}_{error_type}.png')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_convergence_vs_time(results, error_type='Linf',
                             filename_prefix='convergence'):
    """Plot error vs compute time on log-log scale."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # FV WENO3
    fv = results['fv']
    if fv['simtime']:
        st_t, err_t = _trim_negative_eoc(fv['simtime'], fv[error_type])
        ax.loglog(st_t, err_t, 'ks--', markersize=5,
                  linewidth=1.2, label='FV WENO3', zorder=2)

    # DG curves
    for p, dg_err in sorted(results['dg'].items()):
        if not dg_err['simtime']:
            continue
        color = DG_COLORS.get(p, f'C{p}')
        st_t, err_t = _trim_negative_eoc(dg_err['simtime'], dg_err[error_type])
        ax.loglog(st_t, err_t, 'o-', color=color,
                  markersize=6, linewidth=1.5, markerfacecolor='white',
                  markeredgewidth=1.5, label=f'DG P{p}', zorder=3)

    ax.set_xlabel('Compute time [s]', fontsize=14)
    ax.set_ylabel(ERROR_LABELS.get(error_type, f'{error_type} error'), fontsize=14)
    ax.set_title('', fontsize=13)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(True, which='major', alpha=0.3)

    out_path = os.path.join(OUTPUT_DIR, f'{filename_prefix}_{error_type}_vs_time.png')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_solutions(results_data, prefix, fv_ncells, dg_polydegs, dg_nelems,
                   config_label, filename_prefix='solutions'):
    """Plot coarse FV vs finest DG reference."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Finest DG reference
    finest_p = max(dg_polydegs)
    finest_ne = max(dg_nelems)
    fname_dg = dg_filename(prefix, finest_p, finest_ne)
    t_ref, sol_ref, _ = read_outlet(fname_dg)
    ax.plot(t_ref, sol_ref, 'k-', linewidth=1.5,
            label=f'DG P{finest_p} Z{finest_ne} (ref)')

    # Coarse FV solutions to show differences
    fv_coarse = [4, 8, 16]
    fv_styles = {'4': ('--', '#d62728'), '8': ('--', '#ff7f0e'),
                 '16': ('--', '#1f77b4')}
    for nc in fv_coarse:
        fname = fv_filename(prefix, nc)
        if os.path.exists(fname):
            t, sol, _ = read_outlet(fname)
            ls, col = fv_styles[str(nc)]
            ax.plot(t, sol, ls, color=col, linewidth=1.2, alpha=0.8,
                    label=f'FV Z{nc}')

    ax.set_xlabel('Time [s]', fontsize=14)
    ax.set_ylabel('Outlet concentration [mol/m$^3$]', fontsize=14)
    ax.set_title('', fontsize=13)
    ax.legend(fontsize=11, loc='upper right')
    ax.tick_params(labelsize=12)
    ax.grid(True, which='major', alpha=0.3)

    out_path = os.path.join(OUTPUT_DIR, f'{filename_prefix}.png')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_thesis_solutions():
    """Clean two-panel figure: finest DG solution for both configs, no title."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Config A: finest DG P5 Z512
    fname_a = dg_filename(TRANSPORT_PREFIX, 5, 512)
    t_a, sol_a, _ = read_outlet(fname_a)
    ax1.plot(t_a, sol_a, 'k-', linewidth=1.5)
    ax1.set_xlabel('Time [s]', fontsize=14)
    ax1.set_ylabel('Outlet concentration [mol/m$^3$]', fontsize=14)
    ax1.tick_params(labelsize=12)
    ax1.grid(True, which='major', alpha=0.3)
    # Config B: finest DG P5 Z256
    fname_b = dg_filename(LRMP_PREFIX, 5, 256)
    t_b, sol_b, _ = read_outlet(fname_b)
    ax2.plot(t_b, sol_b, 'k-', linewidth=1.5)
    ax2.set_xlabel('Time [s]', fontsize=14)
    ax2.set_ylabel('Outlet concentration [mol/m$^3$]', fontsize=14)
    ax2.tick_params(labelsize=12)
    ax2.grid(True, which='major', alpha=0.3)

    out_path = os.path.join(OUTPUT_DIR, 'thesis_solutions.png')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_results_json(results, filename):
    """Save results to JSON for later use."""
    # Convert numpy arrays to lists
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"  Saved: {out_path}")


def save_csv_fv(results, filename):
    """Save FV convergence results as CSV."""
    fv = results['fv']
    if not fv['dof']:
        return

    out_path = os.path.join(OUTPUT_DIR, filename)
    has_simtime = bool(fv.get('simtime'))
    with open(out_path, 'w') as f:
        header = "Ne,DoF,L2error,L2EOC,Linferror,LinfEOC"
        if has_simtime:
            header += ",Simtime"
        f.write(header + "\n")
        eoc_L2 = compute_eoc(fv['dof'], fv['L2'])
        eoc_Linf = compute_eoc(fv['dof'], fv['Linf'])
        for i in range(len(fv['dof'])):
            ne = fv['ncells'][i]
            dof = fv['dof'][i]
            l2 = fv['L2'][i]
            linf = fv['Linf'][i]
            l2eoc = eoc_L2[i - 1] if i > 0 else 0.0
            linfeoc = eoc_Linf[i - 1] if i > 0 else 0.0
            line = f"{ne},{dof},{l2:.6e},{l2eoc:.4f},{linf:.6e},{linfeoc:.4f}"
            if has_simtime:
                line += f",{fv['simtime'][i]:.4f}"
            f.write(line + "\n")
    print(f"  Saved: {out_path}")


def save_csv_dg(results, filename):
    """Save DG convergence results as CSV."""
    out_path = os.path.join(OUTPUT_DIR, filename)
    first_p = True
    has_simtime = any(
        bool(results['dg'][p].get('simtime'))
        for p in results['dg']
    )
    with open(out_path, 'w') as f:
        header = "Ne,DoF,L2error,L2EOC,Linferror,LinfEOC"
        if has_simtime:
            header += ",Simtime"
        f.write(header + "\n")
        for p in sorted(results['dg'].keys()):
            dg_err = results['dg'][p]
            if not dg_err['dof']:
                continue
            eoc_L2 = compute_eoc(dg_err['dof'], dg_err['L2'])
            eoc_Linf = compute_eoc(dg_err['dof'], dg_err['Linf'])
            for i in range(len(dg_err['dof'])):
                ne = dg_err['nelem'][i]
                dof = dg_err['dof'][i]
                l2 = dg_err['L2'][i]
                linf = dg_err['Linf'][i]
                l2eoc = eoc_L2[i - 1] if i > 0 else 0.0
                linfeoc = eoc_Linf[i - 1] if i > 0 else 0.0
                line = f"{ne},{dof},{l2:.6e},{l2eoc:.4f},{linf:.6e},{linfeoc:.4f}"
                if has_simtime:
                    line += f",{dg_err['simtime'][i]:.4f}"
                f.write(line + "\n")
    print(f"  Saved: {out_path}")


# ============================================================================
# Main
# ============================================================================

def plot_effio_comparison():
    """Compare our finest DG solutions against Effio et al. (2016) Fig. 3.

    Digitized data from:
      Ladd Effio et al., J. Chromatogr. A 1429 (2016) 142-154, Fig. 3.
      - Fig. 3a: Dextran T2000 pulse (UV 215 nm vs volume)
      - Fig. 3b: NaCl pulse (conductivity vs volume)

    System and capsule void volumes were subtracted in the original.
    CADET time converted to volume: V(mL) = t(s) * F, F = 0.05 mL/s.
    Both curves normalized to peak height = 1 for shape comparison.
    """
    F_mL_per_s = 0.05  # 3 mL/min

    # ----------------------------------------------------------------
    # Hand-digitized from Effio et al. Fig. 3a (Dextran T2000)
    # Volume [mL] vs UV 215 nm [mAU], void-subtracted
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # Hand-digitized from Effio et al. Fig. 3b (NaCl)
    # Volume [mL] vs Conductivity [mS/cm], void-subtracted
    # ----------------------------------------------------------------
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

    # Normalize Effio data to peak = 1
    effio_dex_meas_n = effio_dex_meas_y / effio_dex_meas_y.max()
    effio_dex_sim_n = effio_dex_sim_y / effio_dex_sim_y.max()
    effio_nacl_meas_n = effio_nacl_meas_y / effio_nacl_meas_y.max()
    effio_nacl_sim_n = effio_nacl_sim_y / effio_nacl_sim_y.max()

    # ----------------------------------------------------------------
    # Read our finest DG solutions and convert time to volume
    # ----------------------------------------------------------------
    fname_a = dg_filename(TRANSPORT_PREFIX, 5, 512)
    t_a, sol_a, _ = read_outlet(fname_a)
    V_a = t_a * F_mL_per_s  # convert time [s] to volume [mL]
    sol_a_n = sol_a / sol_a.max()

    fname_b = dg_filename(LRMP_PREFIX, 5, 256)
    t_b, sol_b, _ = read_outlet(fname_b)
    V_b = t_b * F_mL_per_s
    sol_b_n = sol_b / sol_b.max()

    # ----------------------------------------------------------------
    # Plot on volume axis (matching Effio's original Fig. 3)
    # ----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Config A: Dextran
    ax1.plot(V_a, sol_a_n, 'k-', linewidth=1.8, label='CADET rDG P5', zorder=3)
    ax1.plot(effio_dex_sim_V, effio_dex_sim_n, '--', color='#2ca02c',
             linewidth=1.5, label='Effio et al. (sim.)', zorder=2)
    ax1.plot(effio_dex_meas_V, effio_dex_meas_n, 'o', color='#d62728',
             markersize=5, markerfacecolor='none', markeredgewidth=1.2,
             label='Effio et al. (exp.)', zorder=4)
    ax1.set_xlabel('Volume [mL]', fontsize=14)
    ax1.set_ylabel('Normalized outlet signal', fontsize=14)
    ax1.set_title('Config A: Dextran T2000', fontsize=14)
    ax1.set_xlim(0, 4)
    ax1.tick_params(labelsize=12)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, which='major', alpha=0.3)

    # Config B: NaCl
    ax2.plot(V_b, sol_b_n, 'k-', linewidth=1.8, label='CADET rDG P5', zorder=3)
    ax2.plot(effio_nacl_sim_V, effio_nacl_sim_n, '--', color='#2ca02c',
             linewidth=1.5, label='Effio et al. (sim.)', zorder=2)
    ax2.plot(effio_nacl_meas_V, effio_nacl_meas_n, 'o', color='#d62728',
             markersize=5, markerfacecolor='none', markeredgewidth=1.2,
             label='Effio et al. (exp.)', zorder=4)
    ax2.set_xlabel('Volume [mL]', fontsize=14)
    ax2.set_ylabel('Normalized outlet signal', fontsize=14)
    ax2.set_title('Config B: NaCl', fontsize=14)
    ax2.set_xlim(0, 6)
    ax2.tick_params(labelsize=12)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, which='major', alpha=0.3)

    out_path = os.path.join(OUTPUT_DIR, 'effio_comparison.png')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Config A: pure transport
    results_A = analyze_config(
        TRANSPORT_PREFIX,
        TRANSPORT_DG_POLYDEGS,
        TRANSPORT_DG_NELEMS,
        TRANSPORT_FV_NCELLS,
        'Config A: Dextran T2000 (radial transport)'
    )

    print("\nGenerating plots for Config A...")
    for etype in ['L1', 'L2', 'Linf']:
        plot_convergence(results_A, error_type=etype,
                         filename_prefix='configA_transport')
    plot_convergence_vs_time(results_A, error_type='Linf',
                             filename_prefix='configA_transport')

    plot_solutions(results_A, TRANSPORT_PREFIX,
                   TRANSPORT_FV_NCELLS, TRANSPORT_DG_POLYDEGS,
                   TRANSPORT_DG_NELEMS,
                   'Config A: Dextran T2000 (radial transport)',
                   filename_prefix='configA_solutions')

    save_results_json(results_A, 'configA_transport_results.json')
    save_csv_fv(results_A, 'configA_transport_FV.csv')
    save_csv_dg(results_A, 'configA_transport_DG.csv')

    # Config B: LRMP
    results_B = analyze_config(
        LRMP_PREFIX,
        LRMP_DG_POLYDEGS,
        LRMP_DG_NELEMS,
        LRMP_FV_NCELLS,
        'Config B: NaCl (radial LRMP, no binding)'
    )

    print("\nGenerating plots for Config B...")
    for etype in ['L1', 'L2', 'Linf']:
        plot_convergence(results_B, error_type=etype,
                         filename_prefix='configB_LRMP')
    plot_convergence_vs_time(results_B, error_type='Linf',
                             filename_prefix='configB_LRMP')

    plot_solutions(results_B, LRMP_PREFIX,
                   LRMP_FV_NCELLS, LRMP_DG_POLYDEGS,
                   LRMP_DG_NELEMS,
                   'Config B: NaCl (radial LRMP, no binding)',
                   filename_prefix='configB_solutions')

    save_results_json(results_B, 'configB_LRMP_results.json')
    save_csv_fv(results_B, 'configB_LRMP_FV.csv')
    save_csv_dg(results_B, 'configB_LRMP_DG.csv')

    # Thesis figure: both configs, finest DG only
    print("\nGenerating thesis solution figure...")
    plot_thesis_solutions()

    # Comparison against Effio et al. (2016)
    print("\nGenerating Effio et al. comparison figure...")
    plot_effio_comparison()

    print(f"\n{'='*60}")
    print(f"  All results saved to {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
