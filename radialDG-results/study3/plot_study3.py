#!/usr/bin/env python3
"""
Study 3 plotting: Langmuir 2-comp oscillation study (radial LRM).

Reproduces plots from Breuer thesis Section 5.3:
  - Figure 5.14 style: Outlet profiles showing oscillations at different Ne
  - Figure 5.15 style: Error vs DOFs and Error vs Compute time (benchmark)
  - Figure 5.16 style: Minimum (most negative) concentration vs DOFs
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
H5_DIR = os.path.join(STUDY_DIR, 'h5')
OUT_DIR = os.path.join(STUDY_DIR, 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

# Thesis-style colors and markers
COLORS = {
    'FV':  '#000000',   # black, filled squares
    'P3':  '#0000FF',   # blue
    'P4':  '#FF8C00',   # orange
    'P5':  '#FF00FF',   # magenta
}
MARKERS = {'FV': 's', 'P3': 'o', 'P4': 'o', 'P5': 'o'}
LABELS = {'FV': 'FV WENO', 'P3': 'DG P3', 'P4': 'DG P4', 'P5': 'DG P5'}

DISP_LABELS = {'D1e4': r'$D_\mathrm{ax} = 10^{-4}$', 'D1e5': r'$D_\mathrm{ax} = 10^{-5}$'}


def error_ylabel(error_key):
    if error_key == 'Max. error':
        return r'$L^\infty$ error in time'
    return f'{error_key.strip("$")} error in time'


def error_filetag(error_key):
    if error_key == 'Max. error':
        return 'Linf_error'
    return error_key.strip("$").replace(" ", "_").replace(".", "")


def load_json(fname):
    with open(os.path.join(STUDY_DIR, fname)) as f:
        return json.load(f)


def get_conv_data(jdata):
    """Extract convergence data from a JSON dict (first key)."""
    key = list(jdata['convergence'].keys())[0]
    d = jdata['convergence'][key]['outlet']
    return d


def compute_dofs(ne, nd):
    """DOFs for DG: Ne*(Np+1), for FV: Ne."""
    nd_val = int(nd[0]) if isinstance(nd, list) else int(nd)
    if nd_val == 0:
        return np.array(ne)
    return np.array(ne) * (nd_val + 1)


# =========================================================================
#  Plot 1: Error vs DOFs  (Breuer Fig 5.15 right panel style)
# =========================================================================
def plot_error_vs_dof(disp_label, error_key='$L^1$ error'):
    fig, ax = plt.subplots(figsize=(6, 5))

    # FV
    try:
        fv = get_conv_data(load_json(f'convergence_radLRM_FV_WENO3_lang_2comp_{disp_label}.json'))
        ne_fv = np.array(fv['$N_e^z$'])
        err_fv = np.array(fv[error_key])
        dof_fv = ne_fv  # FV DOFs = Ne
        ax.loglog(dof_fv, err_fv, color=COLORS['FV'], marker=MARKERS['FV'],
                  markersize=6, linewidth=1.5, label=LABELS['FV'], markerfacecolor=COLORS['FV'])
    except FileNotFoundError:
        pass

    # DG P3, P4, P5
    for p in [3, 4, 5]:
        pk = f'P{p}'
        try:
            dg = get_conv_data(load_json(f'convergence_radLRM_DG_lang_2comp_{disp_label}_{pk}.json'))
            ne = np.array(dg['$N_e^z$'])
            err = np.array(dg[error_key])
            dof = compute_dofs(ne, dg['$N_d$'])
            ax.loglog(dof, err, color=COLORS[pk], marker=MARKERS[pk],
                      markersize=6, linewidth=1.5, label=LABELS[pk],
                      markerfacecolor='white', markeredgecolor=COLORS[pk])
        except FileNotFoundError:
            pass

    ax.set_xlabel('DOFs', fontsize=18)
    ax.set_ylabel(error_ylabel(error_key), fontsize=18)

    ax.legend(fontsize=16)
    ax.grid(True, which='major', alpha=0.3)
    ax.tick_params(labelsize=16)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT_DIR, f'study3_{disp_label}_error_vs_dof_{error_filetag(error_key)}.{ext}'), dpi=200)
    plt.close(fig)


# =========================================================================
#  Plot 2: Error vs Compute time  (Breuer Fig 5.15 left panel style)
# =========================================================================
def plot_error_vs_time(disp_label, error_key='$L^1$ error'):
    fig, ax = plt.subplots(figsize=(6, 5))

    # FV
    try:
        fv = get_conv_data(load_json(f'convergence_radLRM_FV_WENO3_lang_2comp_{disp_label}.json'))
        t_fv = np.array(fv['Sim. time'])
        err_fv = np.array(fv[error_key])
        ax.loglog(t_fv, err_fv, color=COLORS['FV'], marker=MARKERS['FV'],
                  markersize=6, linewidth=1.5, label=LABELS['FV'], markerfacecolor=COLORS['FV'])
    except FileNotFoundError:
        pass

    for p in [3, 4, 5]:
        pk = f'P{p}'
        try:
            dg = get_conv_data(load_json(f'convergence_radLRM_DG_lang_2comp_{disp_label}_{pk}.json'))
            t = np.array(dg['Sim. time'])
            err = np.array(dg[error_key])
            ax.loglog(t, err, color=COLORS[pk], marker=MARKERS[pk],
                      markersize=6, linewidth=1.5, label=LABELS[pk],
                      markerfacecolor='white', markeredgecolor=COLORS[pk])
        except FileNotFoundError:
            pass

    ax.set_xlabel('Compute time in seconds', fontsize=18)
    ax.set_ylabel(error_ylabel(error_key), fontsize=18)

    ax.legend(fontsize=16)
    ax.grid(True, which='major', alpha=0.3)
    ax.tick_params(labelsize=16)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT_DIR, f'study3_{disp_label}_error_vs_time_{error_filetag(error_key)}.{ext}'), dpi=200)
    plt.close(fig)


# =========================================================================
#  Plot 3: Combined benchmark (side-by-side like Breuer Fig 5.15)
# =========================================================================
def plot_benchmark(disp_label, error_key='$L^1$ error'):
    fig, (ax_t, ax_d) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, x_key, x_label in [
        (ax_t, 'Sim. time', 'Compute time in seconds'),
        (ax_d, 'DOF', 'DOFs'),
    ]:
        # FV
        try:
            fv = get_conv_data(load_json(f'convergence_radLRM_FV_WENO3_lang_2comp_{disp_label}.json'))
            if x_key == 'DOF':
                x_fv = np.array(fv.get('DoF', fv['$N_e^z$']))
            else:
                x_fv = np.array(fv[x_key])
            err_fv = np.array(fv[error_key])
            ax.loglog(x_fv, err_fv, color=COLORS['FV'], marker=MARKERS['FV'],
                      markersize=6, linewidth=1.5, label=LABELS['FV'], markerfacecolor=COLORS['FV'])
        except FileNotFoundError:
            pass

        for p_val in [3, 4, 5]:
            pk = f'P{p_val}'
            try:
                dg = get_conv_data(load_json(f'convergence_radLRM_DG_lang_2comp_{disp_label}_{pk}.json'))
                if x_key == 'DOF':
                    x = np.array(dg.get('DoF', compute_dofs(dg['$N_e^z$'], dg['$N_d$'])))
                else:
                    x = np.array(dg[x_key])
                err = np.array(dg[error_key])
                ax.loglog(x, err, color=COLORS[pk], marker=MARKERS[pk],
                          markersize=6, linewidth=1.5, label=LABELS[pk],
                          markerfacecolor='white', markeredgecolor=COLORS[pk])
            except FileNotFoundError:
                pass

        ax.set_xlabel(x_label, fontsize=18)
        ax.set_ylabel(error_ylabel(error_key), fontsize=18)
        ax.legend(fontsize=15)
        ax.grid(True, which='major', alpha=0.3)
        ax.tick_params(labelsize=16)


    fig.tight_layout()
    err_tag = error_filetag(error_key)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT_DIR, f'study3_{disp_label}_benchmark_{err_tag}.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)


# =========================================================================
#  Plot 4: Minimum negative concentration vs DOFs  (Breuer Fig 5.16 style)
# =========================================================================
def plot_min_value(disp_label):
    fig, ax = plt.subplots(figsize=(6, 5))

    all_plotted = []  # collect all plotted abs min values for y-limit

    # FV
    try:
        fv = get_conv_data(load_json(f'convergence_radLRM_FV_WENO3_lang_2comp_{disp_label}.json'))
        dof_fv = np.array(fv.get('DoF', fv['$N_e^z$']))
        min_fv = np.abs(np.array(fv['Min. value']))
        # Replace numerical noise (< 1e-20) with 1e-13 for plotting
        min_fv[min_fv < 1e-20] = 1e-13
        mask = min_fv > 0
        if mask.any():
            ax.loglog(dof_fv[mask], min_fv[mask], color=COLORS['FV'], marker=MARKERS['FV'],
                      markersize=6, linewidth=1.5, label=LABELS['FV'], markerfacecolor=COLORS['FV'])
            all_plotted.extend(min_fv[mask].tolist())
    except (FileNotFoundError, KeyError):
        pass

    for p in [3, 4, 5]:
        pk = f'P{p}'
        try:
            dg = get_conv_data(load_json(f'convergence_radLRM_DG_lang_2comp_{disp_label}_{pk}.json'))
            dof = np.array(dg.get('DoF', compute_dofs(dg['$N_e^z$'], dg['$N_d$'])))
            min_v = np.abs(np.array(dg['Min. value']))
            mask = min_v > 0
            if mask.any():
                ax.loglog(dof[mask], min_v[mask], color=COLORS[pk], marker=MARKERS[pk],
                          markersize=6, linewidth=1.5, label=LABELS[pk],
                          markerfacecolor='white', markeredgecolor=COLORS[pk])
                all_plotted.extend(min_v[mask].tolist())
        except (FileNotFoundError, KeyError):
            pass

    # Set y-axis limits
    if disp_label == 'D1e4':
        ax.set_ylim(1e-14, 1e1)
    elif all_plotted:
        min_exp = int(np.floor(np.log10(min(all_plotted))))
        ax.set_ylim(10 ** min_exp, 1e1)

    ax.set_xlabel('DOFs', fontsize=18)
    ax.set_ylabel('Largest negative value', fontsize=18)

    ax.legend(fontsize=16)
    ax.grid(True, which='major', alpha=0.3)
    ax.tick_params(labelsize=16)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT_DIR, f'study3_{disp_label}_min_value.{ext}'), dpi=200)
    plt.close(fig)


# =========================================================================
#  Plot 5: Outlet profiles at different Ne  (Breuer Fig 5.14 style)
# =========================================================================
def plot_outlet_profiles(disp_label='D1e5'):
    try:
        import h5py
    except ImportError:
        print("h5py not available, skipping outlet profile plots")
        return

    # DG P3 at different Ne + "exact" (highest-res DG P3)
    # Upper row: DG P3 vs Exact at Ne=32, 64, 128
    # Lower row: FV vs Exact at Ne=64, 128, 256
    dg_files = {
        32:  f'radLRM_DG_lang_2comp_{disp_label}_P3_DG_P3Z32.h5',
        64:  f'radLRM_DG_lang_2comp_{disp_label}_P3_DG_P3Z64.h5',
        128: f'radLRM_DG_lang_2comp_{disp_label}_P3_DG_P3Z128.h5',
    }
    fv_files = {
        64:  f'radLRM_FV_WENO3_lang_2comp_{disp_label}_FV_Z64.h5',
        128: f'radLRM_FV_WENO3_lang_2comp_{disp_label}_FV_Z128.h5',
        256: f'radLRM_FV_WENO3_lang_2comp_{disp_label}_FV_Z256.h5',
    }
    ref_file = f'radLRM_DG_lang_2comp_{disp_label}_P3_DG_P3Z1024.h5'

    # Load reference
    ref_path = os.path.join(H5_DIR, ref_file)
    if not os.path.exists(ref_path):
        print(f"Reference file not found: {ref_path}, skipping outlet profiles")
        return

    with h5py.File(ref_path, 'r') as f:
        t_ref = f['output/solution/SOLUTION_TIMES'][:]
        c_ref = f['output/solution/unit_001/SOLUTION_OUTLET'][:]  # (Nt, Ncomp)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Top row: DG P3
    for col, ne in enumerate([32, 64, 128]):
        ax = axes[0, col]
        fpath = os.path.join(H5_DIR, dg_files[ne])
        if not os.path.exists(fpath):
            ax.text(0.5, 0.5, f'File not found:\n{dg_files[ne]}',
                    transform=ax.transAxes, ha='center', va='center')
            continue
        with h5py.File(fpath, 'r') as f:
            t = f['output/solution/SOLUTION_TIMES'][:]
            c = f['output/solution/unit_001/SOLUTION_OUTLET'][:]

        # Plot both components
        ax.plot(t, c[:, 0], color='red', linewidth=1.0, label='DG P3')
        ax.plot(t, c[:, 1], color='red', linewidth=1.0)
        ax.plot(t_ref, c_ref[:, 0], color='black', linewidth=0.8, linestyle='-', label='Exact')
        ax.plot(t_ref, c_ref[:, 1], color='black', linewidth=0.8, linestyle='-')

        ax.set_title(f'$N_e = {ne}$', fontsize=17)
        ax.set_xlabel('time / sec', fontsize=16)
        if col == 0:
            ax.set_ylabel(r'Outlet concentration / mol $\cdot$ m$^{-3}$', fontsize=15)
        ax.legend(fontsize=14, loc='upper right')
        ax.set_xlim(0, 40)
        ax.tick_params(labelsize=15)

    # Bottom row: FV
    for col, ne in enumerate([64, 128, 256]):
        ax = axes[1, col]
        fpath = os.path.join(H5_DIR, fv_files[ne])
        if not os.path.exists(fpath):
            ax.text(0.5, 0.5, f'File not found:\n{fv_files[ne]}',
                    transform=ax.transAxes, ha='center', va='center')
            continue
        with h5py.File(fpath, 'r') as f:
            t = f['output/solution/SOLUTION_TIMES'][:]
            c = f['output/solution/unit_001/SOLUTION_OUTLET'][:]

        ax.plot(t, c[:, 0], color='blue', linewidth=1.0, label='FV')
        ax.plot(t, c[:, 1], color='blue', linewidth=1.0)
        ax.plot(t_ref, c_ref[:, 0], color='black', linewidth=0.8, linestyle='-', label='Exact')
        ax.plot(t_ref, c_ref[:, 1], color='black', linewidth=0.8, linestyle='-')

        ax.set_title(f'$N_e = {ne}$', fontsize=17)
        ax.set_xlabel('time / sec', fontsize=16)
        if col == 0:
            ax.set_ylabel(r'Outlet concentration / mol $\cdot$ m$^{-3}$', fontsize=15)
        ax.legend(fontsize=14, loc='upper right')
        ax.set_xlim(0, 40)
        ax.tick_params(labelsize=15)


    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT_DIR, f'study3_{disp_label}_outlet_profiles.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)


# =========================================================================
#  Plot 5b: High-resolution outlet profiles (2x1: D1e4 and D1e5)
# =========================================================================
def plot_highres_profiles():
    try:
        import h5py
    except ImportError:
        print("h5py not available, skipping high-res profile plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for col, (label, title) in enumerate([
        ('D1e4', r'$D_\mathrm{ax} = 10^{-4}$'),
        ('D1e5', r'$D_\mathrm{ax} = 10^{-5}$'),
    ]):
        ax = axes[col]
        # Use finest available P5 reference
        t, c = None, None
        for candidate in [
            f'radLRM_DG_lang_2comp_{label}_P5_DG_P5Z1024.h5',
            f'radLRM_DG_lang_2comp_{label}_P5_DG_P5Z256.h5',
        ]:
            fpath = os.path.join(H5_DIR, candidate)
            if not os.path.exists(fpath):
                continue
            with h5py.File(fpath, 'r') as f:
                if 'output/solution/SOLUTION_TIMES' in f:
                    t = f['output/solution/SOLUTION_TIMES'][:]
                    c = f['output/solution/unit_001/SOLUTION_OUTLET'][:]
                    break
        if t is None:
            ax.text(0.5, 0.5, f'No reference file found for {label}',
                    transform=ax.transAxes, ha='center', va='center')
            continue

        ax.plot(t, c[:, 0], color='goldenrod', linewidth=1.2, label='Comp. 1')
        ax.plot(t, c[:, 1], color='blue', linewidth=1.2, label='Comp. 2')

        ax.set_xlabel('Time / s', fontsize=18)
        if col == 0:
            ax.set_ylabel(r'Outlet concentration / mol $\cdot$ m$^{-3}$', fontsize=18)
        ax.legend(fontsize=16, loc='upper right')
        ax.set_xlim(0, 40)
        ax.set_xticks(np.arange(0, 45, 5))
        ax.grid(True, which='major', alpha=0.3)
        ax.tick_params(labelsize=16)

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT_DIR, f'study3_highres_profiles.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)


# =========================================================================
#  Plot 6: Convergence table (CSV + LaTeX)
# =========================================================================
def write_convergence_table(disp_label):
    import csv

    fieldnames = ['Method', 'Ne', 'Linf_error', 'Linf_EOC', 'L1_error', 'L1_EOC', 'Min_value', 'Sim_time']
    rows = []
    # DG
    for p in [3, 4, 5]:
        pk = f'P{p}'
        try:
            dg = get_conv_data(load_json(f'convergence_radLRM_DG_lang_2comp_{disp_label}_{pk}.json'))
            ne_list = dg['$N_e^z$']
            for i, ne in enumerate(ne_list):
                rows.append({
                    'Method': f'DG P{p}',
                    'Ne': int(ne),
                    'Linf_error': dg['Max. error'][i],
                    'Linf_EOC': dg['Max. EOC'][i],
                    'L1_error': dg['$L^1$ error'][i],
                    'L1_EOC': dg['$L^1$ EOC'][i],
                    'Min_value': dg['Min. value'][i],
                    'Sim_time': dg['Sim. time'][i],
                })
        except FileNotFoundError:
            pass

    # FV
    try:
        fv = get_conv_data(load_json(f'convergence_radLRM_FV_WENO3_lang_2comp_{disp_label}.json'))
        ne_list = fv['$N_e^z$']
        for i, ne in enumerate(ne_list):
            rows.append({
                'Method': 'FV WENO',
                'Ne': int(ne),
                'Linf_error': fv['Max. error'][i],
                'Linf_EOC': fv['Max. EOC'][i],
                'L1_error': fv['$L^1$ error'][i],
                'L1_EOC': fv['$L^1$ EOC'][i],
                'Min_value': fv['Min. value'][i],
                'Sim_time': fv['Sim. time'][i],
            })
    except FileNotFoundError:
        pass

    # CSV
    csv_path = os.path.join(OUT_DIR, f'study3_{disp_label}_convergence.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # LaTeX table
    tex_path = os.path.join(OUT_DIR, f'study3_{disp_label}_table.tex')
    with open(tex_path, 'w') as f:
        f.write(r'\begin{tabular}{l r c c c c c r}' + '\n')
        f.write(r'\toprule' + '\n')
        f.write(r'Method & $N_e^z$ & $L^\infty$ error & $L^\infty$ EOC & $L^1$ error & $L^1$ EOC & Min. value & Compute time \\' + '\n')
        f.write(r'\midrule' + '\n')
        prev_method = None
        for r in rows:
            if r['Method'] != prev_method and prev_method is not None:
                f.write(r'\midrule' + '\n')
            prev_method = r['Method']
            eoc_linf = f"{r['Linf_EOC']:.2f}" if r['Linf_EOC'] != 0 else '--'
            eoc_l1 = f"{r['L1_EOC']:.2f}" if r['L1_EOC'] != 0 else '--'
            f.write(f"  {r['Method']} & {r['Ne']} & "
                    f"{r['Linf_error']:.2e} & {eoc_linf} & "
                    f"{r['L1_error']:.2e} & {eoc_l1} & "
                    f"{r['Min_value']:.2e} & {r['Sim_time']:.1f} "
                    r'\\' + '\n')
        f.write(r'\bottomrule' + '\n')
        f.write(r'\end{tabular}' + '\n')

    print(f"  Written: {csv_path}")
    print(f"  Written: {tex_path}")


# =========================================================================
#  Main
# =========================================================================
if __name__ == '__main__':
    for disp in ['D1e4', 'D1e5']:
        print(f"\n=== {disp} ===")

        # Benchmark plots (error vs DOF, error vs time) — Fig 5.15 style
        for err_key in ['Max. error']:
            print(f"  Plotting benchmark ({err_key})...")
            plot_benchmark(disp, error_key=err_key)
            plot_error_vs_dof(disp, error_key=err_key)
            plot_error_vs_time(disp, error_key=err_key)

        # Min negative value — Fig 5.16 style
        print("  Plotting min value...")
        plot_min_value(disp)

        # Convergence tables
        print("  Writing convergence tables...")
        write_convergence_table(disp)

    # Outlet profiles (Fig 5.14 style) — only for D1e5 (most oscillations)
    print("\n=== Outlet profiles (D1e5) ===")
    plot_outlet_profiles('D1e5')

    # High-res reference profiles (DG P10, Ne=300)
    print("\n=== High-resolution profiles ===")
    plot_highres_profiles()

    print(f"\nAll plots saved to: {OUT_DIR}")
