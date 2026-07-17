#!/usr/bin/env python3
"""Regenerate Study 1 CGL vs LGL plots with white-filled circle markers."""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd', 6: '#8c564b'}


def load_transport_csv():
    """Load pure_transport_eoc_table.csv → dict keyed by P."""
    data = {}
    with open(os.path.join(OUT_DIR, 'pure_transport_eoc_table.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = int(row['P'])
            if p not in data:
                data[p] = {'DoF': [], 'CGL_Linf': [], 'CGL_Time': [],
                           'LGL_Linf': [], 'LGL_Time': [], 'Nelem': []}
            data[p]['DoF'].append(int(row['DoF']))
            data[p]['Nelem'].append(int(row['Nelem']))
            data[p]['CGL_Linf'].append(float(row['CGL_Linf']))
            data[p]['CGL_Time'].append(float(row['CGL_Time']))
            data[p]['LGL_Linf'].append(float(row['LGL_Linf']))
            data[p]['LGL_Time'].append(float(row['LGL_Time']))
    return data


def load_grm_sma_json(drop_first=0):
    """Load rGRM SMA data directly from convergence JSONs.

    drop_first : int
        Number of leading data points to skip per polydeg.
    """
    import json
    grm_dir = os.path.join(os.path.dirname(OUT_DIR), 'radialDG-results-study1-GRM-v2')
    data = {}
    for p in range(1, 6):
        cgl_path = os.path.join(grm_dir, f'convergence_radGRM_DG_CGL_SMA_4comp_P{p}.json')
        lgl_path = os.path.join(grm_dir, f'convergence_radGRM_DG_LGL_SMA_4comp_P{p}.json')
        if not os.path.exists(cgl_path):
            continue
        with open(cgl_path) as f:
            cgl = json.load(f)['convergence'][f'DG_P{p}']['outlet']
        with open(lgl_path) as f:
            lgl = json.load(f)['convergence'][f'DG_P{p}']['outlet']
        nelem_list = cgl['$N_e^z$']
        data[p] = {'DoF': [], 'CGL_Linf': [], 'CGL_Time': [],
                   'LGL_Linf': [], 'LGL_Time': [], 'nElem': []}
        for i, ne in enumerate(nelem_list):
            if i < drop_first:
                continue
            ne = int(ne)
            dof = ne * (p + 1) * 4
            data[p]['DoF'].append(dof)
            data[p]['nElem'].append(ne)
            data[p]['CGL_Linf'].append(cgl['Max. error'][i])
            data[p]['CGL_Time'].append(cgl['Sim. time'][i])
            data[p]['LGL_Linf'].append(lgl['Max. error'][i])
            data[p]['LGL_Time'].append(lgl['Sim. time'][i])
    return data


def load_binding_csv():
    """Load linear_binding_eoc_table.csv → dict keyed by P."""
    data = {}
    with open(os.path.join(OUT_DIR, 'linear_binding_eoc_table.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = int(row['P'])
            if p not in data:
                data[p] = {'DoF': [], 'CGL_Linf': [], 'CGL_Time': [],
                           'LGL_Linf': [], 'LGL_Time': [], 'nCells': []}
            data[p]['DoF'].append(int(row['DoF']))
            data[p]['nCells'].append(int(row['nCells']))
            data[p]['CGL_Linf'].append(float(row['CGL_Linf']))
            data[p]['CGL_Time'].append(float(row['CGL_Time']))
            data[p]['LGL_Linf'].append(float(row['LGL_Linf']))
            data[p]['LGL_Time'].append(float(row['LGL_Time']))
    return data


def plot_linf_vs_dof(data, poly_degs, title_prefix, fname,
                     drop_last=None):
    """L_inf error vs DoF — only CGL (CGL ≈ LGL in error).

    drop_last : list of P values whose last data point should be dropped.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for p in poly_degs:
        dof = np.array(data[p]['DoF'])
        err = np.array(data[p]['CGL_Linf'])
        if drop_last and p in drop_last:
            dof, err = dof[:-1], err[:-1]
        ax.loglog(dof, err, 'o-', color=COLORS[p],
                  markersize=6, linewidth=1.5,
                  markerfacecolor='white', markeredgewidth=1.5,
                  label=f'P{p}')
    ax.set_xlabel('DoF', fontsize=13)
    ax.set_ylabel('$L^\\infty$ error', fontsize=13)
    ax.legend(fontsize=11, loc='lower left')
    ax.tick_params(labelsize=11)
    ax.grid(True, which='major', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def plot_reldiff_vs_dof(data, poly_degs, title_prefix, fname):
    """Relative time difference (CGL - LGL) / LGL * 100 vs DoF."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for p in poly_degs:
        dof = np.array(data[p]['DoF'])
        cgl_t = np.array(data[p]['CGL_Time'])
        lgl_t = np.array(data[p]['LGL_Time'])
        reldiff = (cgl_t - lgl_t) / lgl_t * 100
        ax.semilogx(dof, reldiff, 'o-', color=COLORS[p],
                     markersize=6, linewidth=1.5,
                     markerfacecolor='white', markeredgewidth=1.5,
                     label=f'P{p}')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('DoF', fontsize=13)
    ax.set_ylabel(r'$\frac{t_{\rm CGL} - t_{\rm LGL}}{t_{\rm LGL}} \times 100$  (%)',
                  fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(labelsize=11)
    ax.grid(True, which='major', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def plot_timeratio_vs_dof(data, poly_degs, title_prefix, fname,
                          skip_points=None, exclude_degs=None):
    """CGL / LGL time ratio vs DoF.

    skip_points : dict {P: [indices to drop]}, optional
    exclude_degs : list of P values to exclude entirely, optional
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for p in poly_degs:
        if exclude_degs and p in exclude_degs:
            continue
        dof = np.array(data[p]['DoF'])
        cgl_t = np.array(data[p]['CGL_Time'])
        lgl_t = np.array(data[p]['LGL_Time'])
        ratio = cgl_t / lgl_t
        if skip_points and p in skip_points:
            mask = np.ones(len(dof), dtype=bool)
            for idx in skip_points[p]:
                mask[idx] = False
            dof, ratio = dof[mask], ratio[mask]
        ax.semilogx(dof, ratio, 'o-', color=COLORS[p],
                     markersize=6, linewidth=1.5,
                     markerfacecolor='white', markeredgewidth=1.5,
                     label=f'P{p}')
    ax.axhline(1, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('DoF', fontsize=13)
    ax.set_ylabel('CGL / LGL time ratio', fontsize=13)
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(labelsize=11)
    ax.grid(True, which='major', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def main():
    # Pure transport
    transport = load_transport_csv()
    poly_transport = sorted(transport.keys())
    plot_linf_vs_dof(transport, poly_transport, 'Pure transport',
                     'pure_transport_Linf_vs_dof.png',
                     drop_last=[4])
    plot_reldiff_vs_dof(transport, poly_transport, 'Pure transport',
                        'pure_transport_reldiff_vs_dof.png')
    plot_timeratio_vs_dof(transport, poly_transport, 'Pure transport',
                          'pure_transport_timeratio_vs_dof.png',
                          exclude_degs=[4])

    # Linear binding
    binding = load_binding_csv()
    poly_binding = sorted(binding.keys())
    plot_linf_vs_dof(binding, poly_binding, 'Linear binding',
                     'linear_binding_Linf_vs_dof.png')
    plot_reldiff_vs_dof(binding, poly_binding, 'Linear binding',
                        'linear_binding_reldiff_vs_dof.png')
    plot_timeratio_vs_dof(binding, poly_binding, 'Linear binding',
                          'linear_binding_timeratio_vs_dof.png',
                          skip_points={3: [2]})

    # rGRM SMA
    grm_sma_trimmed = load_grm_sma_json(drop_first=3)
    grm_sma_full = load_grm_sma_json(drop_first=0)
    poly_grm = sorted(grm_sma_trimmed.keys())
    grm_sma_drop2 = load_grm_sma_json(drop_first=2)
    plot_linf_vs_dof(grm_sma_drop2, poly_grm, 'rGRM SMA',
                     'grm_sma_Linf_vs_dof.png')
    plot_reldiff_vs_dof(grm_sma_full, poly_grm, 'rGRM SMA',
                        'grm_sma_reldiff_vs_dof.png')
    plot_timeratio_vs_dof(grm_sma_full, poly_grm, 'rGRM SMA',
                          'grm_sma_timeratio_vs_dof.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
