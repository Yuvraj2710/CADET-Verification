#!/usr/bin/env python3
"""Generate CSV table and convergence plots for rLRM lin varCoeff (Study 2, Config 2).

Produces:
  - CSV: single file with FV rows first, then DG P1-P5
  - Plots: L_inf error vs DoF, L_inf error vs compute time
"""

import json
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'test_cadet-core', 'radialDG')


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    print("Loading data...")

    fv_data = load_json(os.path.join(BASE, 'convergence_radLRM_FV_WENO3_lin_1comp_varCoeff.json'))
    poly_degs = [1, 2, 3, 4, 5]
    dg_data_list = []
    for p in poly_degs:
        dg_data_list.append(load_json(
            os.path.join(BASE, f'convergence_radLRM_DG_lin_1comp_varCoeff_P{p}.json')))

    # --- FV CSV ---
    print("Writing CSVs...")
    header = ['Ne', 'DoF', 'L2error', 'L2EOC', 'Linferror', 'LinfEOC', 'Simtime']

    fv = fv_data['convergence']['FVWENO3nonEq']['outlet']
    fv_csv = os.path.join(OUT_DIR, 'rLRM_lin_varCoeff_FV.csv')
    with open(fv_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(fv['$N_e^z$'])):
            w.writerow([
                int(fv['$N_e^z$'][i]),
                int(fv['Bulk DoF'][i]),
                f"{fv['$L^2$ error'][i]:.6e}",
                f"{fv['$L^2$ EOC'][i]:.4f}",
                f"{fv['Max. error'][i]:.6e}",
                f"{fv['Max. EOC'][i]:.4f}",
                f"{fv['Sim. time'][i]:.3f}",
            ])
    print(f"  Saved: rLRM_lin_varCoeff_FV.csv")

    # --- DG CSV ---
    dg_header = ['Method', 'Ne', 'DoF', 'L2error', 'L2EOC', 'Linferror', 'LinfEOC', 'Simtime']
    dg_csv = os.path.join(OUT_DIR, 'rLRM_lin_varCoeff_DG.csv')
    with open(dg_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(dg_header)
        for idx, p in enumerate(poly_degs):
            d = dg_data_list[idx]['convergence'][f'DG_P{p}']['outlet']
            for i in range(len(d['$N_e^z$'])):
                w.writerow([
                    f'DG P{p}',
                    int(d['$N_e^z$'][i]),
                    int(d['Bulk DoF'][i]),
                    f"{d['$L^2$ error'][i]:.6e}",
                    f"{d['$L^2$ EOC'][i]:.4f}",
                    f"{d['Max. error'][i]:.6e}",
                    f"{d['Max. EOC'][i]:.4f}",
                    f"{d['Sim. time'][i]:.3f}",
                ])
    print(f"  Saved: rLRM_lin_varCoeff_DG.csv")

    # --- Plots ---
    print("Generating plots...")

    colors_dg = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd'}

    # Helper: get valid DG data (before hitting FV floor)
    def get_valid(d, poly_deg, eoc_key='Max. EOC'):
        err = np.array(d['Max. error'])
        eoc = np.array(d[eoc_key])
        valid = np.ones(len(err), dtype=bool)
        for i in range(2, len(err)):
            if eoc[i] < -0.3:
                valid[i:] = False
                break
        # Drop one extra trailing point for P2 and P3 (near floor, EOC already degraded)
        if poly_deg in (2, 3):
            last_valid = np.where(valid)[0]
            if len(last_valid) > 1:
                valid[last_valid[-1]] = False
        return valid

    # --- Plot 1: L_inf vs DoF ---
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # FV
    fv = fv_data['convergence']['FVWENO3nonEq']['outlet']
    ax.loglog(fv['Bulk DoF'], fv['Max. error'], 'ks--', markersize=5, linewidth=1.2,
              label='FV WENO3', zorder=2)

    # DG
    for idx, p in enumerate(poly_degs):
        d = dg_data_list[idx]['convergence'][f'DG_P{p}']['outlet']
        dof = np.array(d['Bulk DoF'])
        err = np.array(d['Max. error'])
        valid = get_valid(d, p)

        ax.loglog(dof[valid], err[valid], 'o-', color=colors_dg[p],
                  markersize=6, linewidth=1.5, markerfacecolor='white',
                  markeredgewidth=1.5, label=f'DG P{p}', zorder=3)

    ax.set_xlabel('Degrees of freedom', fontsize=11)
    ax.set_ylabel('$L^\\infty$ error', fontsize=11)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, which='major', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Linf_vs_DoF.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: Linf_vs_DoF.png")

    # --- Plot 2: L_inf vs compute time ---
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # FV
    ax.loglog(fv['Sim. time'], fv['Max. error'], 'ks--', markersize=5, linewidth=1.2,
              label='FV WENO3', zorder=2)

    # DG
    for idx, p in enumerate(poly_degs):
        d = dg_data_list[idx]['convergence'][f'DG_P{p}']['outlet']
        time = np.array(d['Sim. time'])
        err = np.array(d['Max. error'])
        valid = get_valid(d, p)

        ax.loglog(time[valid], err[valid], 'o-', color=colors_dg[p],
                  markersize=6, linewidth=1.5, markerfacecolor='white',
                  markeredgewidth=1.5, label=f'DG P{p}', zorder=3)

    ax.set_xlabel('Compute time [s]', fontsize=11)
    ax.set_ylabel('$L^\\infty$ error', fontsize=11)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, which='major', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'Linf_vs_compute.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: Linf_vs_compute.png")

    print("\nDone!")


if __name__ == '__main__':
    main()
