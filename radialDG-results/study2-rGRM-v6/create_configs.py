#!/usr/bin/env python3
"""Create h5 input files for rGRM v6 study — v5 dispersion + original pore diffusion.

  COL_DISPERSION:  5.75e-8  -> 5.75e-11  (1000x less, same as v5)
  FILM_DIFFUSION:  6.9e-6   -> 6.9e-8    (100x less, same as v5)
  PORE_DIFFUSION:  unchanged (original base values)
    lin 1comp: 6.07e-10
    SMA 4comp: [7e-10, 6.07e-11, 6.07e-11, 6.07e-11]

Creates for both lin 1comp and SMA 4comp:
  - DG P1-P4, nElem = 4,8,16,32,64,128
  - FV WENO3 Z16,32,64,128,256,512,1024
"""

import sys
import os
import copy
import numpy as np
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import src.benchmark_models.setting_radCol1D_DG_GRM_lin_1comp as setting_GRM_lin
import src.benchmark_models.setting_radCol1D_DG_GRM_SMA_4comp as setting_GRM_SMA

OUTPUT_DIR_LIN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', '..', 'output', 'test_cadet-core', 'radialDG', 'v6-lin')
OUTPUT_DIR_SMA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', '..', 'output', 'test_cadet-core', 'radialDG', 'v6-SMA')

TIME_INTEGRATOR_STRICT = {
    'ABSTOL': 1e-12, 'RELTOL': 1e-10, 'ALGTOL': 1e-10,
    'USE_MODIFIED_NEWTON': False,
    'INIT_STEP_SIZE': 1e-6,
    'MAX_STEPS': 5000000
}

V6_DISPERSION_LIN = {
    'col_dispersion': [5.75e-11],
    'film_diffusion': [6.9e-8],
    # pore_diffusion: keep original 6.07e-10
}

V6_DISPERSION_SMA = {
    'col_dispersion': [5.75e-11] * 4,
    'film_diffusion': [6.9e-8] * 4,
    # pore_diffusion: keep original [7e-10, 6.07e-11, 6.07e-11, 6.07e-11]
}


def grid_radial_equivolume(r0, r1, n):
    r2_faces = np.linspace(r0**2, r1**2, n + 1)
    return np.sqrt(r2_faces)


def write_dict_to_h5(group, d):
    for key, val in d.items():
        GROUP_PREFIXES = ('unit_', 'sec_', 'switch_', 'particle_type_')
        GROUP_NAMES = ('input', 'model', 'solver', 'sections', 'time_integrator',
                       'connections', 'return', 'discretization', 'adsorption', 'weno')
        if isinstance(val, dict) and (key in GROUP_NAMES or any(key.startswith(p) for p in GROUP_PREFIXES)):
            h5key = key
        else:
            h5key = key.upper()

        if isinstance(val, dict):
            sub = group.require_group(h5key)
            write_dict_to_h5(sub, val)
        elif isinstance(val, (list, np.ndarray)):
            arr = np.array(val)
            if h5key in group:
                del group[h5key]
            group.create_dataset(h5key, data=arr)
        elif isinstance(val, str):
            if h5key in group:
                del group[h5key]
            group.create_dataset(h5key, data=np.bytes_(val))
        elif isinstance(val, bool):
            if h5key in group:
                del group[h5key]
            group.create_dataset(h5key, data=int(val))
        elif isinstance(val, (int, float)):
            if h5key in group:
                del group[h5key]
            group.create_dataset(h5key, data=val)


def save_config(config_data, filepath):
    if not filepath.endswith('.h5'):
        filepath += '.h5'
    with h5py.File(filepath, 'w') as f:
        write_dict_to_h5(f, config_data)
    return filepath


def apply_dispersion(unit_cfg, disp):
    unit_cfg['col_dispersion'] = disp['col_dispersion']
    unit_cfg['particle_type_000']['film_diffusion'] = disp['film_diffusion']
    # pore_diffusion left at base value


def enable_particle_output(config_data):
    ret = config_data['input']['return']['unit_001']
    ret['write_solution_particle'] = 1
    ret['write_solution_solid'] = 1


def create_DG_config(base_model, output_dir, prefix, polyDeg, nElem, disp):
    config_name = f'{prefix}_P{polyDeg}_DG_P{polyDeg}Z{nElem}'
    config_data = copy.deepcopy(base_model)

    config_data['input']['solver']['time_integrator'] = TIME_INTEGRATOR_STRICT

    unit_cfg = config_data['input']['model']['unit_001']
    unit_cfg['unit_type'] = 'RADIAL_GENERAL_RATE_MODEL'

    apply_dispersion(unit_cfg, disp)
    enable_particle_output(config_data)

    disc = unit_cfg['discretization']
    disc['SPATIAL_METHOD'] = 'DG'
    disc['POLYDEG'] = polyDeg
    disc['NELEM'] = nElem

    unit_cfg['POLYNOMIAL_INTERPOLATION_NODES'] = 'CGL'

    par_disc = unit_cfg['particle_type_000']['discretization']
    par_nelem = max(1, nElem // 4)
    par_disc['NCELLS'] = par_nelem
    par_disc['PAR_POLYDEG'] = polyDeg
    par_disc['PAR_NELEM'] = par_nelem

    save_config(config_data, os.path.join(output_dir, config_name))
    return config_name


def create_FV_config(base_model, output_dir, prefix, nCol, disp):
    config_name = f'{prefix}_FV_Z{nCol}'
    config_data = copy.deepcopy(base_model)

    config_data['input']['solver']['time_integrator'] = TIME_INTEGRATOR_STRICT

    unit_cfg = config_data['input']['model']['unit_001']
    unit_cfg['unit_type'] = 'RADIAL_GENERAL_RATE_MODEL'

    apply_dispersion(unit_cfg, disp)
    enable_particle_output(config_data)

    disc = unit_cfg['discretization']
    disc['SPATIAL_METHOD'] = 'FV'
    disc['NCOL'] = nCol
    disc['MAX_KRYLOV'] = 0
    disc['MAX_RESTARTS'] = 10
    disc['GS_TYPE'] = 1
    disc['SCHUR_SAFETY'] = 1e-8
    disc['RECONSTRUCTION'] = 'WENO'
    disc['weno'] = {
        'WENO_ORDER': 3,
        'WENO_EPS': 1e-10,
        'BOUNDARY_MODEL': 0
    }

    r0 = unit_cfg['col_radius_inner']
    r1 = unit_cfg['col_radius_outer']
    disc['GRID_FACES'] = grid_radial_equivolume(r0, r1, nCol).tolist()

    par_disc = unit_cfg['particle_type_000']['discretization']
    par_disc['NCELLS'] = max(1, nCol // 4)
    par_disc['PAR_POLYDEG'] = 3
    par_disc['PAR_NELEM'] = 1

    save_config(config_data, os.path.join(output_dir, config_name))
    return config_name


def main():
    poly_degs = [1, 2, 3, 4]
    nelem_list = [4, 8, 16, 32, 64, 128]
    fv_levels = [16, 32, 64, 128, 256, 512, 1024]

    # === LIN 1comp ===
    os.makedirs(OUTPUT_DIR_LIN, exist_ok=True)
    base_lin = setting_GRM_lin.get_model()
    created_lin = []

    print("=== GRM lin 1comp -- v6 (v5 dispersion + original pore diffusion) ===")
    print(f"  COL_DISPERSION = {V6_DISPERSION_LIN['col_dispersion'][0]:.2e}")
    print(f"  FILM_DIFFUSION = {V6_DISPERSION_LIN['film_diffusion'][0]:.2e}")
    print(f"  PORE_DIFFUSION = 6.07e-10 (original)")
    print()

    print("DG configs:")
    for p in poly_degs:
        for ne in nelem_list:
            name = create_DG_config(base_lin, OUTPUT_DIR_LIN, 'radGRM_DG_lin_1comp', p, ne, V6_DISPERSION_LIN)
            created_lin.append(name)
            print(f"  {name}")

    print("\nFV configs:")
    for z in fv_levels:
        name = create_FV_config(base_lin, OUTPUT_DIR_LIN, 'radGRM_FV_WENO3_lin_1comp', z, V6_DISPERSION_LIN)
        created_lin.append(name)
        print(f"  {name}")

    print(f"\nLin total: {len(created_lin)} in {OUTPUT_DIR_LIN}")

    # === SMA 4comp ===
    os.makedirs(OUTPUT_DIR_SMA, exist_ok=True)
    base_sma = setting_GRM_SMA.get_model()
    created_sma = []

    print(f"\n{'='*60}")
    print("=== GRM SMA 4comp -- v6 (v5 dispersion + original pore diffusion) ===")
    print(f"  COL_DISPERSION = {V6_DISPERSION_SMA['col_dispersion'][0]:.2e}")
    print(f"  FILM_DIFFUSION = {V6_DISPERSION_SMA['film_diffusion'][0]:.2e}")
    print(f"  PORE_DIFFUSION = [7e-10, 6.07e-11, 6.07e-11, 6.07e-11] (original)")
    print()

    print("DG configs:")
    for p in poly_degs:
        for ne in nelem_list:
            name = create_DG_config(base_sma, OUTPUT_DIR_SMA, 'radGRM_DG_SMA_4comp', p, ne, V6_DISPERSION_SMA)
            created_sma.append(name)
            print(f"  {name}")

    print("\nFV configs:")
    for z in fv_levels:
        name = create_FV_config(base_sma, OUTPUT_DIR_SMA, 'radGRM_FV_WENO3_SMA_4comp', z, V6_DISPERSION_SMA)
        created_sma.append(name)
        print(f"  {name}")

    print(f"\nSMA total: {len(created_sma)} in {OUTPUT_DIR_SMA}")
    print(f"\nGrand total: {len(created_lin) + len(created_sma)} configs")


if __name__ == '__main__':
    main()
