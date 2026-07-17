#!/usr/bin/env python3
"""Create h5 input files for rGRM v4 study — SMA 4comp, very low dispersion.

  COL_DISPERSION:  5.75e-8  -> 5.75e-12  (10000x less, per component)
  FILM_DIFFUSION:  6.9e-6   -> 6.9e-9    (1000x less, per component)
  PORE_DIFFUSION:  [7e-10, 6.07e-11, 6.07e-11, 6.07e-11]
                -> [7e-13, 6.07e-14, 6.07e-14, 6.07e-14]  (1000x less)

Creates:
  GRM SMA 4comp:
    - DG P1-P4, nElem = 4,8,16,32,64,128
    - FV WENO3 Z16,32,64,128,256,512,1024
"""

import sys
import os
import copy
import numpy as np
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import src.benchmark_models.setting_radCol1D_DG_GRM_SMA_4comp as setting_GRM_SMA

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', '..', 'output', 'test_cadet-core', 'radialDG', 'v4-SMA')

TIME_INTEGRATOR_STRICT = {
    'ABSTOL': 1e-12, 'RELTOL': 1e-10, 'ALGTOL': 1e-10,
    'USE_MODIFIED_NEWTON': False,
    'INIT_STEP_SIZE': 1e-6,
    'MAX_STEPS': 5000000
}

NCOMP = 4

V4_DISPERSION = {
    'col_dispersion': [5.75e-12] * NCOMP,
    'film_diffusion': [6.9e-9] * NCOMP,
    'pore_diffusion': [7e-13, 6.07e-14, 6.07e-14, 6.07e-14],
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


def apply_dispersion(unit_cfg):
    unit_cfg['col_dispersion'] = V4_DISPERSION['col_dispersion']
    unit_cfg['particle_type_000']['film_diffusion'] = V4_DISPERSION['film_diffusion']
    unit_cfg['particle_type_000']['pore_diffusion'] = V4_DISPERSION['pore_diffusion']


def enable_particle_output(config_data):
    ret = config_data['input']['return']['unit_001']
    ret['write_solution_particle'] = 1
    ret['write_solution_solid'] = 1


def create_DG_config(base_model, prefix, polyDeg, nElem):
    config_name = f'{prefix}_P{polyDeg}_DG_P{polyDeg}Z{nElem}'
    config_data = copy.deepcopy(base_model)

    config_data['input']['solver']['time_integrator'] = TIME_INTEGRATOR_STRICT

    unit_cfg = config_data['input']['model']['unit_001']
    unit_cfg['unit_type'] = 'RADIAL_GENERAL_RATE_MODEL'

    apply_dispersion(unit_cfg)
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

    save_config(config_data, os.path.join(OUTPUT_DIR, config_name))
    return config_name


def create_FV_config(base_model, prefix, nCol):
    config_name = f'{prefix}_FV_Z{nCol}'
    config_data = copy.deepcopy(base_model)

    config_data['input']['solver']['time_integrator'] = TIME_INTEGRATOR_STRICT

    unit_cfg = config_data['input']['model']['unit_001']
    unit_cfg['unit_type'] = 'RADIAL_GENERAL_RATE_MODEL'

    apply_dispersion(unit_cfg)
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

    save_config(config_data, os.path.join(OUTPUT_DIR, config_name))
    return config_name


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_sma = setting_GRM_SMA.get_model()

    created = []

    poly_degs = [1, 2, 3, 4]
    nelem_list = [4, 8, 16, 32, 64, 128]

    print("=== GRM SMA 4comp -- VERY LOW DISPERSION (v4) ===")
    print(f"  COL_DISPERSION = {V4_DISPERSION['col_dispersion'][0]:.2e}")
    print(f"  FILM_DIFFUSION = {V4_DISPERSION['film_diffusion'][0]:.2e}")
    print(f"  PORE_DIFFUSION = {V4_DISPERSION['pore_diffusion']}")
    print()

    print("DG configs (PAR_POLYDEG = POLYDEG, particle output ON):")
    for p in poly_degs:
        for ne in nelem_list:
            name = create_DG_config(base_sma, 'radGRM_DG_SMA_4comp', p, ne)
            created.append(name)
            par_ne = max(1, ne // 4)
            print(f"  {name}  (PAR_POLYDEG={p}, PAR_NELEM={par_ne})")

    print("\nFV configs (particle output ON, Z1024 = reference):")
    for z in [16, 32, 64, 128, 256, 512, 1024]:
        name = create_FV_config(base_sma, 'radGRM_FV_WENO3_SMA_4comp', z)
        created.append(name)
        print(f"  {name}  (NCELLS={max(1, z // 4)})")

    print(f"\nTotal: {len(created)} h5 files created in {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
