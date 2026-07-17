#!/usr/bin/env python3
"""Create FV Z2048 and Z4096 reference configs for v5 study — outlet only (no particle output).

Same intermediate dispersion as v5:
  COL_DISPERSION = 5.75e-11
  FILM_DIFFUSION = 6.9e-8
  PAR_DIFFUSION  = 6.07e-12
"""

import sys
import os
import copy
import numpy as np
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import src.benchmark_models.setting_radCol1D_DG_GRM_lin_1comp as setting_GRM_lin

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', '..', 'output', 'test_cadet-core', 'radialDG', 'v5')

TIME_INTEGRATOR_STRICT = {
    'ABSTOL': 1e-12, 'RELTOL': 1e-10, 'ALGTOL': 1e-10,
    'USE_MODIFIED_NEWTON': False,
    'INIT_STEP_SIZE': 1e-6,
    'MAX_STEPS': 5000000
}

MID_DISPERSION = {
    'col_dispersion': [5.75e-11],
    'film_diffusion': [6.9e-8],
    'pore_diffusion': [6.07e-12],
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_lin = setting_GRM_lin.get_model()

    for nCol in [2048, 4096]:
        config_name = f'radGRM_FV_WENO3_lin_1comp_FV_Z{nCol}'
        config_data = copy.deepcopy(base_lin)

        config_data['input']['solver']['time_integrator'] = TIME_INTEGRATOR_STRICT

        unit_cfg = config_data['input']['model']['unit_001']
        unit_cfg['unit_type'] = 'RADIAL_GENERAL_RATE_MODEL'

        # Apply intermediate dispersion
        unit_cfg['col_dispersion'] = MID_DISPERSION['col_dispersion']
        unit_cfg['particle_type_000']['film_diffusion'] = MID_DISPERSION['film_diffusion']
        unit_cfg['particle_type_000']['pore_diffusion'] = MID_DISPERSION['pore_diffusion']

        # Disable particle output (outlet only)
        ret = config_data['input']['return']['unit_001']
        ret['write_solution_particle'] = 0
        ret['write_solution_solid'] = 0

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

        path = save_config(config_data, os.path.join(OUTPUT_DIR, config_name))
        print(f"Created: {config_name}  (NCELLS={max(1, nCol // 4)}, particle output OFF)")

    print(f"\nOutput dir: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
