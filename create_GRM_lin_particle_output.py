#!/usr/bin/env python3
"""Re-create GRM lin h5 configs with particle/solid output enabled.

Overwrites existing files with WRITE_SOLUTION_PARTICLE=1, WRITE_SOLUTION_SOLID=1.
"""

import sys
import os
import copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cadet import Cadet

import src.benchmark_models.setting_radCol1D_DG_GRM_lin_1comp as setting_GRM_lin

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'output', 'test_cadet-core', 'radialDG')

CADET_PATH = '/Users/yuvj/CADET/build'

TIME_INTEGRATOR_STRICT = {
    'ABSTOL': 1e-12, 'RELTOL': 1e-10, 'ALGTOL': 1e-10,
    'USE_MODIFIED_NEWTON': False,
    'INIT_STEP_SIZE': 1e-6,
    'MAX_STEPS': 5000000
}


def grid_radial_equivolume(r0, r1, n):
    r2_faces = np.linspace(r0**2, r1**2, n + 1)
    return np.sqrt(r2_faces)


def create_DG_config(base_model, prefix, polyDeg, nElem):
    config_name = f'{prefix}_P{polyDeg}_DG_P{polyDeg}Z{nElem}'
    config_data = copy.deepcopy(base_model)
    config_data['input']['solver']['time_integrator'] = TIME_INTEGRATOR_STRICT

    unit_cfg = config_data['input']['model']['unit_001']
    unit_cfg['unit_type'] = 'RADIAL_GENERAL_RATE_MODEL'

    disc = unit_cfg['discretization']
    disc['SPATIAL_METHOD'] = 'DG'
    disc['POLYDEG'] = polyDeg
    disc['NELEM'] = nElem
    unit_cfg['POLYNOMIAL_INTERPOLATION_NODES'] = 'CGL'

    par_disc = unit_cfg['particle_type_000']['discretization']
    par_nelem = max(1, nElem // 4)
    par_disc['NCELLS'] = par_nelem
    par_disc['PAR_POLYDEG'] = 3
    par_disc['PAR_NELEM'] = par_nelem

    # Enable particle and solid output
    config_data['input']['return']['unit_001']['write_solution_particle'] = 1
    config_data['input']['return']['unit_001']['write_solution_solid'] = 1

    model = Cadet(install_path=CADET_PATH)
    model.root.input = config_data['input']
    model.filename = os.path.join(OUTPUT_DIR, config_name + '.h5')
    model.save()
    return config_name


def create_FV_config(base_model, prefix, nCol):
    config_name = f'{prefix}_FV_Z{nCol}'
    config_data = copy.deepcopy(base_model)
    config_data['input']['solver']['time_integrator'] = TIME_INTEGRATOR_STRICT

    unit_cfg = config_data['input']['model']['unit_001']
    unit_cfg['unit_type'] = 'RADIAL_GENERAL_RATE_MODEL'

    disc = unit_cfg['discretization']
    disc['SPATIAL_METHOD'] = 'FV'
    disc['NCOL'] = nCol
    disc['MAX_KRYLOV'] = 0
    disc['MAX_RESTARTS'] = 10
    disc['GS_TYPE'] = 1
    disc['SCHUR_SAFETY'] = 1e-8
    disc['RECONSTRUCTION'] = 'WENO'
    disc['weno'] = {
        'WENO_ORDER': 3, 'WENO_EPS': 1e-10, 'BOUNDARY_MODEL': 0
    }

    r0 = unit_cfg['col_radius_inner']
    r1 = unit_cfg['col_radius_outer']
    disc['GRID_FACES'] = grid_radial_equivolume(r0, r1, nCol).tolist()

    par_disc = unit_cfg['particle_type_000']['discretization']
    par_disc['NCELLS'] = max(1, nCol // 4)
    par_disc['PAR_POLYDEG'] = 3
    par_disc['PAR_NELEM'] = 1

    # Enable particle and solid output
    config_data['input']['return']['unit_001']['write_solution_particle'] = 1
    config_data['input']['return']['unit_001']['write_solution_solid'] = 1

    model = Cadet(install_path=CADET_PATH)
    model.root.input = config_data['input']
    model.filename = os.path.join(OUTPUT_DIR, config_name + '.h5')
    model.save()
    return config_name


def main():
    base_lin = setting_GRM_lin.get_model()
    created = []

    # DG P1-P4, nElem = 4,8,16,32,64,128
    print("Creating DG configs with particle output...")
    for p in [1, 2, 3, 4]:
        for ne in [4, 8, 16, 32, 64, 128]:
            name = create_DG_config(base_lin, 'radGRM_DG_lin_1comp', p, ne)
            created.append(name)

    # FV reference Z8192 + all levels
    print("Creating FV configs with particle output...")
    for z in [128, 256, 512, 1024, 2048, 4096, 8192]:
        name = create_FV_config(base_lin, 'radGRM_FV_WENO3_lin_1comp', z)
        created.append(name)

    print(f"Total: {len(created)} h5 files created")


if __name__ == '__main__':
    main()
