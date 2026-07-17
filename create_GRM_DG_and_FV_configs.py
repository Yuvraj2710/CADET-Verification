#!/usr/bin/env python3
"""Create h5 input files for GRM DG convergence + FV reruns.

Creates:
  - DG P1-P4 configs for GRM lin (nElem=4,8,16,32,64,128) with particle co-refinement
  - DG P1-P4 configs for GRM SMA (nElem=4,8,16,32,64,128) with particle co-refinement
  - FV WENO3 Z16384 for GRM lin (rerun)
  - FV WENO3 Z4096 for GRM SMA (rerun)

All files saved to output/test_cadet-core/radialDG/
"""

import sys
import os
import copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cadet import Cadet

import src.benchmark_models.setting_radCol1D_DG_GRM_lin_1comp as setting_GRM_lin
import src.benchmark_models.setting_radCol1D_DG_GRM_SMA_4comp as setting_GRM_SMA

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'output', 'test_cadet-core', 'radialDG')

# Use local CADET path (files will be transferred to HPC for running)
CADET_PATH = '/Users/yuvj/CADET/build'

TIME_INTEGRATOR_STRICT = {
    'ABSTOL': 1e-12, 'RELTOL': 1e-10, 'ALGTOL': 1e-10,
    'USE_MODIFIED_NEWTON': False,
    'INIT_STEP_SIZE': 1e-6,
    'MAX_STEPS': 5000000
}

# Radial unit type mapping
RADIAL_UNIT_TYPE_MAP = {
    'GENERAL_RATE_PARTICLE': 'RADIAL_GENERAL_RATE_MODEL',
}


def grid_radial_equivolume(r0, r1, n):
    r2_faces = np.linspace(r0**2, r1**2, n + 1)
    return np.sqrt(r2_faces)


def create_DG_config(base_model, prefix, polyDeg, nElem):
    """Create a DG h5 input file with particle co-refinement."""
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

    # Co-refine particle: PAR_NELEM = max(1, nElem // 4)
    par_disc = unit_cfg['particle_type_000']['discretization']
    par_nelem = max(1, nElem // 4)
    par_disc['NCELLS'] = par_nelem
    par_disc['PAR_POLYDEG'] = 3
    par_disc['PAR_NELEM'] = par_nelem

    model = Cadet(install_path=CADET_PATH)
    model.root.input = config_data['input']
    model.filename = os.path.join(OUTPUT_DIR, config_name)
    model.save()
    return config_name


def create_FV_config(base_model, prefix, nCol):
    """Create an FV WENO3 h5 input file with particle co-refinement and equivolume grid."""
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
        'WENO_ORDER': 3,
        'WENO_EPS': 1e-10,
        'BOUNDARY_MODEL': 0
    }

    # Equivolume grid
    r0 = unit_cfg['col_radius_inner']
    r1 = unit_cfg['col_radius_outer']
    disc['GRID_FACES'] = grid_radial_equivolume(r0, r1, nCol).tolist()

    # Particle co-refinement: NCELLS = max(1, nCol // 4)
    par_disc = unit_cfg['particle_type_000']['discretization']
    par_disc['NCELLS'] = max(1, nCol // 4)
    par_disc['PAR_POLYDEG'] = 3
    par_disc['PAR_NELEM'] = 1

    model = Cadet(install_path=CADET_PATH)
    model.root.input = config_data['input']
    model.filename = os.path.join(OUTPUT_DIR, config_name)
    model.save()
    return config_name


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_lin = setting_GRM_lin.get_model()
    base_sma = setting_GRM_SMA.get_model()

    created = []

    # DG configs: P1-P4, nElem = 4, 8, 16, 32, 64, 128
    poly_degs = [1, 2, 3, 4]
    nelem_list = [4, 8, 16, 32, 64, 128]

    print("Creating DG configs for GRM lin...")
    for p in poly_degs:
        for ne in nelem_list:
            name = create_DG_config(base_lin, 'radGRM_DG_lin_1comp', p, ne)
            created.append(name)
            print(f"  {name}")

    print("\nCreating DG configs for GRM SMA...")
    for p in poly_degs:
        for ne in nelem_list:
            name = create_DG_config(base_sma, 'radGRM_DG_SMA_4comp', p, ne)
            created.append(name)
            print(f"  {name}")

    # FV rerun configs
    print("\nCreating FV rerun configs...")
    name = create_FV_config(base_lin, 'radGRM_FV_WENO3_lin_1comp', 16384)
    created.append(name)
    print(f"  {name}")

    name = create_FV_config(base_sma, 'radGRM_FV_WENO3_SMA_4comp', 4096)
    created.append(name)
    print(f"  {name}")

    print(f"\nTotal: {len(created)} h5 files created in {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
