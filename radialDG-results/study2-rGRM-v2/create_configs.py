#!/usr/bin/env python3
"""Create h5 input files for rGRM v2 study (Breuer thesis format).

Key difference from original: PAR_POLYDEG = POLYDEG (same order for bulk and particle),
matching Jan Breuer's thesis setup.

Creates:
  GRM lin 1comp:
    - DG P1-P4, nElem = 4,8,16,32,64,128 (PAR_POLYDEG=POLYDEG, PAR_NELEM=max(1,nElem//4))
    - FV WENO3 Z128,256,512,1024,2048,4096,8192 (NCELLS=max(1,NCOL//4))

  GRM SMA 4comp:
    - DG P1-P4, nElem = 4,8,16,32,64,128 (PAR_POLYDEG=POLYDEG, PAR_NELEM=max(1,nElem//4))
    - FV WENO3 Z128,256,512,1024,2048 (NCELLS=max(1,NCOL//4))

All files saved to output/test_cadet-core/radialDG/v2/
"""

import sys
import os
import copy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from cadet import Cadet

import src.benchmark_models.setting_radCol1D_DG_GRM_lin_1comp as setting_GRM_lin
import src.benchmark_models.setting_radCol1D_DG_GRM_SMA_4comp as setting_GRM_SMA

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', '..', 'output', 'test_cadet-core', 'radialDG', 'v2')

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
    """Create a DG h5 config with PAR_POLYDEG = POLYDEG (Breuer format)."""
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

    # Particle co-refinement: PAR_NELEM = max(1, nElem // 4), PAR_POLYDEG = POLYDEG
    par_disc = unit_cfg['particle_type_000']['discretization']
    par_nelem = max(1, nElem // 4)
    par_disc['NCELLS'] = par_nelem
    par_disc['PAR_POLYDEG'] = polyDeg  # Same as bulk POLYDEG (Breuer thesis)
    par_disc['PAR_NELEM'] = par_nelem

    model = Cadet(install_path=CADET_PATH)
    model.root.input = config_data['input']
    model.filename = os.path.join(OUTPUT_DIR, config_name)
    model.save()
    return config_name


def create_FV_config(base_model, prefix, nCol):
    """Create an FV WENO3 h5 config with equivolume grid."""
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

    # Particle: FV cells, co-refinement NCELLS = max(1, nCol // 4)
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

    poly_degs = [1, 2, 3, 4]
    nelem_list = [4, 8, 16, 32, 64, 128]

    # --- GRM lin ---
    print("=== GRM lin 1comp ===")

    print("DG configs (PAR_POLYDEG = POLYDEG):")
    for p in poly_degs:
        for ne in nelem_list:
            name = create_DG_config(base_lin, 'radGRM_DG_lin_1comp', p, ne)
            created.append(name)
            par_ne = max(1, ne // 4)
            print(f"  {name}  (PAR_POLYDEG={p}, PAR_NELEM={par_ne})")

    print("FV configs:")
    for z in [128, 256, 512, 1024, 2048, 4096, 8192]:
        name = create_FV_config(base_lin, 'radGRM_FV_WENO3_lin_1comp', z)
        created.append(name)
        print(f"  {name}  (NCELLS={max(1, z // 4)})")

    # --- GRM SMA ---
    print("\n=== GRM SMA 4comp ===")

    print("DG configs (PAR_POLYDEG = POLYDEG):")
    for p in poly_degs:
        for ne in nelem_list:
            name = create_DG_config(base_sma, 'radGRM_DG_SMA_4comp', p, ne)
            created.append(name)
            par_ne = max(1, ne // 4)
            print(f"  {name}  (PAR_POLYDEG={p}, PAR_NELEM={par_ne})")

    print("FV configs:")
    for z in [128, 256, 512, 1024, 2048]:
        name = create_FV_config(base_sma, 'radGRM_FV_WENO3_SMA_4comp', z)
        created.append(name)
        print(f"  {name}  (NCELLS={max(1, z // 4)})")

    print(f"\nTotal: {len(created)} h5 files created in {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
