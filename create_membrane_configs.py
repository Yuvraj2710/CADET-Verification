#!/usr/bin/env python3
"""Create h5 input files for membrane chromatography DG vs FV convergence.

Reproduces tracer experiments from Ladd Effio et al. (2016),
J. Chromatogr. A 1429, 142–154 (radial Sartobind Q membrane).

Config A: Dextran T2000 pulse — pure radial transport (Fig. 3a)
Config B: NaCl pulse — radial LRMP, no binding (Fig. 3b)

Creates:
  Config A (transport):
    - DG P1-P5, nElem = 2,4,8,...,512  (9 levels each)
    - FV WENO3, nCells = 4,8,...,131072 (16 levels)
  Config B (LRMP):
    - DG P1-P4, nElem = 2,4,8,...,256  (8 levels each)
    - FV WENO3, nCells = 4,8,...,65536  (15 levels)

All files saved to output/test_cadet-core/radialDG/membrane/
"""

import sys
import os
import copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cadet import Cadet

import src.benchmark_models.setting_radCol1D_DG_membrane_transport_1comp as setting_transport
import src.benchmark_models.setting_radCol1D_DG_membrane_LRMP_1comp as setting_LRMP

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'output', 'test_cadet-core', 'radialDG', 'membrane')

TIME_INTEGRATOR_STRICT = {
    'ABSTOL': 1e-12, 'RELTOL': 1e-10, 'ALGTOL': 1e-10,
    'USE_MODIFIED_NEWTON': False,
    'INIT_STEP_SIZE': 1e-6,
    'MAX_STEPS': 5000000
}

TIME_INTEGRATOR_LRMP = {
    'ABSTOL': 1e-10, 'RELTOL': 1e-8, 'ALGTOL': 1e-10,
    'USE_MODIFIED_NEWTON': False,
    'INIT_STEP_SIZE': 1e-8,
    'MAX_STEPS': 10000000
}

RADIAL_UNIT_TYPE_MAP = {
    'HOMOGENEOUS_PARTICLE': 'RADIAL_LUMPED_RATE_MODEL_WITH_PORES',
}


def grid_radial_equivolume(r0, r1, n):
    r2_faces = np.linspace(r0**2, r1**2, n + 1)
    return np.sqrt(r2_faces)


def _get_particle_type(unit_cfg):
    if unit_cfg.get('npartype', 0) == 0:
        return None
    pt = unit_cfg.get('particle_type_000', {})
    film = pt.get('has_film_diffusion', 0)
    pore = pt.get('has_pore_diffusion', 0)
    if film and pore:
        return 'GENERAL_RATE_PARTICLE'
    elif film:
        return 'HOMOGENEOUS_PARTICLE'
    else:
        return 'EQUILIBRIUM_PARTICLE'


def create_DG_config(base_model, prefix, polyDeg, nElem, time_integrator,
                     node_type='CGL'):
    """Create a DG h5 input file."""
    config_name = f'{prefix}_P{polyDeg}_DG_P{polyDeg}Z{nElem}'
    config_data = copy.deepcopy(base_model)

    config_data['input']['solver']['time_integrator'] = time_integrator

    unit_cfg = config_data['input']['model']['unit_001']

    # Switch to dedicated radial unit type if model has particles
    ptype = _get_particle_type(unit_cfg)
    if ptype is not None and ptype in RADIAL_UNIT_TYPE_MAP:
        unit_cfg['unit_type'] = RADIAL_UNIT_TYPE_MAP[ptype]

    disc = unit_cfg['discretization']
    disc['SPATIAL_METHOD'] = 'DG'
    disc['POLYDEG'] = polyDeg
    disc['NELEM'] = nElem

    unit_cfg['POLYNOMIAL_INTERPOLATION_NODES'] = node_type

    model = Cadet()
    model.root.input = config_data['input']
    model.filename = os.path.join(OUTPUT_DIR, config_name)
    model.save()
    return config_name


def create_FV_config(base_model, prefix, nCol, time_integrator):
    """Create an FV WENO3 h5 input file with equivolume grid."""
    config_name = f'{prefix}_FV_Z{nCol}'
    config_data = copy.deepcopy(base_model)

    config_data['input']['solver']['time_integrator'] = time_integrator

    unit_cfg = config_data['input']['model']['unit_001']

    # Switch to dedicated radial unit type if model has particles
    ptype = _get_particle_type(unit_cfg)
    if ptype is not None and ptype in RADIAL_UNIT_TYPE_MAP:
        unit_cfg['unit_type'] = RADIAL_UNIT_TYPE_MAP[ptype]

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

    # Equivolume radial grid
    r0 = unit_cfg['col_radius_inner']
    r1 = unit_cfg['col_radius_outer']
    disc['GRID_FACES'] = grid_radial_equivolume(r0, r1, nCol).tolist()

    model = Cadet()
    model.root.input = config_data['input']
    model.filename = os.path.join(OUTPUT_DIR, config_name)
    model.save()
    return config_name


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_transport = setting_transport.get_model()
    base_lrmp = setting_LRMP.get_model()

    created = []

    # =================================================================
    # Config A: Dextran T2000 — pure radial transport
    # =================================================================
    print("=" * 60)
    print("Config A: Dextran T2000 pulse (radial transport)")
    print("=" * 60)

    # DG: P1-P5, nElem = 2,4,8,16,32,64,128,256,512
    poly_degs_A = [1, 2, 3, 4, 5]
    nelem_list_A = [2, 4, 8, 16, 32, 64, 128, 256, 512]

    print("\nDG configs:")
    for p in poly_degs_A:
        for ne in nelem_list_A:
            name = create_DG_config(base_transport,
                                    'radMembrane_transport_1comp',
                                    p, ne, TIME_INTEGRATOR_STRICT)
            created.append(name)
            print(f"  {name}")

    # FV WENO3: nCells = 4,8,...,131072 (16 levels)
    fv_cells_A = [4 * 2**i for i in range(16)]  # 4 to 131072

    print("\nFV WENO3 configs:")
    for nc in fv_cells_A:
        name = create_FV_config(base_transport,
                                'radMembrane_transport_1comp',
                                nc, TIME_INTEGRATOR_STRICT)
        created.append(name)
        print(f"  {name}")

    # =================================================================
    # Config B: NaCl pulse — radial LRMP (no binding)
    # =================================================================
    print("\n" + "=" * 60)
    print("Config B: NaCl pulse (radial LRMP, no binding)")
    print("=" * 60)

    # DG: P1-P4, nElem = 2,4,8,16,32,64,128,256
    poly_degs_B = [1, 2, 3, 4]
    nelem_list_B = [2, 4, 8, 16, 32, 64, 128, 256]

    print("\nDG configs:")
    for p in poly_degs_B:
        for ne in nelem_list_B:
            name = create_DG_config(base_lrmp,
                                    'radMembrane_LRMP_1comp',
                                    p, ne, TIME_INTEGRATOR_LRMP)
            created.append(name)
            print(f"  {name}")

    # FV WENO3: nCells = 4,8,...,65536 (15 levels)
    fv_cells_B = [4 * 2**i for i in range(15)]  # 4 to 65536

    print("\nFV WENO3 configs:")
    for nc in fv_cells_B:
        name = create_FV_config(base_lrmp,
                                'radMembrane_LRMP_1comp',
                                nc, TIME_INTEGRATOR_LRMP)
        created.append(name)
        print(f"  {name}")

    # =================================================================
    # Summary
    # =================================================================
    n_dg_A = len(poly_degs_A) * len(nelem_list_A)
    n_fv_A = len(fv_cells_A)
    n_dg_B = len(poly_degs_B) * len(nelem_list_B)
    n_fv_B = len(fv_cells_B)

    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Config A (transport): {n_dg_A} DG + {n_fv_A} FV = {n_dg_A + n_fv_A}")
    print(f"  Config B (LRMP):      {n_dg_B} DG + {n_fv_B} FV = {n_dg_B + n_fv_B}")
    print(f"  Total: {len(created)} h5 files in {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
