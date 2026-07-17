#!/usr/bin/env python3
"""Create h5 input files for rGRM v3 study — less dispersive setting.

Reduced dispersion to show proper DG P+1 convergence rates:
  COL_DISPERSION:  5.75e-8  -> 5.75e-10  (100x less axial dispersion)
  PAR_DIFFUSION:   6.07e-10 -> 6.07e-11  (10x less pore diffusion)
  FILM_DIFFUSION:  6.9e-6   -> 6.9e-7    (10x less film transfer)

Particle and solid output enabled for all configs.
PAR_POLYDEG = POLYDEG (Breuer thesis format).

Creates:
  GRM lin 1comp:
    - DG P1-P4, nElem = 4,8,16,32,64,128
    - FV WENO3 Z16,32,64,128,256,512,1024,2048,4096
"""

import sys
import os
import copy
import numpy as np
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import src.benchmark_models.setting_radCol1D_DG_GRM_lin_1comp as setting_GRM_lin

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', '..', 'output', 'test_cadet-core', 'radialDG', 'v3')

TIME_INTEGRATOR_STRICT = {
    'ABSTOL': 1e-12, 'RELTOL': 1e-10, 'ALGTOL': 1e-10,
    'USE_MODIFIED_NEWTON': False,
    'INIT_STEP_SIZE': 1e-6,
    'MAX_STEPS': 5000000
}

# Less dispersive parameters
LESS_DISPERSIVE = {
    'col_dispersion': [5.75e-10],   # 100x less (was 5.75e-8)
    'film_diffusion': [6.9e-7],     # 10x less (was 6.9e-6)
    'pore_diffusion': [6.07e-11],   # 10x less (was 6.07e-10)
}


def grid_radial_equivolume(r0, r1, n):
    r2_faces = np.linspace(r0**2, r1**2, n + 1)
    return np.sqrt(r2_faces)


def write_dict_to_h5(group, d):
    """Recursively write a nested dict/addict to an h5 group.

    Keys are uppercased except group-level paths (unit_000, sec_000, switch_000,
    particle_type_000, etc.) which stay lowercase per CADET convention.
    """
    for key, val in d.items():
        # Only group-path keys stay lowercase (structural h5 groups)
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
            # Use fixed-length byte string (CADET C++ expects this, not variable-length)
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
    """Save config dict to h5 file."""
    if not filepath.endswith('.h5'):
        filepath += '.h5'
    with h5py.File(filepath, 'w') as f:
        write_dict_to_h5(f, config_data)
    return filepath


def apply_less_dispersive(unit_cfg):
    """Override dispersion parameters to make problem less smooth."""
    unit_cfg['col_dispersion'] = LESS_DISPERSIVE['col_dispersion']
    unit_cfg['particle_type_000']['film_diffusion'] = LESS_DISPERSIVE['film_diffusion']
    unit_cfg['particle_type_000']['pore_diffusion'] = LESS_DISPERSIVE['pore_diffusion']


def enable_particle_output(config_data):
    """Enable particle and solid output."""
    ret = config_data['input']['return']['unit_001']
    ret['write_solution_particle'] = 1
    ret['write_solution_solid'] = 1


def create_DG_config(base_model, prefix, polyDeg, nElem):
    """Create a DG h5 config with PAR_POLYDEG = POLYDEG (Breuer format)."""
    config_name = f'{prefix}_P{polyDeg}_DG_P{polyDeg}Z{nElem}'
    config_data = copy.deepcopy(base_model)

    config_data['input']['solver']['time_integrator'] = TIME_INTEGRATOR_STRICT

    unit_cfg = config_data['input']['model']['unit_001']
    unit_cfg['unit_type'] = 'RADIAL_GENERAL_RATE_MODEL'

    apply_less_dispersive(unit_cfg)
    enable_particle_output(config_data)

    disc = unit_cfg['discretization']
    disc['SPATIAL_METHOD'] = 'DG'
    disc['POLYDEG'] = polyDeg
    disc['NELEM'] = nElem

    unit_cfg['POLYNOMIAL_INTERPOLATION_NODES'] = 'CGL'

    # Particle co-refinement: PAR_NELEM = max(1, nElem // 4), PAR_POLYDEG = POLYDEG
    par_disc = unit_cfg['particle_type_000']['discretization']
    par_nelem = max(1, nElem // 4)
    par_disc['NCELLS'] = par_nelem
    par_disc['PAR_POLYDEG'] = polyDeg
    par_disc['PAR_NELEM'] = par_nelem

    filepath = save_config(config_data, os.path.join(OUTPUT_DIR, config_name))
    return config_name


def create_FV_config(base_model, prefix, nCol):
    """Create an FV WENO3 h5 config with equivolume grid."""
    config_name = f'{prefix}_FV_Z{nCol}'
    config_data = copy.deepcopy(base_model)

    config_data['input']['solver']['time_integrator'] = TIME_INTEGRATOR_STRICT

    unit_cfg = config_data['input']['model']['unit_001']
    unit_cfg['unit_type'] = 'RADIAL_GENERAL_RATE_MODEL'

    apply_less_dispersive(unit_cfg)
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

    # Equivolume grid
    r0 = unit_cfg['col_radius_inner']
    r1 = unit_cfg['col_radius_outer']
    disc['GRID_FACES'] = grid_radial_equivolume(r0, r1, nCol).tolist()

    # Particle: FV cells, co-refinement NCELLS = max(1, nCol // 4)
    par_disc = unit_cfg['particle_type_000']['discretization']
    par_disc['NCELLS'] = max(1, nCol // 4)
    par_disc['PAR_POLYDEG'] = 3
    par_disc['PAR_NELEM'] = 1

    filepath = save_config(config_data, os.path.join(OUTPUT_DIR, config_name))
    return config_name


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_lin = setting_GRM_lin.get_model()

    created = []

    poly_degs = [1, 2, 3, 4]
    nelem_list = [4, 8, 16, 32, 64, 128]

    print("=== GRM lin 1comp — LESS DISPERSIVE (v3) ===")
    print(f"  COL_DISPERSION = {LESS_DISPERSIVE['col_dispersion'][0]:.2e}")
    print(f"  FILM_DIFFUSION = {LESS_DISPERSIVE['film_diffusion'][0]:.2e}")
    print(f"  PAR_DIFFUSION  = {LESS_DISPERSIVE['pore_diffusion'][0]:.2e}")
    print()

    print("DG configs (PAR_POLYDEG = POLYDEG):")
    for p in poly_degs:
        for ne in nelem_list:
            name = create_DG_config(base_lin, 'radGRM_DG_lin_1comp', p, ne)
            created.append(name)
            par_ne = max(1, ne // 4)
            print(f"  {name}  (PAR_POLYDEG={p}, PAR_NELEM={par_ne})")

    print("\nFV configs:")
    for z in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        name = create_FV_config(base_lin, 'radGRM_FV_WENO3_lin_1comp', z)
        created.append(name)
        print(f"  {name}  (NCELLS={max(1, z // 4)})")

    print(f"\nTotal: {len(created)} h5 files created in {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
