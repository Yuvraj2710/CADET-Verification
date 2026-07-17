#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Membrane chromatography study: DG vs FV WENO3 convergence.

Reproduces tracer experiments from Ladd Effio et al. (2016),
J. Chromatogr. A 1429, 142–154 (radial Sartobind Q membrane).

Config A: Dextran T2000 pulse — pure radial transport (Fig. 3a)
Config B: NaCl pulse — radial LRMP, no binding (Fig. 3b)

Usage:
  python run_membrane_study.py --cadet /path/to/cadet-cli \\
      --output radialDG-results-membrane --njobs 4

  # Run only one config:
  python run_membrane_study.py ... --configs 0      # dextran only
  python run_membrane_study.py ... --configs 1      # NaCl only
  python run_membrane_study.py ... --configs 0 1    # both

  # Quick sanity test:
  python run_membrane_study.py ... --small
"""

import os
import sys
import copy
import argparse
import traceback
import numpy as np
from functools import partial

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.benchmark_models.setting_radCol1D_DG_membrane_transport_1comp as setting_membrane_transport
import src.benchmark_models.setting_radCol1D_DG_membrane_LRMP_1comp as setting_membrane_LRMP

import src.bench_func as bench_func
import src.bench_configs as bench_configs
import src.utility.convergence as convergence

from cadet import Cadet


def main():
    parser = argparse.ArgumentParser(description='Membrane chromatography DG vs FV study')
    parser.add_argument('--cadet', required=True, help='Path to cadet-cli binary')
    parser.add_argument('--output', default='radialDG-results-membrane',
                        help='Output directory')
    parser.add_argument('--njobs', type=int, default=1, help='Parallel jobs')
    parser.add_argument('--configs', type=int, nargs='+', default=[0, 1],
                        help='Which configs to run: 0=dextran, 1=NaCl')
    parser.add_argument('--small', action='store_true',
                        help='Quick sanity test with fewer discretizations')
    parser.add_argument('--polydegs', type=int, nargs='+', default=None,
                        help='Override DG polynomial degrees (e.g. --polydegs 1 2 3)')
    parser.add_argument('--skip-fv', action='store_true',
                        help='Skip FV WENO3 runs (reuse existing)')
    args = parser.parse_args()

    cadet_path = args.cadet
    output_path = args.output
    n_jobs = args.njobs
    small_test = args.small

    os.makedirs(output_path, exist_ok=True)

    # ---- Time integrator settings ----

    time_integrator_strict = {
        'ABSTOL': 1e-12, 'RELTOL': 1e-10, 'ALGTOL': 1e-10,
        'USE_MODIFIED_NEWTON': False,
        'INIT_STEP_SIZE': 1e-6,
        'MAX_STEPS': 5000000
    }

    time_integrator = {
        'ABSTOL': 1e-10, 'RELTOL': 1e-8, 'ALGTOL': 1e-10,
        'USE_MODIFIED_NEWTON': False,
        'INIT_STEP_SIZE': 1e-8,
        'MAX_STEPS': 10000000
    }

    # ---- Equivolume radial grid ----

    def grid_radial_equivolume(r0, r1, n):
        r2_faces = np.linspace(r0**2, r1**2, n + 1)
        return np.sqrt(r2_faces)

    # ---- Radial unit type mapping ----

    _RADIAL_UNIT_TYPE_MAP = {
        'EQUILIBRIUM_PARTICLE': 'RADIAL_LUMPED_RATE_MODEL_WITHOUT_PORES',
        'HOMOGENEOUS_PARTICLE': 'RADIAL_LUMPED_RATE_MODEL_WITH_PORES',
        'GENERAL_RATE_PARTICLE': 'RADIAL_GENERAL_RATE_MODEL',
    }

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

    # ---- Refinement functions ----

    def refine_DG(config_data, disc_idx, setting_name,
                  polyDeg, node_type='CGL',
                  nelem_start=2, time_integrator=None,
                  unit_id='001', only_return_name=False, **kwargs):

        nElem = nelem_start * 2**disc_idx
        config_name = convergence.generate_1D_name(setting_name, polyDeg, nElem)

        if only_return_name:
            if output_path is not None:
                return str(output_path) + '/' + config_name
            return config_name

        config_data = copy.deepcopy(config_data)

        if time_integrator is not None:
            config_data['input']['solver']['time_integrator'] = time_integrator

        unit_cfg = config_data['input']['model']['unit_' + unit_id]

        ptype = _get_particle_type(unit_cfg)
        if ptype is not None and ptype in _RADIAL_UNIT_TYPE_MAP:
            unit_cfg['unit_type'] = _RADIAL_UNIT_TYPE_MAP[ptype]

        disc = unit_cfg['discretization']
        disc['SPATIAL_METHOD'] = 'DG'
        disc['POLYDEG'] = polyDeg
        disc['NELEM'] = nElem
        unit_cfg['POLYNOMIAL_INTERPOLATION_NODES'] = node_type

        model = Cadet(install_path=cadet_path)
        model.root.input = config_data['input']
        if output_path is not None:
            model.filename = str(output_path) + '/' + config_name
            model.save()
        return model

    def refine_FV_WENO3(config_data, disc_idx, setting_name,
                        nCol_start=8, equivolume=True,
                        time_integrator=None,
                        unit_id='001', only_return_name=False, **kwargs):

        nCol = nCol_start * 2**disc_idx
        config_name = convergence.generate_1D_name(setting_name, 0, nCol)

        if only_return_name:
            if output_path is not None:
                return str(output_path) + '/' + config_name
            return config_name

        config_data = copy.deepcopy(config_data)

        if time_integrator is not None:
            config_data['input']['solver']['time_integrator'] = time_integrator

        unit_cfg = config_data['input']['model']['unit_' + unit_id]

        ptype = _get_particle_type(unit_cfg)
        if ptype is not None and ptype in _RADIAL_UNIT_TYPE_MAP:
            unit_cfg['unit_type'] = _RADIAL_UNIT_TYPE_MAP[ptype]

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

        # Ensure particle discretization fields for LRMP
        pt0 = unit_cfg.get('particle_type_000', {})
        pt_disc = pt0.get('discretization', {})
        if pt_disc:
            pt_disc.setdefault('NCELLS', 10)
            pt_disc.setdefault('PAR_DISC_TYPE', 'EQUIDISTANT_PAR')

        if equivolume:
            r0 = unit_cfg['col_radius_inner']
            r1 = unit_cfg['col_radius_outer']
            disc['GRID_FACES'] = grid_radial_equivolume(r0, r1, nCol).tolist()

        model = Cadet(install_path=cadet_path)
        model.root.input = config_data['input']
        if output_path is not None:
            model.filename = str(output_path) + '/' + config_name
            model.save()
        return model

    # =========================================================================
    #  Config A: Dextran T2000 pulse — pure radial transport
    #  (DG self-convergence + DG vs FV WENO3)
    # =========================================================================

    if 0 in args.configs:
        print("\n" + "="*72)
        print("  Config A: Dextran T2000 pulse (pure radial transport)")
        print("="*72)

        base_model_A = setting_membrane_transport.get_model()

        # --- DG sweep ---
        if small_test:
            poly_degs_A = [1, 2, 3]
            n_disc_DG_A = 4       # nElem: 2,4,8,16
        else:
            poly_degs_A = args.polydegs if args.polydegs else list(range(1, 6))
            n_disc_DG_A = 9       # nElem: 2,4,8,...,512

        # --- FV sweep ---
        if small_test:
            fv_start_A = 4
            n_disc_FV_A = 4       # nCells: 4,8,16,32
        else:
            fv_start_A = 4
            n_disc_FV_A = 16      # nCells: 4,8,...,131072

        # Phase 1: FV WENO3 sweep
        fv_configs_A = []
        fv_config_names_A = []
        fv_include_sens_A = []
        fv_ref_files_A = []
        fv_unit_IDs_A = []
        fv_which_A = []
        fv_idas_abstol_A = []
        fv_ax_methods_A = []
        fv_ax_discs_A = []
        fv_par_methods_A = []
        fv_par_discs_A = []
        fv_disc_refinement_functions_A = []

        fv_name_A = 'radMembrane_FV_WENO3_transport_1comp'
        fv_finest_ncol_A = fv_start_A * 2**(n_disc_FV_A - 1)
        fv_ref_file_A = convergence.generate_1D_name(fv_name_A, 0, fv_finest_ncol_A)

        addition_fv_A = {
            'cadet_config_jsons': [base_model_A],
            'cadet_config_names': [fv_name_A],
            'include_sens': [False],
            'ref_files': [[None]],
            'unit_IDs': ['001'],
            'which': ['outlet'],
            'idas_abstol': [[None]],
            'ax_methods': [[0]],
            'ax_discs': [[bench_func.disc_list(fv_start_A, n_disc_FV_A)]],
            'par_methods': [[None]],
            'par_discs': [[None]],
            'disc_refinement_functions': [
                [partial(refine_FV_WENO3,
                         setting_name=fv_name_A,
                         nCol_start=fv_start_A,
                         equivolume=True,
                         time_integrator=time_integrator_strict)]
            ],
        }

        bench_configs.add_benchmark(
            fv_configs_A, fv_include_sens_A, fv_ref_files_A, fv_unit_IDs_A,
            fv_which_A, fv_ax_methods_A, fv_ax_discs_A,
            par_methods=fv_par_methods_A, par_discs=fv_par_discs_A,
            idas_abstol=fv_idas_abstol_A,
            cadet_config_names=fv_config_names_A, addition=addition_fv_A,
            disc_refinement_functions=fv_disc_refinement_functions_A)

        if not args.skip_fv:
            try:
                print("\n--- Config A: FV WENO3 sweep ---")
                bench_func.run_convergence_analysis(
                    output_path=output_path,
                    cadet_path=cadet_path,
                    cadet_configs=fv_configs_A,
                    cadet_config_names=fv_config_names_A,
                    include_sens=fv_include_sens_A,
                    ref_files=fv_ref_files_A,
                    unit_IDs=fv_unit_IDs_A,
                    which=fv_which_A,
                    ax_methods=fv_ax_methods_A,
                    ax_discs=fv_ax_discs_A,
                    par_methods=fv_par_methods_A,
                    par_discs=fv_par_discs_A,
                    idas_abstol=fv_idas_abstol_A,
                    n_jobs=n_jobs,
                    rerun_sims=True,
                    disc_refinement_functions=fv_disc_refinement_functions_A
                )
            except Exception:
                print(f"\n*** Config A FV FAILED ***\n{traceback.format_exc()}")

        # Phase 2: DG sweep against FV reference
        print("\n--- Config A: DG convergence against FV reference ---")

        dg_configs_A = []
        dg_config_names_A = []
        dg_include_sens_A = []
        dg_ref_files_A = []
        dg_unit_IDs_A = []
        dg_which_A = []
        dg_idas_abstol_A = []
        dg_ax_methods_A = []
        dg_ax_discs_A = []
        dg_par_methods_A = []
        dg_par_discs_A = []
        dg_disc_refinement_functions_A = []

        methods_A = []
        for p in poly_degs_A:
            methods_A.append((f'radMembrane_DG_transport_1comp_P{p}', p))

        addition_dg_A = {
            'cadet_config_jsons': [base_model_A] * len(methods_A),
            'cadet_config_names': [name for name, _ in methods_A],
            'include_sens': [False] * len(methods_A),
            'ref_files': [[fv_ref_file_A] for _ in methods_A],
            'unit_IDs': ['001'] * len(methods_A),
            'which': ['outlet'] * len(methods_A),
            'idas_abstol': [[None] for _ in methods_A],
            'ax_methods': [[p] for _, p in methods_A],
            'ax_discs': [[bench_func.disc_list(2, n_disc_DG_A)] for _ in methods_A],
            'par_methods': [[None] for _ in methods_A],
            'par_discs': [[None] for _ in methods_A],
            'disc_refinement_functions': [
                [partial(refine_DG,
                         setting_name=name,
                         polyDeg=polyDeg,
                         node_type='CGL',
                         nelem_start=2,
                         time_integrator=time_integrator_strict)]
                for name, polyDeg in methods_A
            ],
        }

        bench_configs.add_benchmark(
            dg_configs_A, dg_include_sens_A, dg_ref_files_A, dg_unit_IDs_A,
            dg_which_A, dg_ax_methods_A, dg_ax_discs_A,
            par_methods=dg_par_methods_A, par_discs=dg_par_discs_A,
            idas_abstol=dg_idas_abstol_A,
            cadet_config_names=dg_config_names_A, addition=addition_dg_A,
            disc_refinement_functions=dg_disc_refinement_functions_A)

        try:
            bench_func.run_convergence_analysis(
                output_path=output_path,
                cadet_path=cadet_path,
                cadet_configs=dg_configs_A,
                cadet_config_names=dg_config_names_A,
                include_sens=dg_include_sens_A,
                ref_files=dg_ref_files_A,
                unit_IDs=dg_unit_IDs_A,
                which=dg_which_A,
                ax_methods=dg_ax_methods_A,
                ax_discs=dg_ax_discs_A,
                par_methods=dg_par_methods_A,
                par_discs=dg_par_discs_A,
                idas_abstol=dg_idas_abstol_A,
                n_jobs=n_jobs,
                rerun_sims=True,
                disc_refinement_functions=dg_disc_refinement_functions_A
            )
        except Exception:
            print(f"\n*** Config A DG FAILED ***\n{traceback.format_exc()}")

    # =========================================================================
    #  Config B: NaCl pulse — radial LRMP (no binding)
    # =========================================================================

    if 1 in args.configs:
        print("\n" + "="*72)
        print("  Config B: NaCl pulse (radial LRMP, no binding)")
        print("="*72)

        base_model_B = setting_membrane_LRMP.get_model()

        # --- DG sweep ---
        if small_test:
            poly_degs_B = [1, 2, 3]
            n_disc_DG_B = 4       # nElem: 2,4,8,16
        else:
            poly_degs_B = args.polydegs if args.polydegs else list(range(1, 5))
            n_disc_DG_B = 8       # nElem: 2,4,8,...,256

        # --- FV sweep ---
        if small_test:
            fv_start_B = 4
            n_disc_FV_B = 4       # nCells: 4,8,16,32
        else:
            fv_start_B = 4
            n_disc_FV_B = 15      # nCells: 4,8,...,65536

        # Phase 1: FV WENO3 sweep
        fv_configs_B = []
        fv_config_names_B = []
        fv_include_sens_B = []
        fv_ref_files_B = []
        fv_unit_IDs_B = []
        fv_which_B = []
        fv_idas_abstol_B = []
        fv_ax_methods_B = []
        fv_ax_discs_B = []
        fv_par_methods_B = []
        fv_par_discs_B = []
        fv_disc_refinement_functions_B = []

        fv_name_B = 'radMembrane_FV_WENO3_LRMP_1comp'
        fv_finest_ncol_B = fv_start_B * 2**(n_disc_FV_B - 1)
        fv_ref_file_B = convergence.generate_1D_name(fv_name_B, 0, fv_finest_ncol_B)

        addition_fv_B = {
            'cadet_config_jsons': [base_model_B],
            'cadet_config_names': [fv_name_B],
            'include_sens': [False],
            'ref_files': [[None]],
            'unit_IDs': ['001'],
            'which': ['outlet'],
            'idas_abstol': [[None]],
            'ax_methods': [[0]],
            'ax_discs': [[bench_func.disc_list(fv_start_B, n_disc_FV_B)]],
            'par_methods': [[None]],
            'par_discs': [[None]],
            'disc_refinement_functions': [
                [partial(refine_FV_WENO3,
                         setting_name=fv_name_B,
                         nCol_start=fv_start_B,
                         equivolume=True,
                         time_integrator=time_integrator)]
            ],
        }

        bench_configs.add_benchmark(
            fv_configs_B, fv_include_sens_B, fv_ref_files_B, fv_unit_IDs_B,
            fv_which_B, fv_ax_methods_B, fv_ax_discs_B,
            par_methods=fv_par_methods_B, par_discs=fv_par_discs_B,
            idas_abstol=fv_idas_abstol_B,
            cadet_config_names=fv_config_names_B, addition=addition_fv_B,
            disc_refinement_functions=fv_disc_refinement_functions_B)

        if not args.skip_fv:
            try:
                print("\n--- Config B: FV WENO3 sweep ---")
                bench_func.run_convergence_analysis(
                    output_path=output_path,
                    cadet_path=cadet_path,
                    cadet_configs=fv_configs_B,
                    cadet_config_names=fv_config_names_B,
                    include_sens=fv_include_sens_B,
                    ref_files=fv_ref_files_B,
                    unit_IDs=fv_unit_IDs_B,
                    which=fv_which_B,
                    ax_methods=fv_ax_methods_B,
                    ax_discs=fv_ax_discs_B,
                    par_methods=fv_par_methods_B,
                    par_discs=fv_par_discs_B,
                    idas_abstol=fv_idas_abstol_B,
                    n_jobs=n_jobs,
                    rerun_sims=True,
                    disc_refinement_functions=fv_disc_refinement_functions_B
                )
            except Exception:
                print(f"\n*** Config B FV FAILED ***\n{traceback.format_exc()}")

        # Phase 2: DG sweep against FV reference
        print("\n--- Config B: DG convergence against FV reference ---")

        dg_configs_B = []
        dg_config_names_B = []
        dg_include_sens_B = []
        dg_ref_files_B = []
        dg_unit_IDs_B = []
        dg_which_B = []
        dg_idas_abstol_B = []
        dg_ax_methods_B = []
        dg_ax_discs_B = []
        dg_par_methods_B = []
        dg_par_discs_B = []
        dg_disc_refinement_functions_B = []

        methods_B = []
        for p in poly_degs_B:
            methods_B.append((f'radMembrane_DG_LRMP_1comp_P{p}', p))

        addition_dg_B = {
            'cadet_config_jsons': [base_model_B] * len(methods_B),
            'cadet_config_names': [name for name, _ in methods_B],
            'include_sens': [False] * len(methods_B),
            'ref_files': [[fv_ref_file_B] for _ in methods_B],
            'unit_IDs': ['001'] * len(methods_B),
            'which': ['outlet'] * len(methods_B),
            'idas_abstol': [[None] for _ in methods_B],
            'ax_methods': [[p] for _, p in methods_B],
            'ax_discs': [[bench_func.disc_list(2, n_disc_DG_B)] for _ in methods_B],
            'par_methods': [[None] for _ in methods_B],
            'par_discs': [[None] for _ in methods_B],
            'disc_refinement_functions': [
                [partial(refine_DG,
                         setting_name=name,
                         polyDeg=polyDeg,
                         node_type='CGL',
                         nelem_start=2,
                         time_integrator=time_integrator)]
                for name, polyDeg in methods_B
            ],
        }

        bench_configs.add_benchmark(
            dg_configs_B, dg_include_sens_B, dg_ref_files_B, dg_unit_IDs_B,
            dg_which_B, dg_ax_methods_B, dg_ax_discs_B,
            par_methods=dg_par_methods_B, par_discs=dg_par_discs_B,
            idas_abstol=dg_idas_abstol_B,
            cadet_config_names=dg_config_names_B, addition=addition_dg_B,
            disc_refinement_functions=dg_disc_refinement_functions_B)

        try:
            bench_func.run_convergence_analysis(
                output_path=output_path,
                cadet_path=cadet_path,
                cadet_configs=dg_configs_B,
                cadet_config_names=dg_config_names_B,
                include_sens=dg_include_sens_B,
                ref_files=dg_ref_files_B,
                unit_IDs=dg_unit_IDs_B,
                which=dg_which_B,
                ax_methods=dg_ax_methods_B,
                ax_discs=dg_ax_discs_B,
                par_methods=dg_par_methods_B,
                par_discs=dg_par_discs_B,
                idas_abstol=dg_idas_abstol_B,
                n_jobs=n_jobs,
                rerun_sims=True,
                disc_refinement_functions=dg_disc_refinement_functions_B
            )
        except Exception:
            print(f"\n*** Config B DG FAILED ***\n{traceback.format_exc()}")

    print("\n" + "="*72)
    print("  Membrane study complete.")
    print("="*72)


if __name__ == '__main__':
    main()
