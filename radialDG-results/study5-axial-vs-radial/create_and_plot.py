#!/usr/bin/env python3
"""Study 5: Axial vs Radial outlet comparison.

Uses the same GRM parameters from study 2 v2 (lin 1comp + SMA 4comp).
Creates axial GRM configs, runs them, then plots outlet profiles
per component overlaying radial vs axial.

Radial solutions: reuse fine DG P4 Z128 from study 2 v2.
Axial solutions: create and run with same DG P4 Z128.

Axial mapping:
  - col_length = r_out - r_in = 0.014 m
  - velocity = velocity_coeff / r_in (inlet velocity)
  - col_dispersion = same constant value
  - All particle/binding params unchanged
"""

import os
import sys
import copy
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cadet import Cadet

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, '..', '..')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'test_cadet-core', 'radialDG', 'v2')
CADET_PATH = os.path.expanduser('~/local-fix')

# Radial parameters
R_IN = 0.01
R_OUT = 0.024
COL_LENGTH = R_OUT - R_IN  # 0.014 m
VELOCITY_COEFF = 9.775e-6
VELOCITY_AXIAL = 5.75e-4
COL_DISP = 5.75e-8
COL_POROSITY = 0.37

# DG discretization
POLYDEG = 4
NELEM = 128
PAR_POLYDEG = POLYDEG
PAR_NELEM = max(1, NELEM // 4)

TIME_INTEGRATOR = {
    'ABSTOL': 1e-12, 'RELTOL': 1e-10, 'ALGTOL': 1e-10,
    'INIT_STEP_SIZE': 1e-8, 'MAX_STEPS': 1000000
}


def create_axial_lin_1comp():
    """Create axial GRM lin 1comp config."""
    from addict import Dict
    m = Dict()

    m.input.model.nunits = 3
    m.input.model.connections.nswitches = 1
    m.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, 6e-5,
        1.0, 2.0, -1.0, -1.0, 6e-5,
    ]
    m.input.model.connections.switch_000.section = 0

    m.input.model.solver.gs_type = 1
    m.input.model.solver.max_krylov = 0
    m.input.model.solver.max_restarts = 10
    m.input.model.solver.schur_safety = 1e-8

    # Inlet
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = 1
    m.input.model.unit_000.sec_000.const_coeff = [1.0]
    m.input.model.unit_000.sec_000.lin_coeff = [0.0]
    m.input.model.unit_000.sec_000.quad_coeff = [0.0]
    m.input.model.unit_000.sec_000.cube_coeff = [0.0]
    m.input.model.unit_000.sec_001.const_coeff = [0.0]
    m.input.model.unit_000.sec_001.lin_coeff = [0.0]
    m.input.model.unit_000.sec_001.quad_coeff = [0.0]
    m.input.model.unit_000.sec_001.cube_coeff = [0.0]

    # Axial GRM
    col = m.input.model.unit_001
    col.unit_type = 'GENERAL_RATE_MODEL'
    col.ncomp = 1
    col.npartype = 1
    col.col_length = COL_LENGTH
    col.col_porosity = COL_POROSITY
    col.par_type_volfrac = [1.0]
    col.col_dispersion = [COL_DISP]
    col.velocity = VELOCITY_AXIAL
    col.init_c = [0.0]

    # Particle (same as radial)
    pt = col.particle_type_000
    pt.has_film_diffusion = 1
    pt.has_pore_diffusion = 1
    pt.has_surface_diffusion = 0
    pt.par_porosity = 0.75
    pt.par_radius = 4.5e-5
    pt.par_coreradius = 0.0
    pt.par_geom = 'SPHERE'
    pt.film_diffusion = [6.9e-6]
    pt.pore_diffusion = [6.07e-10]
    pt.surface_diffusion = [0.0]
    pt.nbound = [1]
    pt.init_cp = [0.0]
    pt.init_cs = [0.0]

    # Binding: Linear kinetic
    pt.adsorption_model = 'LINEAR'
    pt.adsorption.is_kinetic = 1
    pt.adsorption.lin_ka = [3.55]
    pt.adsorption.lin_kd = [0.1]

    # Particle discretization
    pt.discretization.NCELLS = 256
    pt.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'

    col.discretization.USE_ANALYTIC_JACOBIAN = 1
    col.discretization.SPATIAL_METHOD = 'FV'
    col.discretization.NCOL = 1024
    col.discretization.RECONSTRUCTION = 'WENO'
    col.discretization.GS_TYPE = 1
    col.discretization.MAX_KRYLOV = 0
    col.discretization.MAX_RESTARTS = 10
    col.discretization.SCHUR_SAFETY = 1e-8
    col.discretization.weno.WENO_ORDER = 3
    col.discretization.weno.WENO_EPS = 1e-10
    col.discretization.weno.BOUNDARY_MODEL = 0

    # Outlet
    m.input.model.unit_002.unit_type = 'OUTLET'
    m.input.model.unit_002.ncomp = 1

    # Return
    m.input['return'].split_components_data = 0
    m.input['return'].split_ports_data = 0
    m.input['return'].unit_001.write_solution_outlet = 1
    m.input['return'].unit_001.write_solution_bulk = 0
    m.input['return'].unit_001.write_solution_inlet = 0

    # Solver
    m.input.solver.consistent_init_mode = 1
    m.input.solver.nthreads = 1
    m.input.solver.sections.nsec = 2
    m.input.solver.sections.section_continuity = [0]
    m.input.solver.sections.section_times = [0.0, 10.0, 1500.0]
    m.input.solver.time_integrator = TIME_INTEGRATOR
    m.input.solver.user_solution_times = np.linspace(0.0, 1500.0, 1501)

    return m


def create_axial_sma_4comp():
    """Create axial GRM SMA 4comp config."""
    from addict import Dict
    ncomp = 4
    m = Dict()

    m.input.model.nunits = 3
    m.input.model.connections.nswitches = 1
    m.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, 6e-5,
        1.0, 2.0, -1.0, -1.0, 6e-5,
    ]
    m.input.model.connections.switch_000.section = 0

    m.input.model.solver.gs_type = 1
    m.input.model.solver.max_krylov = 0
    m.input.model.solver.max_restarts = 10
    m.input.model.solver.schur_safety = 1e-8

    # Inlet — 3 sections
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = ncomp
    m.input.model.unit_000.sec_000.const_coeff = [50.0, 1.0, 1.0, 1.0]
    m.input.model.unit_000.sec_000.lin_coeff = [0.0] * ncomp
    m.input.model.unit_000.sec_000.quad_coeff = [0.0] * ncomp
    m.input.model.unit_000.sec_000.cube_coeff = [0.0] * ncomp
    m.input.model.unit_000.sec_001.const_coeff = [50.0, 0.0, 0.0, 0.0]
    m.input.model.unit_000.sec_001.lin_coeff = [0.0] * ncomp
    m.input.model.unit_000.sec_001.quad_coeff = [0.0] * ncomp
    m.input.model.unit_000.sec_001.cube_coeff = [0.0] * ncomp
    m.input.model.unit_000.sec_002.const_coeff = [100.0, 0.0, 0.0, 0.0]
    m.input.model.unit_000.sec_002.lin_coeff = [0.0] * ncomp
    m.input.model.unit_000.sec_002.quad_coeff = [0.0] * ncomp
    m.input.model.unit_000.sec_002.cube_coeff = [0.0] * ncomp

    # Axial GRM
    col = m.input.model.unit_001
    col.unit_type = 'GENERAL_RATE_MODEL'
    col.ncomp = ncomp
    col.npartype = 1
    col.col_length = COL_LENGTH
    col.col_porosity = COL_POROSITY
    col.par_type_volfrac = [1.0]
    col.col_dispersion = [COL_DISP] * ncomp
    col.velocity = VELOCITY_AXIAL
    col.init_c = [50.0, 0.0, 0.0, 0.0]

    # Particle (same as radial)
    pt = col.particle_type_000
    pt.has_film_diffusion = 1
    pt.has_pore_diffusion = 1
    pt.has_surface_diffusion = 0
    pt.par_porosity = 0.75
    pt.par_radius = 4.5e-5
    pt.par_coreradius = 0.0
    pt.par_geom = 'SPHERE'
    pt.film_diffusion = [6.9e-6] * ncomp
    pt.pore_diffusion = [7e-10, 6.07e-11, 6.07e-11, 6.07e-11]
    pt.surface_diffusion = [0.0] * ncomp
    pt.nbound = [1] * ncomp
    pt.init_cp = [50.0, 0.0, 0.0, 0.0]
    pt.init_cs = [1200.0, 0.0, 0.0, 0.0]

    # Binding: SMA kinetic
    pt.adsorption_model = 'STERIC_MASS_ACTION'
    pt.adsorption.is_kinetic = 1
    pt.adsorption.sma_lambda = 1200.0
    pt.adsorption.sma_ka = [0.0, 35.5, 1.59, 7.7]
    pt.adsorption.sma_kd = [0.0, 1000.0, 1000.0, 1000.0]
    pt.adsorption.sma_nu = [0.0, 4.7, 5.29, 3.7]
    pt.adsorption.sma_sigma = [0.0, 11.83, 10.6, 10.0]

    # Particle discretization (FV)
    pt.discretization.NCELLS = 256
    pt.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'

    col.discretization.USE_ANALYTIC_JACOBIAN = 1
    col.discretization.SPATIAL_METHOD = 'FV'
    col.discretization.NCOL = 1024
    col.discretization.RECONSTRUCTION = 'WENO'
    col.discretization.GS_TYPE = 1
    col.discretization.MAX_KRYLOV = 0
    col.discretization.MAX_RESTARTS = 10
    col.discretization.SCHUR_SAFETY = 1e-8
    col.discretization.weno.WENO_ORDER = 3
    col.discretization.weno.WENO_EPS = 1e-10
    col.discretization.weno.BOUNDARY_MODEL = 0

    # Outlet
    m.input.model.unit_002.unit_type = 'OUTLET'
    m.input.model.unit_002.ncomp = ncomp

    # Return
    m.input['return'].split_components_data = 0
    m.input['return'].split_ports_data = 0
    m.input['return'].unit_001.write_solution_outlet = 1
    m.input['return'].unit_001.write_solution_bulk = 0
    m.input['return'].unit_001.write_solution_inlet = 0

    # Solver
    m.input.solver.consistent_init_mode = 1
    m.input.solver.nthreads = 1
    m.input.solver.sections.nsec = 3
    m.input.solver.sections.section_continuity = [0, 0]
    m.input.solver.sections.section_times = [0.0, 10.0, 90.0, 1500.0]
    m.input.solver.time_integrator = TIME_INTEGRATOR
    m.input.solver.user_solution_times = np.linspace(0.0, 1500.0, 1501)

    return m


def create_radial_lin_1comp():
    """Create radial GRM lin 1comp config with VELOCITY_COEFF."""
    from addict import Dict
    m = Dict()

    m.input.model.nunits = 3
    m.input.model.connections.nswitches = 1
    m.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, 6e-5,
        1.0, 2.0, -1.0, -1.0, 6e-5,
    ]
    m.input.model.connections.switch_000.section = 0

    m.input.model.solver.gs_type = 1
    m.input.model.solver.max_krylov = 0
    m.input.model.solver.max_restarts = 10
    m.input.model.solver.schur_safety = 1e-8

    # Inlet
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = 1
    m.input.model.unit_000.sec_000.const_coeff = [1.0]
    m.input.model.unit_000.sec_000.lin_coeff = [0.0]
    m.input.model.unit_000.sec_000.quad_coeff = [0.0]
    m.input.model.unit_000.sec_000.cube_coeff = [0.0]
    m.input.model.unit_000.sec_001.const_coeff = [0.0]
    m.input.model.unit_000.sec_001.lin_coeff = [0.0]
    m.input.model.unit_000.sec_001.quad_coeff = [0.0]
    m.input.model.unit_000.sec_001.cube_coeff = [0.0]

    # Radial GRM
    col = m.input.model.unit_001
    col.unit_type = 'RADIAL_GENERAL_RATE_MODEL'
    col.ncomp = 1
    col.npartype = 1
    col.col_radius_inner = R_IN
    col.col_radius_outer = R_OUT
    col.col_porosity = COL_POROSITY
    col.par_type_volfrac = [1.0]
    col.col_dispersion = [COL_DISP]
    col.velocity_coeff = VELOCITY_COEFF
    col.init_c = [0.0]

    # Particle
    pt = col.particle_type_000
    pt.has_film_diffusion = 1
    pt.has_pore_diffusion = 1
    pt.has_surface_diffusion = 0
    pt.par_porosity = 0.75
    pt.par_radius = 4.5e-5
    pt.par_coreradius = 0.0
    pt.par_geom = 'SPHERE'
    pt.film_diffusion = [6.9e-6]
    pt.pore_diffusion = [6.07e-10]
    pt.surface_diffusion = [0.0]
    pt.nbound = [1]
    pt.init_cp = [0.0]
    pt.init_cs = [0.0]

    # Binding: Linear kinetic
    pt.adsorption_model = 'LINEAR'
    pt.adsorption.is_kinetic = 1
    pt.adsorption.lin_ka = [3.55]
    pt.adsorption.lin_kd = [0.1]

    # Particle discretization
    pt.discretization.NCELLS = PAR_NELEM
    pt.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
    pt.discretization.PAR_POLYDEG = POLYDEG
    pt.discretization.PAR_NELEM = PAR_NELEM

    col.discretization.USE_ANALYTIC_JACOBIAN = 1
    col.discretization.SPATIAL_METHOD = 'DG'
    col.discretization.NELEM = NELEM
    col.discretization.POLYDEG = POLYDEG

    # Outlet
    m.input.model.unit_002.unit_type = 'OUTLET'
    m.input.model.unit_002.ncomp = 1

    # Return
    m.input['return'].split_components_data = 0
    m.input['return'].split_ports_data = 0
    m.input['return'].unit_001.write_solution_outlet = 1
    m.input['return'].unit_001.write_solution_bulk = 0
    m.input['return'].unit_001.write_solution_inlet = 0

    # Solver
    m.input.solver.consistent_init_mode = 1
    m.input.solver.nthreads = 1
    m.input.solver.sections.nsec = 2
    m.input.solver.sections.section_continuity = [0]
    m.input.solver.sections.section_times = [0.0, 10.0, 1500.0]
    m.input.solver.time_integrator = TIME_INTEGRATOR
    m.input.solver.user_solution_times = np.linspace(0.0, 1500.0, 1501)

    return m


def create_radial_sma_4comp():
    """Create radial GRM SMA 4comp config with VELOCITY_COEFF."""
    from addict import Dict
    ncomp = 4
    m = Dict()

    m.input.model.nunits = 3
    m.input.model.connections.nswitches = 1
    m.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, 6e-5,
        1.0, 2.0, -1.0, -1.0, 6e-5,
    ]
    m.input.model.connections.switch_000.section = 0

    m.input.model.solver.gs_type = 1
    m.input.model.solver.max_krylov = 0
    m.input.model.solver.max_restarts = 10
    m.input.model.solver.schur_safety = 1e-8

    # Inlet — 3 sections
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = ncomp
    m.input.model.unit_000.sec_000.const_coeff = [50.0, 1.0, 1.0, 1.0]
    m.input.model.unit_000.sec_000.lin_coeff = [0.0] * ncomp
    m.input.model.unit_000.sec_000.quad_coeff = [0.0] * ncomp
    m.input.model.unit_000.sec_000.cube_coeff = [0.0] * ncomp
    m.input.model.unit_000.sec_001.const_coeff = [50.0, 0.0, 0.0, 0.0]
    m.input.model.unit_000.sec_001.lin_coeff = [0.0] * ncomp
    m.input.model.unit_000.sec_001.quad_coeff = [0.0] * ncomp
    m.input.model.unit_000.sec_001.cube_coeff = [0.0] * ncomp
    m.input.model.unit_000.sec_002.const_coeff = [100.0, 0.0, 0.0, 0.0]
    m.input.model.unit_000.sec_002.lin_coeff = [0.0] * ncomp
    m.input.model.unit_000.sec_002.quad_coeff = [0.0] * ncomp
    m.input.model.unit_000.sec_002.cube_coeff = [0.0] * ncomp

    # Radial GRM
    col = m.input.model.unit_001
    col.unit_type = 'RADIAL_GENERAL_RATE_MODEL'
    col.ncomp = ncomp
    col.npartype = 1
    col.col_radius_inner = R_IN
    col.col_radius_outer = R_OUT
    col.col_porosity = COL_POROSITY
    col.par_type_volfrac = [1.0]
    col.col_dispersion = [COL_DISP] * ncomp
    col.velocity_coeff = VELOCITY_COEFF
    col.init_c = [50.0, 0.0, 0.0, 0.0]

    # Particle
    pt = col.particle_type_000
    pt.has_film_diffusion = 1
    pt.has_pore_diffusion = 1
    pt.has_surface_diffusion = 0
    pt.par_porosity = 0.75
    pt.par_radius = 4.5e-5
    pt.par_coreradius = 0.0
    pt.par_geom = 'SPHERE'
    pt.film_diffusion = [6.9e-6] * ncomp
    pt.pore_diffusion = [7e-10, 6.07e-11, 6.07e-11, 6.07e-11]
    pt.surface_diffusion = [0.0] * ncomp
    pt.nbound = [1] * ncomp
    pt.init_cp = [50.0, 0.0, 0.0, 0.0]
    pt.init_cs = [1200.0, 0.0, 0.0, 0.0]

    # Binding: SMA kinetic
    pt.adsorption_model = 'STERIC_MASS_ACTION'
    pt.adsorption.is_kinetic = 1
    pt.adsorption.sma_lambda = 1200.0
    pt.adsorption.sma_ka = [0.0, 35.5, 1.59, 7.7]
    pt.adsorption.sma_kd = [0.0, 1000.0, 1000.0, 1000.0]
    pt.adsorption.sma_nu = [0.0, 4.7, 5.29, 3.7]
    pt.adsorption.sma_sigma = [0.0, 11.83, 10.6, 10.0]

    # Particle discretization
    pt.discretization.NCELLS = PAR_NELEM
    pt.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
    pt.discretization.PAR_POLYDEG = POLYDEG
    pt.discretization.PAR_NELEM = PAR_NELEM

    col.discretization.USE_ANALYTIC_JACOBIAN = 1
    col.discretization.SPATIAL_METHOD = 'DG'
    col.discretization.NELEM = NELEM
    col.discretization.POLYDEG = POLYDEG

    # Outlet
    m.input.model.unit_002.unit_type = 'OUTLET'
    m.input.model.unit_002.ncomp = ncomp

    # Return
    m.input['return'].split_components_data = 0
    m.input['return'].split_ports_data = 0
    m.input['return'].unit_001.write_solution_outlet = 1
    m.input['return'].unit_001.write_solution_bulk = 0
    m.input['return'].unit_001.write_solution_inlet = 0

    # Solver
    m.input.solver.consistent_init_mode = 1
    m.input.solver.nthreads = 1
    m.input.solver.sections.nsec = 3
    m.input.solver.sections.section_continuity = [0, 0]
    m.input.solver.sections.section_times = [0.0, 10.0, 90.0, 1500.0]
    m.input.solver.time_integrator = TIME_INTEGRATOR
    m.input.solver.user_solution_times = np.linspace(0.0, 1500.0, 1501)

    return m


def run_or_load(config, name):
    """Save config, run CADET if no output, return outlet solution."""
    filepath = os.path.join(OUTPUT_DIR, name + '.h5')
    model = Cadet(install_path=CADET_PATH)

    if os.path.exists(filepath):
        # Check if already has output
        with h5py.File(filepath, 'r') as f:
            if 'output' in f:
                sol = f['output']['solution']['unit_001']['SOLUTION_OUTLET'][:]
                times = f['output']['solution']['SOLUTION_TIMES'][:]
                print(f"  Loaded existing: {name}")
                return times, sol

    # Save and run
    model.root.input = config['input']
    model.filename = filepath
    model.save()
    print(f"  Running: {name} ...")
    ret = model.run_load()
    if ret.return_code != 0:
        print(f"  ERROR: {name} failed with code {ret.return_code}")
        return None, None
    sol = model.root.output.solution.unit_001.solution_outlet
    times = np.array(model.root.output.solution.solution_times)
    print(f"  Done: {name} (sim_time={model.root.meta.time_sim:.1f}s)")
    return times, sol


def plot_comparison(times_rad, sol_rad, times_ax, sol_ax,
                    ncomp, comp_names, title, fname):
    """Plot outlet per component: radial vs axial."""
    fig, axes = plt.subplots(ncomp, 1, figsize=(8, 3.5 * ncomp), sharex=True)
    if ncomp == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(times_rad, sol_rad[:, i], '-', color='#1f77b4',
                linewidth=1.2, label='Radial')
        ax.plot(times_ax, sol_ax[:, i], '--', color='#d62728',
                linewidth=1.2, label='Axial')
        ax.set_ylabel(f'{comp_names[i]}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    axes[-1].set_xlabel('Time [s]', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Linear 1comp ---
    print("=== GRM lin 1comp ===")

    # Radial
    rad_lin_config = create_radial_lin_1comp()
    times_lin, sol_rad_lin = run_or_load(rad_lin_config, 'radGRM_study5_lin_1comp_DG_P4Z128')

    # Axial
    ax_lin_config = create_axial_lin_1comp()
    times_ax_lin, sol_ax_lin = run_or_load(ax_lin_config, 'axGRM_study5_lin_1comp_FV_Z1024')

    if sol_rad_lin is not None and sol_ax_lin is not None:
        plot_comparison(times_lin, sol_rad_lin, times_ax_lin, sol_ax_lin,
                        1, ['Concentration'], 'GRM lin 1comp',
                        'lin_1comp_radial_vs_axial.png')

    # --- SMA 4comp ---
    print("\n=== GRM SMA 4comp ===")

    # Radial
    rad_sma_config = create_radial_sma_4comp()
    times_sma, sol_rad_sma = run_or_load(rad_sma_config, 'radGRM_study5_SMA_4comp_DG_P4Z128')

    # Axial
    ax_sma_config = create_axial_sma_4comp()
    times_ax_sma, sol_ax_sma = run_or_load(ax_sma_config, 'axGRM_study5_SMA_4comp_FV_Z1024')

    if sol_rad_sma is not None and sol_ax_sma is not None:
        plot_comparison(times_sma, sol_rad_sma, times_ax_sma, sol_ax_sma,
                        4, ['Salt', 'Protein 1', 'Protein 2', 'Protein 3'],
                        'GRM SMA 4comp',
                        'sma_4comp_radial_vs_axial.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
