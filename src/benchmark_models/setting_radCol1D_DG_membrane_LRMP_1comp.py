# -*- coding: utf-8 -*-
"""
Membrane study, Config B: NaCl pulse — radial LRMP (no binding).

Reproduces the NaCl tracer experiment from Ladd Effio et al. (2016),
J. Chromatogr. A 1429, 142–154, Fig. 3b.

Model: RADIAL_COLUMN_MODEL_1D → RADIAL_LUMPED_RATE_MODEL_WITH_PORES
Domain: r in [3.25e-3, 11.25e-3] (Sartobind Q nano, 3 mL)
Inward radial flow, F = 3 mL/min.

NaCl enters both the interstitial volume and hydrogel pores (small molecule).
No binding; mass transfer via lumped film diffusion.

From the paper:
  k_eff,NaCl = 0.1829 s^{-1}  (lumped effective mass transfer rate)
  Mapped to CADET LRMP:  3*kf/R_p = k_eff  →  kf = k_eff * R_p / 3

Pulse injection: 100 uL at 3 mL/min → t_inj = 2 s.
Section 0: [0, 2]  c = 1  (injection)
Section 1: [2, 120] c = 0  (elution)

Parameters (SI):
  r_inner        = 3.25e-3 m
  r_outer        = 11.3986e-3 m  (adjusted so pi*(r_out^2-r_in^2)*H = 3 mL = manufacturer VM)
  eps_pores      = 0.518  (interstitial = col_porosity)
  eps_hydrogel   = 0.581  (particle = par_porosity)
  D_ax           = 3.74e-8 m^2/s
  velocity_coeff = -1.9204e-6 m^2/s  (inward)
  R_p            = 4.5e-5 m  (fictitious particle radius)
  kf             = 2.7435e-6 m/s  (= 0.1829 * 4.5e-5 / 3)
"""

import numpy as np
from addict import Dict


def get_model():

    r_in = 3.25e-3     # inner radius [m]
    r_out = 11.3986e-3   # outer radius [m] (adjusted for VM = 3 mL)

    F = 5.0e-8           # volumetric flow rate [m^3/s]
    velocity_coeff = -1.9204e-6  # inward radial flow [m^2/s]

    # Mass transfer: k_eff = 0.1829 s^{-1}
    # CADET LRMP: 3*kf/R_p = k_eff
    R_p = 4.5e-5         # fictitious particle radius [m]
    k_eff_NaCl = 0.1829  # [1/s]
    kf = k_eff_NaCl * R_p / 3.0  # = 2.7435e-6 [m/s]

    m = Dict()

    m.input.model.nunits = 3
    m.input.model.connections.nswitches = 1
    m.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, F,
        1.0, 2.0, -1.0, -1.0, F,
    ]
    m.input.model.connections.switch_000.section = 0

    m.input.model.solver.gs_type = 1
    m.input.model.solver.max_krylov = 0
    m.input.model.solver.max_restarts = 10
    m.input.model.solver.schur_safety = 1e-8

    # Inlet — pulse injection
    m.input.model.unit_000.unit_type = 'INLET'
    m.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    m.input.model.unit_000.ncomp = 1
    m.input.model.unit_000.sec_000.const_coeff = [1.0]
    m.input.model.unit_000.sec_000.lin_coeff   = [0.0]
    m.input.model.unit_000.sec_000.quad_coeff  = [0.0]
    m.input.model.unit_000.sec_000.cube_coeff  = [0.0]
    m.input.model.unit_000.sec_001.const_coeff = [0.0]
    m.input.model.unit_000.sec_001.lin_coeff   = [0.0]
    m.input.model.unit_000.sec_001.quad_coeff  = [0.0]
    m.input.model.unit_000.sec_001.cube_coeff  = [0.0]

    # Radial column — LRMP (lumped rate model with pores)
    col = m.input.model.unit_001
    col.unit_type = 'RADIAL_COLUMN_MODEL_1D'
    col.ncomp = 1
    col.npartype = 1
    col.col_radius_inner = r_in
    col.col_radius_outer = r_out
    col.col_porosity = 0.518       # interstitial (membrane) porosity
    col.par_type_volfrac = [1.0]
    col.col_dispersion = [3.74e-8]  # D_ax [m^2/s]
    col.velocity_coeff = velocity_coeff
    col.init_c = [0.0]

    # Particle (hydrogel layer): film diffusion only, no pore diffusion, no binding
    pt = col.particle_type_000
    pt.has_film_diffusion = 1
    pt.has_pore_diffusion = 0
    pt.has_surface_diffusion = 0
    pt.par_porosity = 0.581        # hydrogel porosity
    pt.par_radius = R_p
    pt.par_coreradius = 0.0
    pt.par_geom = 'SPHERE'
    pt.film_diffusion = [kf]
    pt.nbound = [0]                # no binding
    pt.init_cp = [0.0]

    col.discretization.USE_ANALYTIC_JACOBIAN = 1

    # Outlet
    m.input.model.unit_002.unit_type = 'OUTLET'
    m.input.model.unit_002.ncomp = 1

    # Return config
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
    m.input.solver.sections.section_times = [0.0, 2.0, 120.0]
    m.input.solver.time_integrator.abstol = 1e-12
    m.input.solver.time_integrator.reltol = 1e-10
    m.input.solver.time_integrator.algtol = 1e-10
    m.input.solver.time_integrator.init_step_size = 1e-8
    m.input.solver.time_integrator.max_steps = 1000000
    m.input.solver.user_solution_times = np.linspace(0.0, 120.0, 2401)

    return m
