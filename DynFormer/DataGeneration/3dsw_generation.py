"""
Dedalus script simulating the viscous shallow water equations on a sphere. This
script demonstrates solving an initial value problem on the sphere. It can be
ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_sphere.py` script can be used to produce
plots from the saved data. The simulation should about 5 cpu-minutes to run.

The script implements the test case of a barotropically unstable mid-latitude
jet from Galewsky et al. 2004 (https://doi.org/10.3402/tellusa.v56i5.14436).
The initial height field balanced the imposed jet is solved with an LBVP.
A perturbation is then added and the solution is evolved as an IVP.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 shallow_water.py
    $ mpiexec -n 4 python3 plot_sphere.py snapshots/*.h5
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import os

# Simulation units
meter = 1 / 6.37122e6
hour = 1
second = hour / 3600

# Parameters
Nphi = 256
Ntheta = 128
dealias = 3/2
R = 6.37122e6 * meter
Omega = 7.292e-5 / second
nu = 1e5 * meter**2 / second / 32**2 # Hyperdiffusion matched at ell=32
g = 9.80616 * meter / second**2
H = 1e4 * meter
timestep = 600 * second
stop_sim_time = 360 * hour
dtype = np.float64

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

# Generate multiple cases with different hpert values
# Create base directory for all cases
base_dir = 'multiple_cases'
os.makedirs(base_dir, exist_ok=True)

# Generate parameter grid for alpha and beta
alpha_values = np.linspace(1/120, 10, 40)  # 40 values for alpha
beta_values = np.linspace(1/300, 2, 30)  # 30 values for beta
# Total cases will be 40 * 30 = 1200

case_idx = 0
for alpha_value in alpha_values:
    for beta_value in beta_values:
        # Fields
        u = dist.VectorField(coords, name='u', bases=basis)
        h = dist.Field(name='h', bases=basis)

        # Substitutions
        zcross = lambda A: d3.MulCosine(d3.skew(A))

        # Initial conditions: zonal jet
        phi, theta = dist.local_grids(basis)
        lat = np.pi / 2 - theta + 0*phi
        umax = 80 * meter / second
        lat0 = np.pi / 7
        lat1 = np.pi / 2 - lat0
        en = np.exp(-4 / (lat1 - lat0)**2)
        jet = (lat0 <= lat) * (lat <= lat1)
        u_jet = umax / en * np.exp(1 / (lat[jet] - lat0) / (lat[jet] - lat1))
        u['g'][0][jet]  = u_jet

        # Initial conditions: balanced height
        c = dist.Field(name='c')
        problem = d3.LBVP([h, c], namespace=locals())
        problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*zcross(u))")
        problem.add_equation("ave(h) = 0")
        solver = problem.build_solver()
        solver.solve()

        # Add perturbation with current parameters
        lat2 = np.pi / 4
        hpert = 120 * meter
        alpha = alpha_value
        beta = beta_value
        h['g'] += hpert * np.cos(lat) * np.exp(-(phi/alpha)**2) * np.exp(-((lat2-lat)/beta)**2)

        # Create case-specific directory
        case_dir = os.path.join(base_dir, f'case_{case_idx:04d}')
        os.makedirs(case_dir, exist_ok=True)

        # Problem
        problem = d3.IVP([u, h], namespace=locals())
        problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = - u@grad(u)")
        problem.add_equation("dt(h) + nu*lap(lap(h)) + H*div(u) = - div(h*u)")

        # Solver
        solver = problem.build_solver(d3.RK222)
        solver.stop_sim_time = stop_sim_time

        # Analysis - save snapshots in case-specific directory
        snapshots = solver.evaluator.add_file_handler(os.path.join(case_dir, 'snapshots'), sim_dt=12*hour, max_writes=30)
        snapshots.add_task(h, name='height')
        snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

        # Save parameters for reference
        with open(os.path.join(case_dir, 'parameters.txt'), 'w') as f:
            f.write(f'alpha = {alpha_value}\n')
            f.write(f'beta = {beta_value}\n')
            f.write(f'hpert = {120} meters\n')

        # Main loop
        try:
            logger.info(f'Starting main loop for case {case_idx} with alpha = {alpha_value}, beta = {beta_value}')
            while solver.proceed:
                solver.step(timestep)
                if (solver.iteration-1) % 2000 == 0:
                    logger.info('Case %d: Iteration=%i, Time=%e, dt=%e' %(case_idx, solver.iteration, solver.sim_time, timestep))
        except:
            logger.error(f'Exception raised in case {case_idx}, triggering end of main loop.')
            raise
        finally:
            solver.log_stats()
            logger.info(f'Completed case {case_idx}')
            case_idx += 1

logger.info('All cases completed successfully')
