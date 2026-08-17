"""
Quick test of optimization methods comparison.
"""

import numpy as np
from config import Params
from cir_process import CIRProcess
from bond_valuation import BondValuation
from firm_valuation import FirmValuation
from refinancing_solver import RefinancingSetSolver
from optimization_methods import compare_methods

# Setup
p = Params()
cir = CIRProcess(p.kappa, p.mu, p.sigma)
bond_val = BondValuation(cir, p.r, p.eta, p.C)
firm_val = FirmValuation(bond_val)

# Solve for K̄*(x) 
print("Solving for K̄*(x)...")
x_grid = np.linspace(p.x_min, 25, 20)
K_grid_solve = np.linspace(p.K_min, p.K_max, 20)
T_grid_solve = np.linspace(p.T_min, p.T_max, 15)

ref_solver = RefinancingSetSolver(bond_val, x_grid, K_grid_solve, T_grid_solve)
_, K_bar_func = ref_solver.solve(tol=1e-3, max_iter=30, verbose=False)
print("✓ K̄*(x) computed\n")

# V̄ approximation
def V_bar_approx(x, K):
    F_x = bond_val.F_unlevered(x)
    if K_bar_func(x) >= K + p.C:
        return F_x - K
    else:
        return max(0, F_x - K)

# Test parameters
x_test = 10.0
K_0_test = p.K_0

# Grids for grid search
K_grid = np.linspace(K_0_test, 400, 30)
T_grid = np.linspace(0.1, 30, 25)

# Run comparison
results = compare_methods(
    x=x_test,
    K_0=K_0_test,
    bond_val=bond_val,
    firm_val=firm_val,
    K_bar_func=K_bar_func,
    V_bar_func=V_bar_approx,
    K_grid=K_grid,
    T_grid=T_grid
)
