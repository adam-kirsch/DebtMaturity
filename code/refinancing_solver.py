"""
Solve for refinancing set K̄*(x) 
"""

import numpy as np
from scipy.interpolate import interp1d
import time


class RefinancingSetSolver:
    """Solve for K̄*(x) using fixed point iteration."""
    
    def __init__(self, bond_valuation, x_grid, K_grid, T_grid):
        self.bond_val = bond_valuation
        self.x_grid = x_grid
        self.K_grid = K_grid
        self.T_grid = T_grid
    
    def T_B_operator(self, K_bar_array):
        """Apply T_B operator: max over bonds."""
        K_bar_func = interp1d(self.x_grid, K_bar_array, 
                             kind='linear', bounds_error=False, 
                             fill_value=(K_bar_array[0], K_bar_array[-1]))
        
        K_bar_new = np.zeros(len(self.x_grid))
        
        for i, x in enumerate(self.x_grid):
            max_value = 0.0
            
            for K in self.K_grid:
                for T in self.T_grid:
                    B_I = self.bond_val.B_illiquid(x, T, K, K_bar_func)
                    if B_I > max_value:
                        max_value = B_I
            
            K_bar_new[i] = max_value
        
        return K_bar_new
    
    def solve(self, tol=1e-3, max_iter=100, verbose=True):
        """Iterate until convergence."""
        # Initial guess
        K_bar = np.array([0.5 * self.bond_val.F_unlevered(x) for x in self.x_grid])
        
        if verbose:
            print("="*60)
            print("Solving for K̄*(x)")
            print("="*60)
            print(f"Grid: x={len(self.x_grid)}, K={len(self.K_grid)}, T={len(self.T_grid)}")
        
        start_time = time.time()
        
        for iteration in range(max_iter):
            K_bar_new = self.T_B_operator(K_bar)
            error = np.max(np.abs(K_bar_new - K_bar))
            
            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration:3d}: error = {error:.6f}")
            
            if error < tol:
                elapsed = time.time() - start_time
                if verbose:
                    print(f"\n✓ Converged in {iteration} iterations ({elapsed:.1f}s)")
                    print(f"  K̄*(x_min={self.x_grid[0]:.1f}) = {K_bar_new[0]:.2f}")
                    print(f"  K̄*(x_max={self.x_grid[-1]:.1f}) = {K_bar_new[-1]:.2f}")
                
                K_bar_star_func = interp1d(self.x_grid, K_bar_new,
                                          kind='linear', bounds_error=False,
                                          fill_value=(K_bar_new[0], K_bar_new[-1]))
                
                return K_bar_new, K_bar_star_func
            
            K_bar = K_bar_new
        
        print(f"\n⚠ Warning: Did not converge (error={error:.6f})")
        K_bar_star_func = interp1d(self.x_grid, K_bar,
                                   kind='linear', bounds_error=False,
                                   fill_value=(K_bar[0], K_bar[-1]))
        return K_bar, K_bar_star_func
