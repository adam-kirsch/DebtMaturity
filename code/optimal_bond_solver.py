"""
Find optimal bond (K̂, T̂)
"""

import numpy as np
from scipy.interpolate import interp1d


class OptimalBondSolver:
    """Find optimal bond choice for given earnings level."""
    
    def __init__(self, bond_valuation, K_bar_star_func):
        self.bond_val = bond_valuation
        self.K_bar_star_func = K_bar_star_func
    
    def find_optimal_bond(self, x, K_0, K_grid, T_grid, verbose=False):
        """
        Find (K̂, T̂) that:
        1. Raises at least K_0 
        2. Maximizes some objective (for now, minimize spread)
        
        Parameters
        ----------
        x : float
            Current earnings
        K_0 : float
            Amount needed to raise
        K_grid : array
            Face values to search
        T_grid : array
            Maturities to search
            
        Returns
        -------
        K_opt : float
            Optimal face value
        T_opt : float
            Optimal maturity
        B_opt : float
            Bond value
        """
        best_obj = -np.inf
        best_K, best_T, best_B = None, None, None
        
        feasible_found = False
        
        for K in K_grid:
            for T in T_grid:
                # Check if bond raises K_0
                B_I = self.bond_val.B_illiquid(x, T, K, self.K_bar_star_func)
                
                if B_I >= K_0:
                    feasible_found = True
                    
                    # Objective: minimize face value (all else equal, lower K is better)
                    # Or: maximize bond value relative to face value
                    obj = B_I / K  # Higher is better (less default risk)
                    
                    if obj > best_obj:
                        best_obj = obj
                        best_K = K
                        best_T = T
                        best_B = B_I
        
        if not feasible_found:
            if verbose:
                print(f"  ✗ No feasible bond at x={x:.2f} (cannot raise K_0={K_0})")
            return None, None, None
        
        return best_K, best_T, best_B
    
    def solve_for_grid(self, x_grid, K_0, K_grid, T_grid, verbose=True):
        """
        Find optimal bond for each x in grid.
        
        Returns
        -------
        dict with arrays:
            x, K_hat, T_hat, B_hat, spread
        """
        results = {
            'x': [],
            'K_hat': [],
            'T_hat': [],
            'B_hat': [],
            'spread_bp': [],
            'feasible': []
        }
        
        if verbose:
            print("\n" + "="*60)
            print(f"Finding Optimal Bonds (K_0 = {K_0})")
            print("="*60)
        
        for x in x_grid:
            K_hat, T_hat, B_hat = self.find_optimal_bond(x, K_0, K_grid, T_grid)
            
            if K_hat is not None:
                # Calculate spread
                implied_yield = np.log(K_hat / B_hat) / T_hat
                spread_bp = (implied_yield - self.bond_val.r) * 10000
                
                results['x'].append(x)
                results['K_hat'].append(K_hat)
                results['T_hat'].append(T_hat)
                results['B_hat'].append(B_hat)
                results['spread_bp'].append(spread_bp)
                results['feasible'].append(True)
            else:
                results['feasible'].append(False)
        
        # Convert to arrays
        for key in results:
            results[key] = np.array(results[key])
        
        if verbose:
            n_feasible = np.sum(results['feasible'])
            print(f"\n✓ Found {n_feasible}/{len(x_grid)} feasible solutions")
            if n_feasible > 0:
                idx = results['feasible']
                print(f"  K̂ range: [{np.min(results['K_hat']):.1f}, {np.max(results['K_hat']):.1f}]")
                print(f"  T̂ range: [{np.min(results['T_hat']):.1f}, {np.max(results['T_hat']):.1f}]")
                print(f"  Spread range: [{np.min(results['spread_bp']):.1f}, {np.max(results['spread_bp']):.1f}] bp")
        
        return results
