"""Find optimal bond (K̂, T̂)"""

import numpy as np
from scipy.interpolate import interp1d
from firm_valuation import FirmValuation
from optimization_methods import BondOptimizer


class OptimalBondSolver:
    """Find optimal bond choice for given earnings level."""
    
    def __init__(self, bond_valuation, K_bar_star_func, method='grid_search'):
        """
        Parameters
        ----------
        bond_valuation : BondValuation
        K_bar_star_func : callable
            Refinancing capacity K̄*(x)
        method : str
            Optimization method: 'grid_search', 'slsqp', 'trust-constr', 'hybrid'
            Default: 'grid_search'
        """
        self.bond_val = bond_valuation
        self.K_bar_star_func = K_bar_star_func
        self.firm_val = FirmValuation(bond_valuation)
        self.method = method
        
        # Create optimizer instance
        self.optimizer = BondOptimizer(
            bond_valuation,
            self.firm_val,
            K_bar_star_func,
            self.V_bar_approximation
        )
    
    def V_bar_approximation(self, x, K):
        """
        Approximate continuation value V̄*(x, K) at maturity.
        
        Represents firm value if it reaches maturity with earnings x
        and outstanding debt K.
        
        Parameters
        ----------
        x : float
            Earnings at maturity
        K : float
            Face value outstanding
            
        Returns
        -------
        float
            Approximate continuation value
        """
        F_x = self.bond_val.F_unlevered(x)
        
        # Check if firm can refinance at maturity
        if self.K_bar_star_func(x) >= K + self.bond_val.C:
            # Can refinance: firm continues, gets unlevered value minus debt
            return F_x - K
        else:
            # Cannot refinance: potential default, get residual value
            return max(0, F_x - K)
    
    def find_optimal_bond(self, x, K_0, K_grid=None, T_grid=None, verbose=False):
        """
        Find (K̂, T̂) that maximizes firm value V^I.
        
        Uses the optimization method specified in __init__.
        
        Parameters
        ----------
        x : float
            Current earnings
        K_0 : float
            Amount needed to raise
        K_grid : array, optional
            Face values to search (required for grid_search)
        T_grid : array, optional
            Maturities to search (required for grid_search)
        verbose : bool
            Print progress
            
        Returns
        -------
        K_opt : float
            Optimal face value
        T_opt : float
            Optimal maturity
        B_opt : float
            Bond value
        """
        # Use BondOptimizer based on selected method
        if self.method == 'grid_search':
            if K_grid is None or T_grid is None:
                raise ValueError("grid_search requires K_grid and T_grid")
            result = self.optimizer.optimize(x, K_0, method='grid_search',
                                            K_grid=K_grid, T_grid=T_grid, verbose=verbose)
        else:
            # For other methods, pass bounds based on grids (if provided)
            if K_grid is not None and T_grid is not None:
                bounds = [(np.min(K_grid), np.max(K_grid)),
                         (np.min(T_grid), np.max(T_grid))]
            else:
                bounds = [(K_0, 500), (0.1, 50)]  # Default bounds
            
            result = self.optimizer.optimize(x, K_0, method=self.method,
                                            bounds=bounds, verbose=verbose)
        
        if result['success']:
            return result['K'], result['T'], result['B']
        else:
            if verbose:
                print(f"  ✗ No feasible bond at x={x:.2f} (cannot raise K_0={K_0})")
            return None, None, None
    
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
            print(f"Method: {self.method}")
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

