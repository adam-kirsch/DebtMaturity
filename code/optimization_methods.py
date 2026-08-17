"""
Optimization algorithms for finding optimal bond (K̂, T̂).

Contains multiple optimization methods:
1. Grid Search (exhaustive)
2. SLSQP (local, constrained)
3. Trust-Constr (robust, constrained)
4. Hybrid (global + local refinement)
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint


class BondOptimizer:
    """
    Optimization algorithms for bond selection problem.
    
    Problem: 
        max_{K,T} V^I(x; K, T)
        s.t.   B^I(x; K, T) ≥ K_0
               K_0 ≤ K ≤ K_max
               T_min ≤ T ≤ T_max
    """
    
    def __init__(self, bond_valuation, firm_valuation, K_bar_func, V_bar_func):
        """
        Parameters
        ----------
        bond_valuation : BondValuation
            Bond valuation object
        firm_valuation : FirmValuation
            Firm valuation object
        K_bar_func : callable
            Refinancing capacity function K̄*(x)
        V_bar_func : callable
            Continuation value function V̄*(x, K)
        """
        self.bond_val = bond_valuation
        self.firm_val = firm_valuation
        self.K_bar_func = K_bar_func
        self.V_bar_func = V_bar_func
    
    def _objective(self, vars, x):
        """
        Objective function: -V^I (negative for minimization).
        
        Parameters
        ----------
        vars : array [K, T]
            Bond parameters
        x : float
            Current earnings
            
        Returns
        -------
        float
            Negative firm value
        """
        K, T = vars
        V_I = self.firm_val.V_illiquid(x, T, K, self.K_bar_func, self.V_bar_func)
        return -V_I  # Negative because scipy minimizes
    
    def _constraint_feasibility(self, vars, x, K_0):
        """
        Feasibility constraint: B^I(x; K, T) - K_0 ≥ 0
        
        Parameters
        ----------
        vars : array [K, T]
            Bond parameters
        x : float
            Current earnings
        K_0 : float
            Required amount to raise
            
        Returns
        -------
        float
            Constraint value (positive = satisfied)
        """
        K, T = vars
        B_I = self.bond_val.B_illiquid(x, T, K, self.K_bar_func)
        return B_I - K_0
    
    def optimize_grid_search(self, x, K_0, K_grid, T_grid, verbose=False):
        """
        Grid search optimization (exhaustive).
        
        Evaluates V^I at every (K, T) combination.
        Guaranteed to find global optimum within grid resolution.
        
        Parameters
        ----------
        x : float
            Current earnings
        K_0 : float
            Required amount to raise
        K_grid : array
            Face value grid
        T_grid : array
            Maturity grid
        verbose : bool
            Print progress
            
        Returns
        -------
        dict
            Optimization result with keys: K, T, B, V, success, nfev, method
        """
        if verbose:
            print(f"  Grid search: {len(K_grid)} × {len(T_grid)} = {len(K_grid)*len(T_grid)} evaluations")
        
        best_V = -np.inf
        best_K, best_T, best_B = None, None, None
        feasible_found = False
        nfev = 0
        
        for K in K_grid:
            for T in T_grid:
                nfev += 1
                
                # Check feasibility
                B_I = self.bond_val.B_illiquid(x, T, K, self.K_bar_func)
                if B_I < K_0:
                    continue
                
                feasible_found = True
                
                # Evaluate objective
                V_I = self.firm_val.V_illiquid(x, T, K, self.K_bar_func, self.V_bar_func)
                
                if V_I > best_V:
                    best_V = V_I
                    best_K = K
                    best_T = T
                    best_B = B_I
        
        return {
            'K': best_K,
            'T': best_T,
            'B': best_B,
            'V': best_V,
            'success': feasible_found,
            'nfev': nfev,
            'method': 'grid_search'
        }
    
    def optimize_slsqp(self, x, K_0, bounds=None, x0=None, verbose=False):
        """
        SLSQP optimization (Sequential Least Squares Programming).
        
        Local optimizer with constraint handling.
        Fast convergence (typically 10-30 iterations).
        
        Parameters
        ----------
        x : float
            Current earnings
        K_0 : float
            Required amount to raise
        bounds : tuple of tuples, optional
            ((K_min, K_max), (T_min, T_max))
            Default: ((K_0, 500), (0.1, 50))
        x0 : array, optional
            Initial guess [K_init, T_init]
            Default: [K_0 * 1.2, 5.0]
        verbose : bool
            Print optimization progress
            
        Returns
        -------
        dict
            Optimization result
        """
        if bounds is None:
            bounds = [(K_0, 500), (0.1, 50)]
        
        if x0 is None:
            x0 = [K_0 * 1.2, 5.0]
        
        if verbose:
            print(f"  SLSQP: Starting from K={x0[0]:.1f}, T={x0[1]:.1f}")
        
        # Constraint: B^I >= K_0
        constraint = {
            'type': 'ineq',
            'fun': lambda vars: self._constraint_feasibility(vars, x, K_0)
        }
        
        result = minimize(
            self._objective,
            x0=x0,
            args=(x,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraint,
            options={'maxiter': 100, 'ftol': 1e-8, 'disp': verbose}
        )
        
        if result.success:
            K_opt, T_opt = result.x
            B_opt = self.bond_val.B_illiquid(x, T_opt, K_opt, self.K_bar_func)
            V_opt = -result.fun
        else:
            K_opt, T_opt, B_opt, V_opt = None, None, None, None
        
        return {
            'K': K_opt,
            'T': T_opt,
            'B': B_opt,
            'V': V_opt,
            'success': result.success,
            'nfev': result.nfev,
            'message': result.message,
            'method': 'SLSQP'
        }
    
    
    def optimize_nelder_mead(self, x, K_0, bounds=None, x0=None, verbose=False):
        """
        Nelder-Mead optimization (Simplex method).
        
        Derivative-free local optimizer.
        Works well for non-smooth objectives.
        Uses penalty method for constraints.
        
        Parameters
        ----------
        x : float
            Current earnings
        K_0 : float
            Required amount to raise
        bounds : tuple of tuples, optional
            ((K_min, K_max), (T_min, T_max))
            Default: ((K_0, 500), (0.1, 50))
        x0 : array, optional
            Initial guess [K_init, T_init]
            Default: [K_0 * 1.2, 5.0]
        verbose : bool
            Print optimization progress
            
        Returns
        -------
        dict
            Optimization result
        """
        if bounds is None:
            bounds = [(K_0, 500), (0.1, 50)]
        
        if x0 is None:
            x0 = [K_0 * 1.2, 5.0]
        
        if verbose:
            print(f"  Nelder-Mead: Starting from K={x0[0]:.1f}, T={x0[1]:.1f}")
        
        def objective_with_penalty(vars):
            """Objective with penalty for infeasibility and bounds."""
            K, T = vars
            
            # Check bounds
            if K < bounds[0][0] or K > bounds[0][1]:
                return 1e10
            if T < bounds[1][0] or T > bounds[1][1]:
                return 1e10
            
            # Check feasibility constraint
            B_I = self.bond_val.B_illiquid(x, T, K, self.K_bar_func)
            if B_I < K_0:
                # Penalize infeasible solutions
                penalty = 1e6 * (K_0 - B_I)
                return 1e10 + penalty
            
            # Evaluate objective
            return self._objective(vars, x)
        
        result = minimize(
            objective_with_penalty,
            x0=x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': verbose}
        )
        
        if result.success:
            K_opt, T_opt = result.x
            B_opt = self.bond_val.B_illiquid(x, T_opt, K_opt, self.K_bar_func)
            V_opt = -self._objective(result.x, x)  # Get true objective (not penalized)
            
            # Verify final feasibility
            if B_opt < K_0 - 1e-6:
                if verbose:
                    print(f"  Warning: Nelder-Mead returned infeasible solution (B={B_opt:.2f} < K_0={K_0:.2f})")
                K_opt, T_opt, B_opt, V_opt = None, None, None, None
                result.success = False
        else:
            K_opt, T_opt, B_opt, V_opt = None, None, None, None
        
        return {
            'K': K_opt,
            'T': T_opt,
            'B': B_opt,
            'V': V_opt,
            'success': result.success,
            'nfev': result.nfev,
            'message': 'Optimization terminated successfully.' if result.success else 'Failed',
            'method': 'Nelder-Mead'
        }
    
    def optimize_hybrid(self, x, K_0, bounds=None, verbose=False):
        """
        Hybrid optimization: Differential Evolution → SLSQP.
        
        Two-stage approach:
        1. Global search with Differential Evolution (coarse)
        2. Local refinement with SLSQP (precise)
        
        Best for finding true global optimum.
        
        Parameters
        ----------
        x : float
            Current earnings
        K_0 : float
            Required amount to raise
        bounds : tuple of tuples, optional
            ((K_min, K_max), (T_min, T_max))
        verbose : bool
            Print optimization progress
            
        Returns
        -------
        dict
            Optimization result
        """
        if bounds is None:
            bounds = [(K_0, 500), (0.1, 50)]
        
        # STAGE 1: Global search with Differential Evolution
        if verbose:
            print(f"  Stage 1: Differential Evolution (global search)...")
        
        def objective_with_penalty(vars):
            """Objective with penalty for infeasibility."""
            K, T = vars
            
            # Check feasibility
            B_I = self.bond_val.B_illiquid(x, T, K, self.K_bar_func)
            if B_I < K_0:
                return 1e10  # Large penalty
            
            # Evaluate objective
            return self._objective(vars, x)
        
        result_global = differential_evolution(
            objective_with_penalty,
            bounds=bounds,
            maxiter=50,      # Coarse search
            popsize=10,      # Population size
            seed=42,
            workers=1,       # Can parallelize if needed
            polish=False,    # Don't polish (we'll do SLSQP)
            disp=verbose
        )
        
        nfev_global = result_global.nfev
        
        # STAGE 2: Local refinement with SLSQP
        if verbose:
            print(f"  Stage 2: SLSQP (local refinement from K={result_global.x[0]:.1f}, T={result_global.x[1]:.1f})...")
        
        result_local = self.optimize_slsqp(
            x, K_0,
            bounds=bounds,
            x0=result_global.x,
            verbose=False
        )
        
        # Combine results
        result_local['nfev'] += nfev_global
        result_local['method'] = 'hybrid (DE+SLSQP)'
        
        if verbose and result_local['success']:
            print(f"  ✓ Converged: K={result_local['K']:.2f}, T={result_local['T']:.2f}, V={result_local['V']:.2f}")
        
        return result_local
    
    def optimize(self, x, K_0, method='grid_search', **kwargs):
        """
        Unified interface for all optimization methods.
        
        Parameters
        ----------
        x : float
            Current earnings
        K_0 : float
            Required amount to raise
        method : str
                        Optimization method:
            - 'grid_search': Exhaustive grid search
            - 'slsqp': Sequential Least Squares Programming
            - 'trust-constr': Trust-region constrained
            - 'nelder-mead': Nelder-Mead simplex (derivative-free)
            - 'hybrid': Differential Evolution + SLSQP
        **kwargs : dict
            Method-specific arguments
            
        Returns
        -------
        dict
            Optimization result with keys:
            - K: Optimal face value
            - T: Optimal maturity
            - B: Bond value
            - V: Firm value
            - success: Whether optimization succeeded
            - nfev: Number of function evaluations
            - method: Method used
            
        Examples
        --------
        >>> optimizer = BondOptimizer(bond_val, firm_val, K_bar_func, V_bar_func)
        >>> result = optimizer.optimize(x=10, K_0=200, method='slsqp')
        >>> print(f"Optimal: K={result['K']:.2f}, T={result['T']:.2f}")
        """
        if method == 'grid_search':
            return self.optimize_grid_search(x, K_0, **kwargs)
        elif method == 'slsqp':
            return self.optimize_slsqp(x, K_0, **kwargs)
        elif method == 'trust-constr':
            return self.optimize_trust_constr(x, K_0, **kwargs)
        elif method == 'nelder-mead':
            return self.optimize_nelder_mead(x, K_0, **kwargs)
        elif method == 'hybrid':
            return self.optimize_hybrid(x, K_0, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from: grid_search, slsqp, trust-constr, nelder-mead, hybrid")


def compare_methods(x, K_0, bond_val, firm_val, K_bar_func, V_bar_func, K_grid, T_grid):
    """
    Compare all optimization methods on same problem.
    
    Useful for validation and benchmarking.
    
    Parameters
    ----------
    x : float
        Current earnings
    K_0 : float
        Required amount to raise
    bond_val : BondValuation
    firm_val : FirmValuation
    K_bar_func : callable
    V_bar_func : callable
    K_grid : array
        Grid for grid search
    T_grid : array
        Grid for grid search
        
    Returns
    -------
    dict
        Results from all methods
    """
    import time
    
    optimizer = BondOptimizer(bond_val, firm_val, K_bar_func, V_bar_func)
    
    results = {}
    
    print("\n" + "="*70)
    print(f"COMPARING OPTIMIZATION METHODS (x={x}, K_0={K_0})")
    print("="*70)
    
    # Grid Search
    print("\n1. Grid Search:")
    t0 = time.time()
    results['grid'] = optimizer.optimize(x, K_0, method='grid_search', 
                                         K_grid=K_grid, T_grid=T_grid, verbose=True)
    results['grid']['time'] = time.time() - t0
    print(f"   Result: K={results['grid']['K']:.2f}, T={results['grid']['T']:.2f}, "
          f"V={results['grid']['V']:.2f}, nfev={results['grid']['nfev']}, time={results['grid']['time']:.2f}s")
    
    # SLSQP
    print("\n2. SLSQP:")
    t0 = time.time()
    results['slsqp'] = optimizer.optimize(x, K_0, method='slsqp', verbose=True)
    results['slsqp']['time'] = time.time() - t0
    if results['slsqp']['success']:
        print(f"   Result: K={results['slsqp']['K']:.2f}, T={results['slsqp']['T']:.2f}, "
              f"V={results['slsqp']['V']:.2f}, nfev={results['slsqp']['nfev']}, time={results['slsqp']['time']:.2f}s")
    else:
        print(f"   Failed: {results['slsqp']['message']}")
    
    # Nelder-Mead
    print("\n3. Nelder-Mead:")
    t0 = time.time()
    results['nelder'] = optimizer.optimize(x, K_0, method='nelder-mead', verbose=True)
    results['nelder']['time'] = time.time() - t0
    if results['nelder']['success']:
        print(f"   Result: K={results['nelder']['K']:.2f}, T={results['nelder']['T']:.2f}, "
              f"V={results['nelder']['V']:.2f}, nfev={results['nelder']['nfev']}, time={results['nelder']['time']:.2f}s")
    else:
        print(f"   Failed: {results['nelder']['message']}")
    
    # Hybrid
    print("\n4. Hybrid (DE + SLSQP):")
    t0 = time.time()
    results['hybrid'] = optimizer.optimize(x, K_0, method='hybrid', verbose=True)
    results['hybrid']['time'] = time.time() - t0
    if results['hybrid']['success']:
        print(f"   Result: K={results['hybrid']['K']:.2f}, T={results['hybrid']['T']:.2f}, "
              f"V={results['hybrid']['V']:.2f}, nfev={results['hybrid']['nfev']}, time={results['hybrid']['time']:.2f}s")
    else:
        print(f"   Failed")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print(f"{'Method':<15} {'K̂':<10} {'T̂':<10} {'V^I':<12} {'Time':<10} {'Speedup':<10}")
    print("-"*70)
    
    base_time = results['grid']['time']
    for name, res in results.items():
        if res['success']:
            speedup = base_time / res['time'] if res['time'] > 0 else np.inf
            print(f"{name:<15} {res['K']:<10.2f} {res['T']:<10.2f} {res['V']:<12.2f} "
                  f"{res['time']:<10.2f} {speedup:<10.1f}x")
    
    return results


if __name__ == '__main__':
    """Run comparison when file is executed directly."""
    
    print("\n" + "="*70)
    print("OPTIMIZATION METHODS COMPARISON")
    print("="*70)
    
    # Setup
    from config import Params
    from cir_process import CIRProcess
    from bond_valuation import BondValuation
    from firm_valuation import FirmValuation
    from refinancing_solver import RefinancingSetSolver
    
    p = Params()
    cir = CIRProcess(p.kappa, p.mu, p.sigma)
    bond_val = BondValuation(cir, p.r, p.eta, p.C)
    firm_val = FirmValuation(bond_val)
    
    # Solve for K̄*(x)
    print("\nSolving for K̄*(x)...")
    x_grid = np.linspace(p.x_min, 25, 20)
    K_grid_solve = np.linspace(p.K_min, p.K_max, 20)
    T_grid_solve = np.linspace(p.T_min, p.T_max, 15)
    
    ref_solver = RefinancingSetSolver(bond_val, x_grid, K_grid_solve, T_grid_solve)
    _, K_bar_func = ref_solver.solve(tol=1e-3, max_iter=30, verbose=False)
    print("✓ K̄*(x) computed")
    
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
    
    # Grids for grid search (moderate resolution)
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
    
    print("\n" + "="*70)
    print("COMPLETE ✓")
    print("="*70)
    print("\nTo use in other scripts:")
    print("  from optimization_methods import BondOptimizer")
    print("  optimizer = BondOptimizer(bond_val, firm_val, K_bar_func, V_bar_func)")
    print("  result = optimizer.optimize(x=10, K_0=200, method='slsqp')")


