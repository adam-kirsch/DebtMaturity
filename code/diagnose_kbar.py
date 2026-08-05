"""
Diagnostic: Check if K̄* actually changes with parameters.
Print numerical values to verify if differences exist.
"""

import numpy as np
from config import Params
from cir_process import CIRProcess
from bond_valuation import BondValuation
from refinancing_solver import RefinancingSetSolver


def diagnose_kbar_sensitivity():
    """Check K̄* sensitivity to parameters."""
    
    print("\n" + "="*70)
    print("DIAGNOSTIC: K̄* Sensitivity Analysis")
    print("="*70)
    
    # Common setup
    x_test = 10.0
    x_grid = np.linspace(0.5, 25, 20)
    K_grid = np.linspace(0, 500, 30)
    T_grid = np.linspace(0.5, 50, 20)
    
    # ========== Test 1: Volatility ==========
    print("\n" + "-"*70)
    print("TEST 1: Volatility (σ)")
    print("-"*70)
    
    sigma_values = [0.40, 0.50, 0.60, 0.70]
    K_bar_at_x = []
    
    for sigma in sigma_values:
        p = Params()
        p.sigma = sigma
        
        if 2 * p.kappa * p.mu < sigma**2:
            print(f"σ={sigma}: Violates Feller condition, skipping")
            continue
        
        cir = CIRProcess(p.kappa, p.mu, sigma)
        bond_val = BondValuation(cir, p.r, p.eta, p.C)
        ref_solver = RefinancingSetSolver(bond_val, x_grid, K_grid, T_grid)
        K_bar_array, _ = ref_solver.solve(tol=1e-3, max_iter=30, verbose=False)
        
        K_bar_value = np.interp(x_test, x_grid, K_bar_array)
        K_bar_at_x.append(K_bar_value)
        
        print(f"σ = {sigma:.2f}: K̄*({x_test}) = {K_bar_value:.4f}")
    
    if len(K_bar_at_x) > 1:
        diff = max(K_bar_at_x) - min(K_bar_at_x)
        pct_change = (diff / np.mean(K_bar_at_x)) * 100
        print(f"\nRange: {min(K_bar_at_x):.2f} to {max(K_bar_at_x):.2f}")
        print(f"Difference: {diff:.4f} ({pct_change:.2f}% of mean)")
        
        if pct_change < 1:
            print("⚠️  WARNING: Less than 1% variation - may be hard to see in plots!")
    
    # ========== Test 2: Illiquidity Duration ==========
    print("\n" + "-"*70)
    print("TEST 2: Illiquidity Duration (η)")
    print("-"*70)
    
    eta_values = [0.1, 0.5, 1.0, 5.0]
    K_bar_at_x = []
    
    for eta in eta_values:
        p = Params()
        p.eta = eta
        
        cir = CIRProcess(p.kappa, p.mu, p.sigma)
        bond_val = BondValuation(cir, p.r, p.eta, p.C)
        ref_solver = RefinancingSetSolver(bond_val, x_grid, K_grid, T_grid)
        K_bar_array, _ = ref_solver.solve(tol=1e-3, max_iter=30, verbose=False)
        
        K_bar_value = np.interp(x_test, x_grid, K_bar_array)
        K_bar_at_x.append(K_bar_value)
        
        print(f"η = {eta:.2f} (duration={1/eta:.1f}): K̄*({x_test}) = {K_bar_value:.4f}")
    
    diff = max(K_bar_at_x) - min(K_bar_at_x)
    pct_change = (diff / np.mean(K_bar_at_x)) * 100
    print(f"\nRange: {min(K_bar_at_x):.2f} to {max(K_bar_at_x):.2f}")
    print(f"Difference: {diff:.4f} ({pct_change:.2f}% of mean)")
    
    if pct_change > 10:
        print("✓ Significant variation - should be visible in plots")
    elif pct_change > 1:
        print("⚠️  Moderate variation - may need zoom or different scale")
    else:
        print("⚠️  WARNING: Less than 1% variation!")
    
    # ========== Test 3: Interest Rate ==========
    print("\n" + "-"*70)
    print("TEST 3: Interest Rate (r)")
    print("-"*70)
    
    r_values = [0.02, 0.03, 0.04, 0.05]
    K_bar_at_x = []
    
    for r in r_values:
        p = Params()
        p.r = r
        
        cir = CIRProcess(p.kappa, p.mu, p.sigma)
        bond_val = BondValuation(cir, r, p.eta, p.C)
        ref_solver = RefinancingSetSolver(bond_val, x_grid, K_grid, T_grid)
        K_bar_array, _ = ref_solver.solve(tol=1e-3, max_iter=30, verbose=False)
        
        K_bar_value = np.interp(x_test, x_grid, K_bar_array)
        K_bar_at_x.append(K_bar_value)
        
        print(f"r = {r:.3f}: K̄*({x_test}) = {K_bar_value:.4f}")
    
    diff = max(K_bar_at_x) - min(K_bar_at_x)
    pct_change = (diff / np.mean(K_bar_at_x)) * 100
    print(f"\nRange: {min(K_bar_at_x):.2f} to {max(K_bar_at_x):.2f}")
    print(f"Difference: {diff:.4f} ({pct_change:.2f}% of mean)")
    
    if pct_change > 5:
        print("✓ Good variation - should be visible in plots")
    else:
        print("⚠️  Small variation - lines may overlap in plots")
    
    # ========== Test 4: Leverage (K₀) ==========
    print("\n" + "-"*70)
    print("TEST 4: Leverage (K₀)")
    print("-"*70)
    
    K0_values = [100, 200, 300, 400]
    K_bar_at_x = []
    
    for K0 in K0_values:
        p = Params()
        p.K_0 = K0  # This shouldn't affect K̄*
        
        cir = CIRProcess(p.kappa, p.mu, p.sigma)
        bond_val = BondValuation(cir, p.r, p.eta, p.C)
        ref_solver = RefinancingSetSolver(bond_val, x_grid, K_grid, T_grid)
        K_bar_array, _ = ref_solver.solve(tol=1e-3, max_iter=30, verbose=False)
        
        K_bar_value = np.interp(x_test, x_grid, K_bar_array)
        K_bar_at_x.append(K_bar_value)
        
        print(f"K₀ = {K0}: K̄*({x_test}) = {K_bar_value:.4f}")
    
    diff = max(K_bar_at_x) - min(K_bar_at_x)
    pct_change = (diff / np.mean(K_bar_at_x)) * 100
    print(f"\nRange: {min(K_bar_at_x):.2f} to {max(K_bar_at_x):.2f}")
    print(f"Difference: {diff:.4f} ({pct_change:.2f}% of mean)")
    
    if pct_change < 0.1:
        print("✓ CORRECT: K̄* should NOT depend on K₀ (it's independent)")
    else:
        print("⚠️  WARNING: K̄* should be constant with K₀!")
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    diagnose_kbar_sensitivity()
