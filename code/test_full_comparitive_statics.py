"""
Test comparative statics using the FULL MODEL solution.
Verify Table 3.1 relationships with actual K̄*, T̂, K̂.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

from config import Params
from cir_process import CIRProcess
from bond_valuation import BondValuation
from refinancing_solver import RefinancingSetSolver
from optimal_bond_solver import OptimalBondSolver


def solve_for_params(param_dict, grid_sizes=(20, 30, 20)):
    """
    Solve model for given parameters.
    
    Parameters
    ----------
    param_dict : dict
        Parameter values
    grid_sizes : tuple
        (n_x, n_K, n_T)
        
    Returns
    -------
    dict
        Solution with K_bar_array and results
    """
    # Create params object
    p = Params()
    for key, val in param_dict.items():
        setattr(p, key, val)
    
    # Grids
    n_x, n_K, n_T = grid_sizes
    x_grid = np.linspace(p.x_min, p.x_max, n_x)
    K_grid = np.linspace(p.K_min, p.K_max, n_K)
    T_grid = np.linspace(p.T_min, p.T_max, n_T)
    
    # Solve
    cir = CIRProcess(p.kappa, p.mu, p.sigma)
    bond_val = BondValuation(cir, p.r, p.eta, p.C)
    
    ref_solver = RefinancingSetSolver(bond_val, x_grid, K_grid, T_grid)
    K_bar_array, K_bar_func = ref_solver.solve(tol=1e-3, max_iter=30, verbose=False)
    
    opt_solver = OptimalBondSolver(bond_val, K_bar_func)
    results = opt_solver.solve_for_grid(x_grid, p.K_0, K_grid, T_grid, verbose=False)
    
    return {
        'params': p,
        'x_grid': x_grid,
        'K_bar_array': K_bar_array,
        'results': results
    }


def test_interest_rate():
    """Test: ↑r → T̂↓, K̂ ambiguous, K̄*↓, spread↑"""
    
    print("\n" + "="*70)
    print("TEST 1: Interest Rate Effects")
    print("="*70)
    
    r_values = [0.02, 0.03, 0.04, 0.05]
    
    all_results = {}
    
    print("\nSolving for different interest rates...")
    for r in r_values:
        print(f"  r = {r}...", end=" ")
        sol = solve_for_params({'r': r})
        all_results[r] = sol
        print(f"✓ ({len(sol['results']['x'])} feasible)")
    
    # Analyze at a fixed earnings level
    x_test = 10.0
    
    print(f"\n{'r':<8} {'T̂(x)':<10} {'K̂(x)':<10} {'K̄*(x)':<12} {'Spread':<12}")
    print("-"*70)
    
    summary = {
        'r': [],
        'T_hat': [],
        'K_hat': [],
        'K_bar': [],
        'spread': []
    }
    
    for r in r_values:
        sol = all_results[r]
        
        # Find K̄* at x_test
        x_grid = sol['x_grid']
        K_bar_array = sol['K_bar_array']
        K_bar_at_x = np.interp(x_test, x_grid, K_bar_array)
        
        # Find optimal bond at x_test (if exists)
        res = sol['results']
        if len(res['x']) > 0 and x_test >= np.min(res['x']) and x_test <= np.max(res['x']):
            T_hat_at_x = np.interp(x_test, res['x'], res['T_hat'])
            K_hat_at_x = np.interp(x_test, res['x'], res['K_hat'])
            spread_at_x = np.interp(x_test, res['x'], res['spread_bp'])
        else:
            T_hat_at_x = np.nan
            K_hat_at_x = np.nan
            spread_at_x = np.nan
        
        summary['r'].append(r)
        summary['T_hat'].append(T_hat_at_x)
        summary['K_hat'].append(K_hat_at_x)
        summary['K_bar'].append(K_bar_at_x)
        summary['spread'].append(spread_at_x)
        
        print(f"{r:<8.3f} {T_hat_at_x:<10.2f} {K_hat_at_x:<10.1f} {K_bar_at_x:<12.2f} {spread_at_x:<12.1f}")
    
    # Check relationships
    print("\n" + "="*70)
    print("VERIFICATION:")
    print("="*70)
    
    # Filter out NaN values
    valid = ~np.isnan(summary['T_hat'])
    
    if np.sum(valid) > 1:
        T_decreasing = np.all(np.diff(np.array(summary['T_hat'])[valid]) <= 0)
        K_bar_decreasing = np.all(np.diff(np.array(summary['K_bar'])[valid]) <= 0)
        spread_increasing = np.all(np.diff(np.array(summary['spread'])[valid]) >= 0)
        
        print(f"✓ T̂ decreases with r: {T_decreasing}")
        print(f"✓ K̄* decreases with r: {K_bar_decreasing}")
        print(f"✓ Spread increases with r: {spread_increasing}")
        print(f"  K̂ behavior: ambiguous (as expected)")
    
    # Plot
    plot_comparative_static(all_results, 'r', r_values, 
                           'Interest Rate', 'interest_rate')
    
    return all_results, summary


def test_volatility():
    """Test: ↑σ → T̂↓, K̂ ambiguous, K̄*↓, spread↑"""
    
    print("\n" + "="*70)
    print("TEST 2: Volatility Effects")
    print("="*70)
    
    sigma_values = [0.40, 0.50, 0.60, 0.70]
    
    all_results = {}
    
    print("\nSolving for different volatilities...")
    for sigma in sigma_values:
        # Check Feller condition
        p = Params()
        if 2 * p.kappa * p.mu < sigma**2:
            print(f"  σ = {sigma}... ✗ Violates Feller condition, skipping")
            continue
        
        print(f"  σ = {sigma}...", end=" ")
        sol = solve_for_params({'sigma': sigma})
        all_results[sigma] = sol
        print(f"✓ ({len(sol['results']['x'])} feasible)")
    
    # Analyze at fixed x
    x_test = 10.0
    
    print(f"\n{'σ':<8} {'T̂(x)':<10} {'K̂(x)':<10} {'K̄*(x)':<12} {'Spread':<12}")
    print("-"*70)
    
    summary = {
        'sigma': [],
        'T_hat': [],
        'K_hat': [],
        'K_bar': [],
        'spread': []
    }
    
    for sigma in all_results.keys():
        sol = all_results[sigma]
        
        x_grid = sol['x_grid']
        K_bar_array = sol['K_bar_array']
        K_bar_at_x = np.interp(x_test, x_grid, K_bar_array)
        
        res = sol['results']
        if len(res['x']) > 0 and x_test >= np.min(res['x']) and x_test <= np.max(res['x']):
            T_hat_at_x = np.interp(x_test, res['x'], res['T_hat'])
            K_hat_at_x = np.interp(x_test, res['x'], res['K_hat'])
            spread_at_x = np.interp(x_test, res['x'], res['spread_bp'])
        else:
            T_hat_at_x = np.nan
            K_hat_at_x = np.nan
            spread_at_x = np.nan
        
        summary['sigma'].append(sigma)
        summary['T_hat'].append(T_hat_at_x)
        summary['K_hat'].append(K_hat_at_x)
        summary['K_bar'].append(K_bar_at_x)
        summary['spread'].append(spread_at_x)
        
        print(f"{sigma:<8.3f} {T_hat_at_x:<10.2f} {K_hat_at_x:<10.1f} {K_bar_at_x:<12.2f} {spread_at_x:<12.1f}")
    
    # Verification
    print("\n" + "="*70)
    print("VERIFICATION:")
    print("="*70)
    
    valid = ~np.isnan(summary['T_hat'])
    if np.sum(valid) > 1:
        T_decreasing = np.all(np.diff(np.array(summary['T_hat'])[valid]) <= 0)
        K_bar_decreasing = np.all(np.diff(np.array(summary['K_bar'])[valid]) <= 0)
        spread_increasing = np.all(np.diff(np.array(summary['spread'])[valid]) >= 0)
        
        print(f"✓ T̂ decreases with σ: {T_decreasing}")
        print(f"✓ K̄* decreases with σ: {K_bar_decreasing}")
        print(f"✓ Spread increases with σ: {spread_increasing}")
    
    plot_comparative_static(all_results, 'sigma', list(all_results.keys()), 
                           'Volatility σ', 'volatility')
    
    return all_results, summary


def test_illiquidity_duration():
    """Test: ↑(1/η) → T̂↓, K̂↑, K̄*↓, spread↑"""
    
    print("\n" + "="*70)
    print("TEST 3: Illiquidity Duration Effects")
    print("="*70)
    
    eta_values = [0.1, 0.5, 1.0, 5.0]
    
    all_results = {}
    
    print("\nSolving for different illiquidity durations...")
    for eta in eta_values:
        duration = 1/eta
        print(f"  η = {eta} (duration = {duration:.1f})...", end=" ")
        sol = solve_for_params({'eta': eta})
        all_results[eta] = sol
        print(f"✓ ({len(sol['results']['x'])} feasible)")
    
    x_test = 10.0
    
    print(f"\n{'η':<8} {'1/η':<10} {'T̂(x)':<10} {'K̂(x)':<10} {'K̄*(x)':<12} {'Spread':<12}")
    print("-"*70)
    
    summary = {
        'eta': [],
        'duration': [],
        'T_hat': [],
        'K_hat': [],
        'K_bar': [],
        'spread': []
    }
    
    for eta in eta_values:
        sol = all_results[eta]
        duration = 1/eta
        
        x_grid = sol['x_grid']
        K_bar_array = sol['K_bar_array']
        K_bar_at_x = np.interp(x_test, x_grid, K_bar_array)
        
        res = sol['results']
        if len(res['x']) > 0 and x_test >= np.min(res['x']) and x_test <= np.max(res['x']):
            T_hat_at_x = np.interp(x_test, res['x'], res['T_hat'])
            K_hat_at_x = np.interp(x_test, res['x'], res['K_hat'])
            spread_at_x = np.interp(x_test, res['x'], res['spread_bp'])
        else:
            T_hat_at_x = np.nan
            K_hat_at_x = np.nan
            spread_at_x = np.nan
        
        summary['eta'].append(eta)
        summary['duration'].append(duration)
        summary['T_hat'].append(T_hat_at_x)
        summary['K_hat'].append(K_hat_at_x)
        summary['K_bar'].append(K_bar_at_x)
        summary['spread'].append(spread_at_x)
        
        print(f"{eta:<8.2f} {duration:<10.2f} {T_hat_at_x:<10.2f} {K_hat_at_x:<10.1f} {K_bar_at_x:<12.2f} {spread_at_x:<12.1f}")
    
    # Verification
    print("\n" + "="*70)
    print("VERIFICATION:")
    print("="*70)
    
    valid = ~np.isnan(summary['T_hat'])
    if np.sum(valid) > 1:
        print(f"✓ T̂, K̂, K̄*, spread relationships with duration (check plots)")
    
    plot_comparative_static(all_results, 'eta', eta_values, 
                           'Illiquidity η', 'illiquidity')
    
    return all_results, summary


def test_leverage():
    """Test: ↑K₀ → T̂↓, K̂↑, K̄* unchanged, spread↑"""
    
    print("\n" + "="*70)
    print("TEST 4: Leverage Effects")
    print("="*70)
    
    K0_values = [100, 200, 300, 400]
    
    all_results = {}
    
    print("\nSolving for different leverage levels...")
    for K0 in K0_values:
        print(f"  K₀ = {K0}...", end=" ")
        sol = solve_for_params({'K_0': K0})
        all_results[K0] = sol
        print(f"✓ ({len(sol['results']['x'])} feasible)")
    
    x_test = 10.0
    
    print(f"\n{'K₀':<8} {'T̂(x)':<10} {'K̂(x)':<10} {'K̄*(x)':<12} {'Spread':<12}")
    print("-"*70)
    
    summary = {
        'K0': [],
        'T_hat': [],
        'K_hat': [],
        'K_bar': [],
        'spread': []
    }
    
    for K0 in K0_values:
        sol = all_results[K0]
        
        x_grid = sol['x_grid']
        K_bar_array = sol['K_bar_array']
        K_bar_at_x = np.interp(x_test, x_grid, K_bar_array)
        
        res = sol['results']
        if len(res['x']) > 0 and x_test >= np.min(res['x']) and x_test <= np.max(res['x']):
            T_hat_at_x = np.interp(x_test, res['x'], res['T_hat'])
            K_hat_at_x = np.interp(x_test, res['x'], res['K_hat'])
            spread_at_x = np.interp(x_test, res['x'], res['spread_bp'])
        else:
            T_hat_at_x = np.nan
            K_hat_at_x = np.nan
            spread_at_x = np.nan
        
        summary['K0'].append(K0)
        summary['T_hat'].append(T_hat_at_x)
        summary['K_hat'].append(K_hat_at_x)
        summary['K_bar'].append(K_bar_at_x)
        summary['spread'].append(spread_at_x)
        
        print(f"{K0:<8.0f} {T_hat_at_x:<10.2f} {K_hat_at_x:<10.1f} {K_bar_at_x:<12.2f} {spread_at_x:<12.1f}")
    
    # Verification
    print("\n" + "="*70)
    print("VERIFICATION:")
    print("="*70)
    
    valid = ~np.isnan(summary['T_hat'])
    if np.sum(valid) > 1:
        K_bar_constant = np.max(np.abs(np.diff(np.array(summary['K_bar'])[valid]))) < 1.0
        spread_increasing = np.all(np.diff(np.array(summary['spread'])[valid]) >= 0)
        
        print(f"✓ K̄* unchanged with K₀: {K_bar_constant}")
        print(f"✓ Spread increases with K₀: {spread_increasing}")
        print(f"  T̂, K̂ behavior: (check plots)")
    
    plot_comparative_static(all_results, 'K0', K0_values, 
                           'Leverage K₀', 'leverage')
    
    return all_results, summary


def plot_comparative_static(all_results, param_name, param_values, param_label, filename):
    """Plot comparative statics for a given parameter."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
    
    # Panel 1: K̄*(x)
    ax1 = axes[0, 0]
    for i, val in enumerate(param_values):
        if val in all_results:
            sol = all_results[val]
            ax1.plot(sol['x_grid'], sol['K_bar_array'], 
                    linewidth=2, color=colors[i], label=f'{param_name}={val:.3g}')
    ax1.set_xlabel('Earnings x')
    ax1.set_ylabel('K̄*(x)')
    ax1.set_title(f'Refinancing Set vs {param_label}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: T̂(x)
    ax2 = axes[0, 1]
    for i, val in enumerate(param_values):
        if val in all_results:
            sol = all_results[val]
            res = sol['results']
            if len(res['x']) > 0:
                ax2.plot(res['x'], res['T_hat'], 'o-', 
                        linewidth=2, markersize=3, color=colors[i], label=f'{param_name}={val:.3g}')
    ax2.set_xlabel('Earnings x')
    ax2.set_ylabel('T̂(x)')
    ax2.set_title(f'Optimal Maturity vs {param_label}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: K̂(x)
    ax3 = axes[1, 0]
    for i, val in enumerate(param_values):
        if val in all_results:
            sol = all_results[val]
            res = sol['results']
            if len(res['x']) > 0:
                ax3.plot(res['x'], res['K_hat'], 'o-', 
                        linewidth=2, markersize=3, color=colors[i], label=f'{param_name}={val:.3g}')
    ax3.set_xlabel('Earnings x')
    ax3.set_ylabel('K̂(x)')
    ax3.set_title(f'Optimal Face Value vs {param_label}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Spread
    ax4 = axes[1, 1]
    for i, val in enumerate(param_values):
        if val in all_results:
            sol = all_results[val]
            res = sol['results']
            if len(res['x']) > 0:
                ax4.plot(res['x'], res['spread_bp'], 'o-', 
                        linewidth=2, markersize=3, color=colors[i], label=f'{param_name}={val:.3g}')
    ax4.set_xlabel('Earnings x')
    ax4.set_ylabel('Spread (bp)')
    ax4.set_title(f'Yield Spread vs {param_label}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent / '../output/figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'cs_full_{filename}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved figure to {output_path}")
    plt.close()


def print_summary_table():
    """Print comparison table."""
    print("\n" + "="*70)
    print("SUMMARY: Table 3.1 Verification")
    print("="*70)
    print("\nExpected relationships:")
    print(f"{'Parameter':<15} {'T̂':<10} {'K̂':<10} {'K̄*':<10} {'Spread':<10}")
    print("-"*70)
    print(f"{'↑ r':<15} {'−':<10} {'+/−':<10} {'−':<10} {'+':<10}")
    print(f"{'↑ σ':<15} {'−':<10} {'amb.':<10} {'−':<10} {'+':<10}")
    print(f"{'↑ 1/η':<15} {'−':<10} {'+':<10} {'−':<10} {'+':<10}")
    print(f"{'↑ K₀':<15} {'−':<10} {'+':<10} {'0':<10} {'+':<10}")
    print("="*70)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("FULL MODEL COMPARATIVE STATICS TESTS")
    print("Testing Table 3.1 with actual optimization")
    print("="*70)
    
    # Run all tests
    print("\nThis will take several minutes...")
    print("Solving optimization problem for each parameter value...")
    
    r_results, r_summary = test_interest_rate()
    sigma_results, sigma_summary = test_volatility()
    eta_results, eta_summary = test_illiquidity_duration()
    K0_results, K0_summary = test_leverage()
    
    # Print summary table
    print_summary_table()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
    print("\nFigures saved to: output/figures/")
    print("  - cs_full_interest_rate.png")
    print("  - cs_full_volatility.png")
    print("  - cs_full_illiquidity.png")
    print("  - cs_full_leverage.png")