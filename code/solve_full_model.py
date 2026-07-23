"""
Solve the complete model - main script with plotting.
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


def plot_results(x_grid, K_bar_array, results, params, output_dir):
    """Plot all results."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # results already contains only feasible points
    x_feas = results['x']
    K_hat = results['K_hat']
    T_hat = results['T_hat']
    B_hat = results['B_hat']
    spread_bp = results['spread_bp']
    
    # Panel 1: K̄*(x) - Refinancing Set
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(x_grid, K_bar_array, 'b-', linewidth=2, label='K̄*(x)')
    ax1.axhline(params.K_0, color='r', linestyle='--', label=f'K₀={params.K_0}')
    ax1.axhline(params.K_0 + params.C, color='orange', linestyle='--', 
                label=f'K₀+C={params.K_0 + params.C}')
    ax1.set_xlabel('Earnings x')
    ax1.set_ylabel('K̄*(x)')
    ax1.set_title('Refinancing Set')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Optimal Maturity T̂(x)
    ax2 = plt.subplot(3, 3, 2)
    if len(x_feas) > 0:
        ax2.plot(x_feas, T_hat, 'o-', linewidth=2, markersize=4)
        ax2.set_xlabel('Earnings x')
        ax2.set_ylabel('T̂(x)')
        ax2.set_title('Optimal Maturity')
        ax2.grid(True, alpha=0.3)
    
    # Panel 3: Optimal Face Value K̂(x)
    ax3 = plt.subplot(3, 3, 3)
    if len(x_feas) > 0:
        ax3.plot(x_feas, K_hat, 'o-', linewidth=2, markersize=4, color='green')
        ax3.axhline(params.K_0, color='r', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Earnings x')
        ax3.set_ylabel('K̂(x)')
        ax3.set_title('Optimal Face Value')
        ax3.grid(True, alpha=0.3)
    
    # Panel 4: Yield Spread
    ax4 = plt.subplot(3, 3, 4)
    if len(x_feas) > 0:
        ax4.plot(x_feas, spread_bp, 'o-', 
                linewidth=2, markersize=4, color='purple')
        ax4.set_xlabel('Earnings x')
        ax4.set_ylabel('Spread (bp)')
        ax4.set_title('Yield Spread')
        ax4.grid(True, alpha=0.3)
    
    # Panel 5: Bond Value B̂(x)
    ax5 = plt.subplot(3, 3, 5)
    if len(x_feas) > 0:
        ax5.plot(x_feas, B_hat, 'o-', 
                linewidth=2, markersize=4, color='brown')
        ax5.axhline(params.K_0, color='r', linestyle='--', label='K₀', alpha=0.5)
        ax5.set_xlabel('Earnings x')
        ax5.set_ylabel('B̂(x)')
        ax5.set_title('Optimal Bond Value')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Panel 6: Maturity vs Face Value
    ax6 = plt.subplot(3, 3, 6)
    if len(x_feas) > 0:
        scatter = ax6.scatter(T_hat, K_hat, c=x_feas, cmap='viridis', s=50)
        ax6.set_xlabel('T̂')
        ax6.set_ylabel('K̂')
        ax6.set_title('Maturity vs Face Value')
        plt.colorbar(scatter, ax=ax6, label='x')
        ax6.grid(True, alpha=0.3)
    
    # Panel 7: K̄*/F(x) ratio
    ax7 = plt.subplot(3, 3, 7)
    F_x_grid = np.array([params.mu * (1/(params.r + params.kappa)) + 
                         params.kappa * params.mu / (params.r * (params.r + params.kappa)) 
                         for _ in x_grid])
    ratio = K_bar_array / F_x_grid
    ax7.plot(x_grid, ratio, 'b-', linewidth=2)
    ax7.set_xlabel('Earnings x')
    ax7.set_ylabel('K̄*(x) / F(x)')
    ax7.set_title('Refinancing Capacity Ratio')
    ax7.grid(True, alpha=0.3)
    
    # Panel 8: Default probability (approximate)
    ax8 = plt.subplot(3, 3, 8)
    if len(x_feas) > 0:
        default_prob = 1 - (B_hat / K_hat)
        ax8.plot(x_feas, default_prob * 100, 'o-', 
                linewidth=2, markersize=4, color='red')
        ax8.set_xlabel('Earnings x')
        ax8.set_ylabel('Default Prob (%)')
        ax8.set_title('Approximate Default Probability')
        ax8.grid(True, alpha=0.3)
    
    # Panel 9: Summary text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    n_feas = len(x_feas)
    if n_feas > 0:
        summary_text = f"""Base Case Results
        
Parameters:
  K₀ = {params.K_0}
  r = {params.r}
  η = {params.eta}
  C = {params.C}
  σ = {params.sigma:.3f}
  
Solutions Found:
  Feasible: {n_feas}/{len(x_grid)}
  
Optimal Maturity:
  Min: {np.min(T_hat):.2f}
  Max: {np.max(T_hat):.2f}
  
Optimal Face Value:
  Min: {np.min(K_hat):.1f}
  Max: {np.max(K_hat):.1f}
  
Spread:
  Min: {np.min(spread_bp):.1f} bp
  Max: {np.max(spread_bp):.1f} bp
"""
    else:
        summary_text = "No feasible solutions found"
    
    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    
    output_path = output_dir / 'base_case_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved figure to {output_path}")
    plt.close()


def solve_base_case():
    """Solve base case from paper."""
    
    print("\n" + "="*70)
    print("SOLVING FULL MODEL - BASE CASE")
    print("="*70)
    
    # Parameters
    p = Params()
    p.validate()
    
    # Use smaller grids for initial test
    x_grid = np.linspace(p.x_min, p.x_max, 30)  # Start small
    K_grid = np.linspace(p.K_min, p.K_max, 40)
    T_grid = np.linspace(p.T_min, p.T_max, 30)
    
    print(f"\nGrid sizes: x={len(x_grid)}, K={len(K_grid)}, T={len(T_grid)}")
    print(f"Total computations: ~{len(x_grid) * len(K_grid) * len(T_grid):,}")
    
    # Initialize CIR process
    cir = CIRProcess(p.kappa, p.mu, p.sigma)
    print(f"\n{cir}")
    
    # Initialize bond valuation
    bond_val = BondValuation(cir, p.r, p.eta, p.C)
    
    # STEP 1: Solve for K̄*(x)
    print("\n" + "-"*70)
    print("STEP 1: Solving for Refinancing Set K̄*(x)")
    print("-"*70)
    
    ref_solver = RefinancingSetSolver(bond_val, x_grid, K_grid, T_grid)
    K_bar_array, K_bar_func = ref_solver.solve(tol=1e-3, max_iter=50, verbose=True)
    
    # STEP 2: Find optimal bonds
    print("\n" + "-"*70)
    print("STEP 2: Finding Optimal Bonds (K̂, T̂)")
    print("-"*70)
    
    opt_solver = OptimalBondSolver(bond_val, K_bar_func)
    results = opt_solver.solve_for_grid(x_grid, p.K_0, K_grid, T_grid, verbose=True)
    
    # STEP 3: Save results
    print("\n" + "-"*70)
    print("STEP 3: Saving Results")
    print("-"*70)
    
    output_dir = Path('data/results/base_case')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        'params': p,
        'x_grid': x_grid,
        'K_grid': K_grid,
        'T_grid': T_grid,
        'K_bar_array': K_bar_array,
        'results': results
    }
    
    with open(output_dir / 'solution.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"✓ Saved to {output_dir / 'solution.pkl'}")
    
    # STEP 4: Plot results
    print("\n" + "-"*70)
    print("STEP 4: Plotting Results")
    print("-"*70)
    
    fig_dir = Path('output/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plot_results(x_grid, K_bar_array, results, p, fig_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("SOLUTION COMPLETE")
    print("="*70)
    
    n_feas = len(results['x'])  # Changed from np.sum(results['feasible'])
    n_total = len(x_grid)
    print(f"\nFeasible solutions: {n_feas}/{n_total}")
    
    if n_feas > 0:
        print(f"\nOptimal Maturity T̂:")
        print(f"  Range: [{np.min(results['T_hat']):.2f}, {np.max(results['T_hat']):.2f}]")
        print(f"  Mean: {np.mean(results['T_hat']):.2f}")
        
        print(f"\nOptimal Face Value K̂:")
        print(f"  Range: [{np.min(results['K_hat']):.1f}, {np.max(results['K_hat']):.1f}]")
        print(f"  Mean: {np.mean(results['K_hat']):.1f}")
        
        print(f"\nYield Spread:")
        print(f"  Range: [{np.min(results['spread_bp']):.1f}, {np.max(results['spread_bp']):.1f}] bp")
        print(f"  Mean: {np.mean(results['spread_bp']):.1f} bp")
    
    return save_data


if __name__ == '__main__':
    results = solve_base_case()
    
    print("\n" + "="*70)
    print("Done! Check output/figures/ for plots")
    print("="*70)
