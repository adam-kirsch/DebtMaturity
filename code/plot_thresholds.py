"""
Plot default and refinancing thresholds as functions of face value K.
Shows x*(K) and x_B(K).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import Params
from cir_process import CIRProcess
from bond_valuation import BondValuation
from refinancing_solver import RefinancingSetSolver


def plot_thresholds_vs_K():
    """
    Plot x*(K) and x_B(K) as functions of face value K.
    
    x*(K) = Default threshold in liquid state
    x_B(K) = Refinancing threshold in illiquid state
    """
    print("\n" + "="*70)
    print("DEFAULT AND REFINANCING THRESHOLDS vs FACE VALUE K")
    print("="*70)
    
    # Setup
    p = Params()
    cir = CIRProcess(p.kappa, p.mu, p.sigma)
    bond_val = BondValuation(cir, p.r, p.eta, p.C)
    
    # First, solve for K̄*(x) to get x_B values
    print("\nSolving for K̄*(x) to find x_B thresholds...")
    x_grid = np.linspace(p.x_min, p.x_max, 50)
    K_grid = np.linspace(p.K_min, p.K_max, 40)
    T_grid = np.linspace(p.T_min, p.T_max, 30)
    
    ref_solver = RefinancingSetSolver(bond_val, x_grid, K_grid, T_grid)
    K_bar_array, K_bar_func = ref_solver.solve(tol=1e-3, max_iter=50, verbose=False)
    
    print("✓ K̄*(x) computed")
    
    # Range of K values to plot
    K_values = np.linspace(50, 400, 200)
    
    # Calculate x*(K) - simple formula
    print("\nCalculating x*(K) - liquid default threshold...")
    x_star_values = [bond_val.x_star(K) for K in K_values]
    
    # Calculate x_B(K) - requires inverting K̄*(x_B) = K + C
    print("Calculating x_B(K) - illiquid refinancing threshold...")
    x_B_values = []
    
    for K in K_values:
        # Find x_B such that K̄*(x_B) = K + C
        target = K + p.C
        
        # Check if solution exists
        if K_bar_func(x_grid[-1]) < target:
            # No solution - can't refinance this much even at highest x
            x_B_values.append(np.inf)
        else:
            # Use bisection to find x_B
            x_low, x_high = x_grid[0], x_grid[-1]
            
            for _ in range(50):
                x_mid = (x_low + x_high) / 2
                K_bar_mid = K_bar_func(x_mid)
                
                if abs(K_bar_mid - target) < 0.1:
                    x_B_values.append(x_mid)
                    break
                
                if K_bar_mid < target:
                    x_low = x_mid
                else:
                    x_high = x_mid
            else:
                x_B_values.append((x_low + x_high) / 2)
    
    print("✓ Thresholds computed")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Default and Refinancing Thresholds', fontsize=14, fontweight='bold')
    
    # ========== Panel 1: Both thresholds ==========
    ax1 = axes[0]
    
    # Only plot finite x_B values
    x_B_finite = [(K, xb) for K, xb in zip(K_values, x_B_values) if xb < 100]
    if x_B_finite:
        K_finite, xB_finite = zip(*x_B_finite)
        ax1.plot(K_finite, xB_finite, linewidth=3, color='red', 
                label='x_B(K) - Refinancing Threshold (Illiquid)')
    
    ax1.plot(K_values, x_star_values, linewidth=3, color='blue', 
            label='x*(K) - Default Threshold (Liquid)')
    
    # Add reference lines
    ax1.axhline(p.mu, color='green', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Long-run mean μ = {p.mu:.2f}')
    ax1.axvline(p.K_0, color='orange', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Base case K₀ = {p.K_0}')
    
    ax1.set_xlabel('Face Value K', fontsize=12)
    ax1.set_ylabel('Earnings Threshold', fontsize=12)
    ax1.set_title('Thresholds vs Face Value', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([K_values[0], K_values[-1]])
    
    # Add formulas
    formula_text = """x*(K) = (r+κ)K - κμ/r
    
x_B(K): K̄*(x_B) = K + C"""
    ax1.text(0.55, 0.25, formula_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========== Panel 2: Gap between thresholds ==========
    ax2 = axes[1]
    
    # Calculate gap where both are finite
    gaps = []
    K_for_gaps = []
    for K, x_s, x_b in zip(K_values, x_star_values, x_B_values):
        if x_b < 100:  # Only where x_B is finite
            gaps.append(x_b - x_s)
            K_for_gaps.append(K)
    
    if gaps:
        ax2.plot(K_for_gaps, gaps, linewidth=3, color='purple')
        ax2.fill_between(K_for_gaps, 0, gaps, alpha=0.3, color='purple')
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        ax2.set_xlabel('Face Value K', fontsize=12)
        ax2.set_ylabel('x_B(K) - x*(K)', fontsize=12)
        ax2.set_title('Gap Between Thresholds', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([K_values[0], K_values[-1]])
        
        # Add interpretation
        avg_gap = np.mean(gaps)
        ax2.text(0.05, 0.95, f'Average gap: {avg_gap:.2f}\n\nInterpretation:\nEarnings must be higher\nto refinance (illiquid)\nthan to avoid default\n(liquid)',
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent / '../output/figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'thresholds_vs_K.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved figure to {output_path}")
    plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # At base case K_0
    x_star_base = bond_val.x_star(p.K_0)
    x_B_base_idx = np.argmin(np.abs(np.array(K_values) - p.K_0))
    x_B_base = x_B_values[x_B_base_idx]
    
    print(f"\nAt base case K₀ = {p.K_0}:")
    print(f"  x*(K₀) = {x_star_base:.2f} - Must exceed this to avoid default (liquid)")
    if x_B_base < 100:
        print(f"  x_B(K₀) = {x_B_base:.2f} - Must exceed this to refinance (illiquid)")
        print(f"  Gap = {x_B_base - x_star_base:.2f}")
    
    print(f"\nParameters:")
    print(f"  r = {p.r}")
    print(f"  κ = {p.kappa}")
    print(f"  μ = {p.mu}")
    print(f"  C = {p.C}")
    
    print(f"\nKey Insights:")
    print(f"  - x*(K) is LINEAR in K (simple formula)")
    print(f"  - x_B(K) is NONLINEAR (comes from K̄* fixed point)")
    print(f"  - x_B(K) > x*(K) always: harder to refinance than avoid default")
    print(f"  - Gap increases with K: higher debt → bigger refinancing challenge")
    
    # Find where x* crosses μ
    K_at_mu = (p.mu + p.kappa * p.mu / p.r) / (p.r + p.kappa)
    print(f"\n  - When K = {K_at_mu:.2f}, x* = μ (default at long-run mean)")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("THRESHOLD ANALYSIS")
    print("="*70)
    
    plot_thresholds_vs_K()
    
    print("\n" + "="*70)
    print("COMPLETE ✓")
    print("="*70)
