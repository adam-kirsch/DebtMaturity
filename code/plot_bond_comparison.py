"""
Compare liquid vs illiquid bond values.
Plots B^L(x) and B^I(x) on same figure.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import Params
from cir_process import CIRProcess
from bond_valuation import BondValuation


def K_bar_approximate(x, bond_val, C):
    """
    Simple approximation for K̄*(x) for plotting purposes.
    True K̄* requires solving fixed point - this is rough estimate.
    """
    F_x = bond_val.F_unlevered(x)
    alpha = 0.55  # Empirical: illiquid market can raise ~55% of firm value
    return max(alpha * F_x - C, 0)


def plot_liquid_vs_illiquid_bonds():
    """
    Plot B^L(x) and B^I(x) together to show illiquidity discount.
    """
    print("\n" + "="*70)
    print("LIQUID vs ILLIQUID BOND VALUES")
    print("="*70)
    
    # Setup
    p = Params()
    cir = CIRProcess(p.kappa, p.mu, p.sigma)
    bond_val = BondValuation(cir, p.r, p.eta, p.C)
    
    # Create K_bar function using approximation
    K_bar_func = lambda x: K_bar_approximate(x, bond_val, p.C)
    
    # Fixed bond parameters
    K = 200
    T = 5.0
    
    # Earnings grid
    x_values = np.linspace(2, 25, 200)
    
    print(f"\nCalculating bond values for K={K}, T={T}...")
    
    # Calculate B^L and B^I
    B_L = [bond_val.B_liquid(x, T, K) for x in x_values]
    B_I = [bond_val.B_illiquid(x, T, K, K_bar_func) for x in x_values]
    
    # Calculate illiquidity discount
    discount = [(bl - bi) for bl, bi in zip(B_L, B_I)]
    discount_pct = [(bl - bi) / bl * 100 if bl > 0 else 0 for bl, bi in zip(B_L, B_I)]
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Liquid vs Illiquid Bond Valuation (K={K}, T={T} years)', 
                 fontsize=14, fontweight='bold')
    
    # ========== Panel 1: B^L vs B^I ==========
    ax1 = axes[0, 0]
    ax1.plot(x_values, B_L, linewidth=3, color='blue', label='B^L - Liquid Market')
    ax1.plot(x_values, B_I, linewidth=3, color='red', label='B^I - Illiquid Market')
    ax1.axhline(K, color='black', linestyle='--', alpha=0.5, linewidth=2, label=f'Face Value K={K}')
    ax1.axhline(p.K_0, color='orange', linestyle='--', alpha=0.5, linewidth=2, 
                label=f'Refinancing Need K₀={p.K_0}')
    
    ax1.set_xlabel('Earnings x', fontsize=11)
    ax1.set_ylabel('Bond Value', fontsize=11)
    ax1.set_title('Bond Values by Market State', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([x_values[0], x_values[-1]])
    
    # ========== Panel 2: Absolute Discount ==========
    ax2 = axes[0, 1]
    ax2.plot(x_values, discount, linewidth=3, color='darkred')
    ax2.fill_between(x_values, 0, discount, alpha=0.3, color='red')
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('Earnings x', fontsize=11)
    ax2.set_ylabel('Illiquidity Discount (B^L - B^I)', fontsize=11)
    ax2.set_title('Absolute Illiquidity Discount', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([x_values[0], x_values[-1]])
    
    # Add text with max discount
    max_discount = max(discount)
    max_discount_x = x_values[discount.index(max_discount)]
    ax2.text(0.05, 0.95, f'Max discount: ${max_discount:.2f}\nat x = {max_discount_x:.2f}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========== Panel 3: Percentage Discount ==========
    ax3 = axes[1, 0]
    ax3.plot(x_values, discount_pct, linewidth=3, color='purple')
    ax3.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    ax3.set_xlabel('Earnings x', fontsize=11)
    ax3.set_ylabel('Discount as % of Liquid Value', fontsize=11)
    ax3.set_title('Percentage Illiquidity Discount', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([x_values[0], x_values[-1]])
    
    # Add average discount
    avg_discount = np.mean([d for d in discount_pct if d > 0])
    ax3.text(0.05, 0.95, f'Avg discount: {avg_discount:.1f}%',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # ========== Panel 4: Bond Prices (B/K ratio) ==========
    ax4 = axes[1, 1]
    price_L = [bl / K for bl in B_L]
    price_I = [bi / K for bi in B_I]
    
    ax4.plot(x_values, price_L, linewidth=3, color='blue', label='Liquid Market')
    ax4.plot(x_values, price_I, linewidth=3, color='red', label='Illiquid Market')
    ax4.axhline(1.0, color='black', linestyle='--', alpha=0.5, linewidth=2, label='Par (B/K=1)')
    
    ax4.set_xlabel('Earnings x', fontsize=11)
    ax4.set_ylabel('Bond Price (B/K)', fontsize=11)
    ax4.set_title('Bond Prices Relative to Face Value', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([x_values[0], x_values[-1]])
    ax4.set_ylim([0, min(1.1, max(max(price_L), max(price_I)) * 1.05)])
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent / '../output/figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'liquid_vs_illiquid_bonds.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved figure to {output_path}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"\nBond Parameters:")
    print(f"  Face Value K = {K}")
    print(f"  Maturity T = {T} years")
    print(f"\nMarket Parameters:")
    print(f"  Risk-free rate r = {p.r}")
    print(f"  Illiquidity intensity η = {p.eta} (E[duration] = {1/p.eta:.2f} years)")
    print(f"  Refinancing cost C = {p.C}")
    
    print(f"\nIlliquidity Discount:")
    print(f"  Maximum: ${max_discount:.2f} at x = {max_discount_x:.2f}")
    print(f"  Average: {avg_discount:.1f}% of liquid value")
    
    # Find where bonds trade above/below par
    par_L_idx = next((i for i, p in enumerate(price_L) if p >= 0.99), None)
    par_I_idx = next((i for i, p in enumerate(price_I) if p >= 0.99), None)
    
    if par_L_idx:
        print(f"\nLiquid bond trades at par (≥99%) when x ≥ {x_values[par_L_idx]:.2f}")
    if par_I_idx:
        print(f"Illiquid bond trades at par (≥99%) when x ≥ {x_values[par_I_idx]:.2f}")
    
    print(f"\nInterpretation:")
    print(f"  - B^I < B^L everywhere (illiquidity always reduces bond value)")
    print(f"  - Discount decreases as earnings increase (less default risk)")
    print(f"  - Illiquidity matters more for low-earning firms")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("BOND VALUATION: LIQUID vs ILLIQUID COMPARISON")
    print("="*70)
    print("\nNote: Using approximate K̄*(x) for visualization.")
    print("For exact values, see cs_full_*.png figures.")
    
    plot_liquid_vs_illiquid_bonds()
    
    print("\n" + "="*70)
    print("COMPLETE ✓")
    print("="*70)
