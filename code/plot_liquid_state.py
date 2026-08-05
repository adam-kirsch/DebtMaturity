"""
Visualize bond and firm valuation in LIQUID markets.
Plots B^L(x) and V^L(x) against earnings x.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import Params
from cir_process import CIRProcess
from bond_valuation import BondValuation
from firm_valuation import FirmValuation


def plot_liquid_values():
    """
    Plot B^L(x) and V^L(x) against earnings x.
    """
    print("\n" + "="*70)
    print("LIQUID STATE VALUATION: B^L(x) and V^L(x)")
    print("="*70)
    
    # Setup
    p = Params()
    cir = CIRProcess(p.kappa, p.mu, p.sigma)
    bond_val = BondValuation(cir, p.r, p.eta, p.C)
    firm_val = FirmValuation(bond_val)
    
    # Earnings grid
    x_values = np.linspace(1, 25, 200)
    
    # Fixed parameters for bonds
    K_fixed = 200
    T_fixed = 5.0
    
    # Calculate B^L and V^L
    print(f"\nCalculating B^L(x) and V^L(x) for K={K_fixed}, T={T_fixed}...")
    B_L = [bond_val.B_liquid(x, T_fixed, K_fixed) for x in x_values]
    V_L = [firm_val.V_liquid(x, T_fixed, K_fixed) for x in x_values]
    F_unlev = [bond_val.F_unlevered(x) for x in x_values]
    
    # Create 2 panel figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Liquid Market Valuation (K={K_fixed}, T={T_fixed} years)', 
                 fontsize=14, fontweight='bold')
    
    # ========== Panel 1: B^L(x) - Bond Value ==========
    ax1 = axes[0]
    ax1.plot(x_values, B_L, linewidth=3, color='blue', label='B^L(x) - Liquid Bond Value')
    ax1.axhline(K_fixed, color='red', linestyle='--', linewidth=2, 
                label=f'Face Value K={K_fixed}')
    ax1.axhline(p.K_0, color='orange', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Refinancing Need K₀={p.K_0}')
    
    ax1.set_xlabel('Earnings x', fontsize=12)
    ax1.set_ylabel('Bond Value B^L(x)', fontsize=12)
    ax1.set_title('Liquid Bond Value', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([x_values[0], x_values[-1]])
    
    # ========== Panel 2: V^L(x) - Firm Value (Equity) ==========
    ax2 = axes[1]
    ax2.plot(x_values, V_L, linewidth=3, color='green', label='V^L(x) - Equity Value (Levered)')
    ax2.plot(x_values, F_unlev, linewidth=3, color='black', linestyle='--', 
             label='F(x) - Unlevered Firm Value')
    
    ax2.set_xlabel('Earnings x', fontsize=12)
    ax2.set_ylabel('Firm Value V^L(x)', fontsize=12)
    ax2.set_title('Liquid Firm Value (Equity)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([x_values[0], x_values[-1]])
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent / 'output/figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'liquid_values_vs_earnings.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved figure to {output_path}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"\nBond Value B^L:")
    print(f"  At x={x_values[0]:.1f}:  B^L = {B_L[0]:.2f}  (discount = {(1-B_L[0]/K_fixed)*100:.1f}%)")
    print(f"  At x={x_values[-1]:.1f}: B^L = {B_L[-1]:.2f}  (discount = {(1-B_L[-1]/K_fixed)*100:.1f}%)")
    
    print(f"\nFirm Value V^L:")
    print(f"  At x={x_values[0]:.1f}:  V^L = {V_L[0]:.2f},  F(x) = {F_unlev[0]:.2f}")
    print(f"  At x={x_values[-1]:.1f}: V^L = {V_L[-1]:.2f},  F(x) = {F_unlev[-1]:.2f}")
    
    print(f"\nDefault threshold x* = {bond_val.x_star(K_fixed):.2f}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("LIQUID STATE VALUATION ANALYSIS")
    print("="*70)
    
    plot_liquid_values()
    
    print("\n" + "="*70)
    print("COMPLETE ✓")
    print("="*70)
