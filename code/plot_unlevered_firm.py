"""
Plot unlevered firm value F(x).
Shows how firm value depends on current earnings x.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import Params
from cir_process import CIRProcess
from bond_valuation import BondValuation


def plot_unlevered_firm_value():
    """
    Plot F(x) = x/(r+κ) + κμ/(r(r+κ))
    Unlevered firm value as function of earnings.
    """
    print("\n" + "="*70)
    print("UNLEVERED FIRM VALUE: F(x)")
    print("="*70)
    
    # Setup
    p = Params()
    cir = CIRProcess(p.kappa, p.mu, p.sigma)
    bond_val = BondValuation(cir, p.r, p.eta, p.C)
    
    # Earnings grid
    x_values = np.linspace(0.5, 30, 300)
    
    # Calculate F(x) for each x
    F_values = [bond_val.F_unlevered(x) for x in x_values]
    
    # Calculate components separately for visualization
    x_component = x_values / (p.r + p.kappa)  # Present value of current earnings
    constant_component = p.kappa * p.mu / (p.r * (p.r + p.kappa))  # PV of long-run mean
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ========== Panel 1: F(x) ==========
    ax1 = axes[0]
    ax1.plot(x_values, F_values, linewidth=3, color='darkblue', label='F(x) = Unlevered Firm Value')
    
    # Add reference lines
    ax1.axhline(constant_component, color='gray', linestyle='--', alpha=0.5,
                label=f'Constant term = {constant_component:.2f}')
    ax1.axvline(p.mu, color='green', linestyle='--', alpha=0.5,
                label=f'Long-run mean μ = {p.mu:.2f}')
    
    # Mark current base case earnings
    x_base = 10.0
    F_base = bond_val.F_unlevered(x_base)
    ax1.plot(x_base, F_base, 'ro', markersize=10, label=f'Base: x={x_base}, F={F_base:.2f}')
    
    ax1.set_xlabel('Current Earnings x', fontsize=12)
    ax1.set_ylabel('Unlevered Firm Value F(x)', fontsize=12)
    ax1.set_title('Unlevered Firm Value', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([x_values[0], x_values[-1]])
    
    # Add formula as text
    formula_text = r'$F(x) = \frac{x}{r+\kappa} + \frac{\kappa\mu}{r(r+\kappa)}$'
    ax1.text(0.05, 0.95, formula_text, transform=ax1.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========== Panel 2: Decomposition ==========
    ax2 = axes[1]
    ax2.plot(x_values, F_values, linewidth=3, color='darkblue', label='Total F(x)')
    ax2.plot(x_values, x_component, linewidth=2, color='red', linestyle='--',
             label='x/(r+κ) - Current earnings PV')
    ax2.axhline(constant_component, linewidth=2, color='green', linestyle='--',
                label=f'κμ/(r(r+κ)) = {constant_component:.2f}')
    
    ax2.set_xlabel('Current Earnings x', fontsize=12)
    ax2.set_ylabel('Value Components', fontsize=12)
    ax2.set_title('Firm Value Decomposition', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([x_values[0], x_values[-1]])
    
    # Add annotation
    annotation_text = f"""Parameters:
r = {p.r:.3f}
κ = {p.kappa:.5f}
μ = {p.mu:.2f}

Constant = {constant_component:.2f}
(independent of x)"""
    
    ax2.text(0.6, 0.3, annotation_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent / '../output/figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'unlevered_firm_value.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved figure to {output_path}")
    plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nFormula: F(x) = x/(r+κ) + κμ/(r(r+κ))")
    print(f"\nComponents:")
    print(f"  x/(r+κ)         = Present value of current earnings")
    print(f"  κμ/(r(r+κ))     = {constant_component:.2f} (constant)")
    print(f"\nAt x = {x_base}:")
    print(f"  F({x_base}) = {F_base:.2f}")
    print(f"\nAt long-run mean x = μ = {p.mu:.2f}:")
    F_mu = bond_val.F_unlevered(p.mu)
    print(f"  F({p.mu:.2f}) = {F_mu:.2f}")
    print(f"\nInterpretation:")
    print(f"  - Firm value increases linearly with current earnings")
    print(f"  - Slope = 1/(r+κ) = {1/(p.r + p.kappa):.2f}")
    print(f"  - Higher earnings → higher firm value")
    print(f"  - Even with x=0, firm has value = {constant_component:.2f} (option on future earnings)")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("UNLEVERED FIRM VALUATION ANALYSIS")
    print("="*70)
    
    plot_unlevered_firm_value()
    
    print("\n" + "="*70)
    print("COMPLETE ✓")
    print("="*70)
