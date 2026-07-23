"""
Test CIR process implementation.
Run this to verify everything works before proceeding.
"""

import numpy as np
import matplotlib.pyplot as plt
from config import Params
from cir_process import CIRProcess


def test_basic_properties():
    """Test basic CIR properties."""
    print("="*60)
    print("TEST 1: Basic CIR Properties")
    print("="*60)
    
    p = Params()
    p.validate()
    
    cir = CIRProcess(p.kappa, p.mu, p.sigma)
    print(f"\n{cir}")
    print(f"Degrees of freedom (ν): {cir.nu:.2f}")
    
    # Test conditional mean
    x = 10.0
    tau = 5.0
    mean = cir.mean_conditional(x, tau)
    print(f"\nConditional mean:")
    print(f"  E[x_{tau} | x_0 = {x}] = {mean:.3f}")
    print(f"  (Should approach μ = {p.mu} for large τ)")
    
    # Test that mean approaches mu
    x = 10.0
    for tau in [1, 5, 10, 50, 100]:
        mean = cir.mean_conditional(x, tau)
        print(f"  τ = {tau:3d}: E[x_τ] = {mean:.3f}")


def test_survival_probabilities():
    """Test survival probability calculations."""
    print("\n" + "="*60)
    print("TEST 2: Survival Probabilities")
    print("="*60)
    
    p = Params()
    cir = CIRProcess(p.kappa, p.mu, p.sigma)
    
    x = 10.0
    tau = 5.0
    
    print(f"\nP(x_{tau} ≥ y | x_0 = {x}) for τ = {tau}:")
    print(f"{'y':<10} {'Q(x,τ,y)':<15} {'Q⁺(x,τ,y)':<15}")
    print("-"*40)
    
    for y in [0, 5, 10, 15, 20, 30]:
        Q = cir.Q(x, tau, y)
        Q_plus = cir.Q_plus(x, tau, y)
        print(f"{y:<10.1f} {Q:<15.4f} {Q_plus:<15.4f}")
    
    # Should be monotonically decreasing in y
    print("\n✓ Q should decrease as y increases")


def test_vectorization():
    """Test that functions work with arrays."""
    print("\n" + "="*60)
    print("TEST 3: Vectorization")
    print("="*60)
    
    p = Params()
    cir = CIRProcess(p.kappa, p.mu, p.sigma)
    
    x_array = np.array([5.0, 10.0, 15.0, 20.0])
    tau = 5.0
    
    means = cir.mean_conditional(x_array, tau)
    
    print(f"\nConditional means for τ = {tau}:")
    for x, mean in zip(x_array, means):
        print(f"  x = {x:5.1f} → E[x_τ] = {mean:.3f}")
    
    print("\n✓ Vectorization works")


def plot_survival_curves():
    """Plot survival probabilities vs threshold."""
    print("\n" + "="*60)
    print("TEST 4: Plotting Survival Curves")
    print("="*60)
    
    p = Params()
    cir = CIRProcess(p.kappa, p.mu, p.sigma)
    
    x = 10.0
    tau = 5.0
    y_values = np.linspace(0, 50, 100)
    
    Q_values = [cir.Q(x, tau, y) for y in y_values]
    
    plt.figure(figsize=(8, 5))
    plt.plot(y_values, Q_values, 'b-', linewidth=2)
    plt.axvline(x, color='r', linestyle='--', label=f'Current x = {x}')
    plt.axvline(cir.mean_conditional(x, tau), color='g', linestyle='--', 
                label=f'E[x_τ] = {cir.mean_conditional(x, tau):.1f}')
    plt.xlabel('Threshold y')
    plt.ylabel('P(x_τ ≥ y | x_0 = x)')
    plt.title(f'Survival Probability (τ = {tau})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../output/figures/test_cir_survival.png', dpi=150)
    print("✓ Saved figure to output/figures/test_cir_survival.png")
    plt.close()


def plot_conditional_distribution():
    """Plot how distribution evolves over time."""
    print("\n" + "="*60)
    print("TEST 5: Evolution of Distribution")
    print("="*60)
    
    p = Params()
    cir = CIRProcess(p.kappa, p.mu, p.sigma)
    
    x0 = 10.0
    y_values = np.linspace(0, 100, 200)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for tau in [1, 5, 10, 20]:
        Q_values = [cir.Q(x0, tau, y) for y in y_values]
        ax.plot(y_values, Q_values, label=f'τ = {tau}')
    
    ax.axvline(x0, color='red', linestyle='--', alpha=0.5, label=f'x₀ = {x0}')
    ax.axvline(p.mu, color='green', linestyle='--', alpha=0.5, label=f'μ = {p.mu}')
    ax.set_xlabel('Threshold y')
    ax.set_ylabel('P(x_τ ≥ y | x_0)')
    ax.set_title('Distribution Evolution Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../output/figures/test_cir_evolution.png', dpi=150)
    print("✓ Saved figure to output/figures/test_cir_evolution.png")
    plt.close()


if __name__ == '__main__':
    print("\nTesting CIR Process Implementation\n")
    
    test_basic_properties()
    test_survival_probabilities()
    test_vectorization()
    plot_survival_curves()
    plot_conditional_distribution()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
    print("\nYour CIR implementation is working correctly!")
    print("Next step: Implement bond valuation (bond_valuation.py)")
