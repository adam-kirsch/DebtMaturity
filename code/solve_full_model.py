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


def create_concentrated_grid(min_val, max_val, n_total, dense_end=2.0, dense_fraction=0.25):
    """
    Create non-uniform grid with more points concentrated at low values.
    
    Parameters
    ----------
    min_val : float
        Minimum value
    max_val : float
        Maximum value
    n_total : int
        Total number of points
    dense_end : float
        Value up to which we want dense spacing
    dense_fraction : float
        Fraction of points to use in dense region (default 0.25 = 25%)
    
    Returns
    -------
    array
        Non-uniform grid
        
    Example
    -------
    For x from 0.5 to 100 with 50 points and 25% dense:
    - 17 points in [0.1, 2.0] (dense region)
    - 33 points in [2.0, 100] (sparse region)
    """
    n_dense = int(n_total * dense_fraction)
    n_sparse = n_total - n_dense
    
    # Dense spacing from min_val to dense_end
    dense_grid = np.linspace(min_val, dense_end, n_dense)
    
    # Regular spacing from dense_end to max_val (exclude first point to avoid duplicate)
    sparse_grid = np.linspace(dense_end, max_val, n_sparse + 1)[1:]
    
    # Concatenate
    grid = np.concatenate([dense_grid, sparse_grid])
    
    return grid


def compute_feasible_maturity_bounds(x_grid, K_0, K_grid, T_grid, bond_val, K_bar_func, verbose=True):
    """
    Compute minimum and maximum feasible maturity for each x.
    
    For each x, find the range [T_min, T_max] such that there exists
    some K where B^I(x, T, K) >= K_0.
    
    Parameters
    ----------
    x_grid : array
        Earnings grid
    K_0 : float
        Required amount to raise
    K_grid : array
        Face value grid to search over
    T_grid : array
        Maturity grid to search over
    bond_val : BondValuation
        Bond valuation object
    K_bar_func : callable
        Refinancing capacity function
    verbose : bool
        Print progress
    
    Returns
    -------
    dict with arrays:
        x_bounds : array of x values where bounds exist
        T_min_bounds : array of minimum feasible maturities
        T_max_bounds : array of maximum feasible maturities
    """
    if verbose:
        print("\nComputing feasible maturity bounds...")
    
    x_bounds = []
    T_min_bounds = []
    T_max_bounds = []
    
    for i, x in enumerate(x_grid):
        if verbose and i % 10 == 0:
            print(f"  Progress: {i}/{len(x_grid)}")
        
        # For each T, check if any K makes bond feasible
        feasible_T = []
        
        for T in T_grid:
            # Check if any K in grid satisfies B^I(x, T, K) >= K_0
            for K in K_grid:
                B_I = bond_val.B_illiquid(x, T, K, K_bar_func)
                if B_I >= K_0:
                    feasible_T.append(T)
                    break  # Found feasible K for this T, move to next T
        
        # If we found feasible maturities, record bounds
        if len(feasible_T) > 0:
            x_bounds.append(x)
            T_min_bounds.append(min(feasible_T))
            T_max_bounds.append(max(feasible_T))
    
    if verbose:
        print(f"  Found feasible maturity bounds for {len(x_bounds)}/{len(x_grid)} earnings levels")
    
    return {
        'x_bounds': np.array(x_bounds),
        'T_min_bounds': np.array(T_min_bounds),
        'T_max_bounds': np.array(T_max_bounds)
    }


def compute_feasible_face_value_bounds(x_grid, K_0, K_grid, T_grid, bond_val, K_bar_func, verbose=True):
    """
    Compute minimum and maximum feasible face value for each x.
    
    For each x, find the range [K_min, K_max] such that there exists
    some T where B^I(x, T, K) >= K_0.
    
    Parameters
    ----------
    x_grid : array
        Earnings grid
    K_0 : float
        Required amount to raise
    K_grid : array
        Face value grid to search over
    T_grid : array
        Maturity grid to search over
    bond_val : BondValuation
        Bond valuation object
    K_bar_func : callable
        Refinancing capacity function
    verbose : bool
        Print progress
    
    Returns
    -------
    dict with arrays:
        x_bounds : array of x values where bounds exist
        K_min_bounds : array of minimum feasible face values
        K_max_bounds : array of maximum feasible face values
    """
    if verbose:
        print("\nComputing feasible face value bounds...")
    
    x_bounds = []
    K_min_bounds = []
    K_max_bounds = []
    
    for i, x in enumerate(x_grid):
        if verbose and i % 10 == 0:
            print(f"  Progress: {i}/{len(x_grid)}")
        
        # For each K, check if any T makes bond feasible
        feasible_K = []
        
        for K in K_grid:
            # Check if any T in grid satisfies B^I(x, T, K) >= K_0
            for T in T_grid:
                B_I = bond_val.B_illiquid(x, T, K, K_bar_func)
                if B_I >= K_0:
                    feasible_K.append(K)
                    break  # Found feasible T for this K, move to next K
        
        # If we found feasible face values, record bounds
        if len(feasible_K) > 0:
            x_bounds.append(x)
            K_min_bounds.append(min(feasible_K))
            K_max_bounds.append(max(feasible_K))
    
    if verbose:
        print(f"  Found feasible face value bounds for {len(x_bounds)}/{len(x_grid)} earnings levels")
    
    return {
        'x_bounds': np.array(x_bounds),
        'K_min_bounds': np.array(K_min_bounds),
        'K_max_bounds': np.array(K_max_bounds)
    }


def plot_results(x_grid, K_bar_array, results, params, output_dir, cir, bond_val, 
                 K_grid, T_grid, K_bar_func, maturity_bounds=None, face_value_bounds=None):
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
    
    # Panel 2: Heatmap of Firm Value V^I(x, T)
    ax2 = plt.subplot(3, 3, 2)
    if len(x_feas) > 0:
        # Create heatmap of firm value for feasible (x, T) combinations
        from firm_valuation import FirmValuation
        firm_val = FirmValuation(bond_val)
        
        # Create a function for V̄ approximation
        def V_bar_approx(x, K):
            F_x = bond_val.F_unlevered(x)
            if K_bar_func(x) >= K + bond_val.C:
                return F_x - K
            else:
                return max(0, F_x - K)
        
        # Create grid for heatmap (use subset for faster computation)
        x_heatmap = x_feas[::max(1, len(x_feas)//30)]  # Sample ~30 points
        T_heatmap = T_grid
        
        # Initialize value matrix (NaN for infeasible)
        V_matrix = np.full((len(T_heatmap), len(x_heatmap)), np.nan)
        
                # Compute firm values for feasible combinations
        for i, x in enumerate(x_heatmap):
            for j, T in enumerate(T_heatmap):
                # Find optimal K for this (x, T) pair
                best_V = -np.inf
                for K in K_grid:
                    B_I = bond_val.B_illiquid(x, T, K, K_bar_func)
                    if B_I >= params.K_0:
                        V_I = firm_val.V_illiquid(x, T, K, K_bar_func, V_bar_approx)
                        if V_I > best_V:
                            best_V = V_I
                
                if best_V > -np.inf:
                    V_matrix[j, i] = best_V
        
        # Diagnostic: Check variation in firm value at a sample earnings level
        if len(x_heatmap) > 10:
            sample_idx = len(x_heatmap) // 2  # Middle point
            sample_x = x_heatmap[sample_idx]
            sample_values = V_matrix[:, sample_idx]
            valid_values = sample_values[~np.isnan(sample_values)]
            if len(valid_values) > 1:
                print(f"\n  Diagnostic for x={sample_x:.2f}:")
                print(f"    Firm values across T: min={np.min(valid_values):.2f}, max={np.max(valid_values):.2f}")
                print(f"    Range: {np.max(valid_values) - np.min(valid_values):.2f}")
                print(f"    Coefficient of variation: {np.std(valid_values)/np.mean(valid_values)*100:.2f}%")
        
        # Plot heatmap
        im = ax2.imshow(V_matrix, aspect='auto', origin='lower',
                       extent=[x_heatmap[0], x_heatmap[-1], T_heatmap[0], T_heatmap[-1]],
                       cmap='viridis', interpolation='nearest')
        
        # Plot optimal maturity on top
        ax2.plot(x_feas, T_hat, 'r-', linewidth=3, label='Optimal T̂(x)')
        ax2.plot(x_feas, T_hat, 'w--', linewidth=1, alpha=0.5)  # White outline for visibility
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax2, label='Firm Value V^I')
        
        ax2.set_xlabel('Earnings x')
        ax2.set_ylabel('Maturity T')
        ax2.set_title('Optimal Maturity')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.2, color='white', linewidth=0.5)
    
    # ORIGINAL CODE (COMMENTED OUT) - Optimal Maturity with feasible bounds
    # ax2 = plt.subplot(3, 3, 2)
    # if len(x_feas) > 0:
    #     # Plot feasible maturity bounds if available
    #     if maturity_bounds is not None:
    #         x_b = maturity_bounds['x_bounds']
    #         T_min_b = maturity_bounds['T_min_bounds']
    #         T_max_b = maturity_bounds['T_max_bounds']
    #         
    #         # Fill between min and max feasible maturity
    #         ax2.fill_between(x_b, T_min_b, T_max_b, 
    #                         alpha=0.2, color='gray', label='Feasible region')
    #         
    #         # Plot bounds
    #         ax2.plot(x_b, T_min_b, '--', linewidth=1.5, color='orange', 
    #                 label='Min feasible T', alpha=0.7)
    #         ax2.plot(x_b, T_max_b, '--', linewidth=1.5, color='red', 
    #                 label='Max feasible T', alpha=0.7)
    #     
    #     # Plot optimal maturity
    #     ax2.plot(x_feas, T_hat, '-', linewidth=2.5, color='steelblue', 
    #             label='Optimal T̂(x)', zorder=10)
    #     
    #     ax2.set_xlabel('Earnings x')
    #     ax2.set_ylabel('Maturity')
    #     ax2.set_title('Optimal Maturity')
    #     ax2.legend(loc='best', fontsize=8)
    #     ax2.grid(True, alpha=0.3)
    
                # Panel 3: Heatmap of Firm Value V^I(x, K)
    ax3 = plt.subplot(3, 3, 3)
    if len(x_feas) > 0:
        # Create heatmap of firm value for feasible (x, K) combinations
        from firm_valuation import FirmValuation
        firm_val = FirmValuation(bond_val)
        
        # Create a function for V̄ approximation
        def V_bar_approx(x, K):
            F_x = bond_val.F_unlevered(x)
            if K_bar_func(x) >= K + bond_val.C:
                return F_x - K
            else:
                return max(0, F_x - K)
        
        # Create grid for heatmap (use subset for faster computation)
        x_heatmap = x_feas[::max(1, len(x_feas)//30)]  # Sample ~30 points
        K_heatmap = K_grid
        
        # Initialize value matrix (NaN for infeasible)
        V_matrix = np.full((len(K_heatmap), len(x_heatmap)), np.nan)
        
        # Compute firm values for feasible combinations
        for i, x in enumerate(x_heatmap):
            for j, K in enumerate(K_heatmap):
                # Find optimal T for this (x, K) pair
                best_V = -np.inf
                for T in T_grid:
                    B_I = bond_val.B_illiquid(x, T, K, K_bar_func)
                    if B_I >= params.K_0:
                        V_I = firm_val.V_illiquid(x, T, K, K_bar_func, V_bar_approx)
                        if V_I > best_V:
                            best_V = V_I
                
                if best_V > -np.inf:
                    V_matrix[j, i] = best_V
        
        # Diagnostic: Check variation in firm value at a sample earnings level
        if len(x_heatmap) > 10:
            sample_idx = len(x_heatmap) // 2  # Middle point
            sample_x = x_heatmap[sample_idx]
            sample_values = V_matrix[:, sample_idx]
            valid_values = sample_values[~np.isnan(sample_values)]
            if len(valid_values) > 1:
                print(f"\n  Diagnostic for x={sample_x:.2f}:")
                print(f"    Firm values across K: min={np.min(valid_values):.2f}, max={np.max(valid_values):.2f}")
                print(f"    Range: {np.max(valid_values) - np.min(valid_values):.2f}")
                print(f"    Coefficient of variation: {np.std(valid_values)/np.mean(valid_values)*100:.2f}%")
        
        # Plot heatmap
        im = ax3.imshow(V_matrix, aspect='auto', origin='lower',
                       extent=[x_heatmap[0], x_heatmap[-1], K_heatmap[0], K_heatmap[-1]],
                       cmap='viridis', interpolation='nearest')
        
        # Plot optimal face value on top
        ax3.plot(x_feas, K_hat, 'r-', linewidth=3, label='Optimal K̂(x)')
        ax3.plot(x_feas, K_hat, 'w--', linewidth=1, alpha=0.5)  # White outline for visibility
        
        # Add K_0 reference line
        ax3.axhline(params.K_0, color='cyan', linestyle=':', 
                   linewidth=2, alpha=0.8, label=f'K₀={params.K_0}')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax3, label='Firm Value V^I')
        
        ax3.set_xlabel('Earnings x')
        ax3.set_ylabel('Face Value K')
        ax3.set_title('Optimal Face Value')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.2, color='white', linewidth=0.5)
    
    # ORIGINAL CODE (COMMENTED OUT) - Optimal Face Value with feasible bounds
    # ax3 = plt.subplot(3, 3, 3)
    # if len(x_feas) > 0:
    #     # Plot feasible face value bounds if available
    #     if face_value_bounds is not None:
    #         x_k = face_value_bounds['x_bounds']
    #         K_min_k = face_value_bounds['K_min_bounds']
    #         K_max_k = face_value_bounds['K_max_bounds']
    #         
    #         # Fill between min and max feasible face value
    #         ax3.fill_between(x_k, K_min_k, K_max_k, 
    #                         alpha=0.2, color='gray', label='Feasible region')
    #         
    #         # Plot bounds
    #         ax3.plot(x_k, K_min_k, '--', linewidth=1.5, color='orange', 
    #                 label='Min feasible K', alpha=0.7)
    #         ax3.plot(x_k, K_max_k, '--', linewidth=1.5, color='red', 
    #                 label='Max feasible K', alpha=0.7)
    #     
    #     # Plot optimal face value
    #     ax3.plot(x_feas, K_hat, '-', linewidth=2.5, color='green', 
    #             label='Optimal K̂(x)', zorder=10)
    #     
    #     # Add K_0 reference line
    #     ax3.axhline(params.K_0, color='purple', linestyle=':', 
    #                linewidth=1.5, alpha=0.6, label=f'K₀={params.K_0}')
    #     
    #     ax3.set_xlabel('Earnings x')
    #     ax3.set_ylabel('Face Value')
    #     ax3.set_title('Optimal Face Value')
    #     ax3.legend(loc='best', fontsize=8)
    #     ax3.grid(True, alpha=0.3)
    
    # Panel 4: Yield Spread
    ax4 = plt.subplot(3, 3, 4)
    if len(x_feas) > 0:
        ax4.plot(x_feas, spread_bp, '-', 
                linewidth=2.5, color='purple')
        ax4.set_xlabel('Earnings x')
        ax4.set_ylabel('Spread (bp)')
        ax4.set_title('Yield Spread')
        ax4.grid(True, alpha=0.3)
    
    # Panel 5: Bond Value B̂(x)
    ax5 = plt.subplot(3, 3, 5)
    if len(x_feas) > 0:
        ax5.plot(x_feas, B_hat, '-', 
                linewidth=2.5, color='brown')
        ax5.axhline(params.K_0, color='r', linestyle='--', label='K₀', alpha=0.5)
        ax5.set_xlabel('Earnings x')
        ax5.set_ylabel('B̂(x)')
        ax5.set_title('Optimal Bond Value')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
                # Format y-axis to show decimals only when needed (no trailing zeros)
        from matplotlib.ticker import FuncFormatter
        def format_func(value, tick_number):
            # Format with high precision, then remove trailing zeros
            formatted = f'{value:.10f}'.rstrip('0').rstrip('.')
            return formatted
        ax5.yaxis.set_major_formatter(FuncFormatter(format_func))
    
        # Print diagnostics
        print(f"\nBond Value B̂(x) Statistics:")
        print(f"  Min: {np.min(B_hat):.2f}")
        print(f"  Max: {np.max(B_hat):.2f}")
        print(f"  Range: {np.max(B_hat) - np.min(B_hat):.2f}")
        print(f"  Mean: {np.mean(B_hat):.2f}")
        print(f"  Std Dev: {np.std(B_hat):.2f}")
    
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
    
    # Panel 8: Default probability (CIR-based)
    ax8 = plt.subplot(3, 3, 8)
    if len(x_feas) > 0:
        # Calculate true default probability using CIR distribution
        # P(default) = P(x_T < x*) = 1 - Q(x, T, x*)
        default_prob_cir = []
        for x, K, T in zip(x_feas, K_hat, T_hat):
            x_star = bond_val.x_star(K)  # Default threshold
            survival_prob = cir.Q(x, T, x_star)  # P(x_T > x*)
            default_prob = 1 - survival_prob  # P(x_T < x*)
            default_prob_cir.append(default_prob)
    
        ax8.plot(x_feas, np.array(default_prob_cir) * 100, '-', 
                linewidth=2.5, color='red')
        ax8.set_xlabel('Earnings x')
        ax8.set_ylabel('Default Prob (%)')
        ax8.set_title('Default Probability')
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


    output_dir = Path(__file__).parent / '../output/figures'
    output_dir.mkdir(parents=True, exist_ok=True)
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
    
    # Create non-uniform grids: 25% of points in low-value regions
    # For x: 17 points in [0.1, 2], 33 points in [2, 100]
    # For T: 10 points in [0.1, 2], 20 points in [2, 50]  
    x_grid = create_concentrated_grid(p.x_min, p.x_max, 50, dense_end=2.0, dense_fraction=0.25)
    K_grid = np.linspace(p.K_min, p.K_max, 40)   # Face value (keep uniform)
    T_grid = create_concentrated_grid(0.1, p.T_max, 30, dense_end=2.0, dense_fraction=0.25)
    
    print(f"\nGrid sizes: x={len(x_grid)}, K={len(K_grid)}, T={len(T_grid)}")
    print(f"Total computations: ~{len(x_grid) * len(K_grid) * len(T_grid):,}")
    
    # Initialize CIR process
    cir = CIRProcess(p.kappa, p.mu, p.sigma)
    print(f"\n{cir}")
    
    # Initialize bond valuation (guess before iteration)
    bond_val = BondValuation(cir, p.r, p.eta, p.C)
    
    # STEP 1: Solve for K̄*(x)
    print("\n" + "-"*70)
    print("STEP 1: Solving for Refinancing Set K̄*(x)")
    print("-"*70)
    
    ref_solver = RefinancingSetSolver(bond_val, x_grid, K_grid, T_grid)
    K_bar_array, K_bar_func = ref_solver.solve(tol=1e-6, max_iter=500, verbose=True)
    
    # STEP 2: Find optimal bonds
    print("\n" + "-"*70)
    print("STEP 2: Finding Optimal Bonds (K̂, T̂)")
    print("-"*70)
    
    # Choose optimization method
    # Options: 'grid_search', 'slsqp', 'nelder-mead', 'hybrid'
    optimization_method = 'slsqp'  # Change this to test different methods
    
    opt_solver = OptimalBondSolver(bond_val, K_bar_func, method=optimization_method)
    results = opt_solver.solve_for_grid(x_grid, p.K_0, K_grid, T_grid, verbose=True)
    
    # STEP 3: Save results
    print("\n" + "-"*70)
    print("STEP 3: Saving Results")
    print("-"*70)
    
    output_dir = Path('Maturity/data/results/base_case')
    output_dir.mkdir(parents=True, exist_ok=True)
    
        # STEP 4: Compute feasible maturity bounds
    print("\n" + "-"*70)
    print("STEP 4: Computing Feasible Maturity Bounds")
    print("-"*70)
    
    maturity_bounds = compute_feasible_maturity_bounds(
        x_grid, p.K_0, K_grid, T_grid, bond_val, K_bar_func, verbose=True
    )
    
    # Also compute feasible face value bounds
    print("\n" + "-"*70)
    print("STEP 4b: Computing Feasible Face Value Bounds")
    print("-"*70)
    
    face_value_bounds = compute_feasible_face_value_bounds(
        x_grid, p.K_0, K_grid, T_grid, bond_val, K_bar_func, verbose=True
    )
    
    save_data = {
        'params': p,
        'x_grid': x_grid,
        'K_grid': K_grid,
        'T_grid': T_grid,
        'K_bar_array': K_bar_array,
        'results': results,
        'maturity_bounds': maturity_bounds,
        'face_value_bounds': face_value_bounds
    }
    
    with open(output_dir / 'solution.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    
        print(f"✓ Saved to {output_dir / 'solution.pkl'}")
    
    # STEP 5: Plot results
    print("\n" + "-"*70)
    print("STEP 5: Plotting Results")
    print("-"*70)
    
    fig_dir = Path('output/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plot_results(x_grid, K_bar_array, results, p, fig_dir, cir, bond_val,
        K_grid, T_grid, K_bar_func, maturity_bounds, face_value_bounds)
    
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
        
                # Print maturity bounds statistics
        if len(maturity_bounds['x_bounds']) > 0:
            print(f"\nFeasible Maturity Bounds:")
            print(f"  Min feasible T: [{np.min(maturity_bounds['T_min_bounds']):.2f}, {np.max(maturity_bounds['T_min_bounds']):.2f}]")
            print(f"  Max feasible T: [{np.min(maturity_bounds['T_max_bounds']):.2f}, {np.max(maturity_bounds['T_max_bounds']):.2f}]")
            print(f"  Avg feasible range: {np.mean(maturity_bounds['T_max_bounds'] - maturity_bounds['T_min_bounds']):.2f}")
        
        # Print face value bounds statistics
        if len(face_value_bounds['x_bounds']) > 0:
            print(f"\nFeasible Face Value Bounds:")
            print(f"  Min feasible K: [{np.min(face_value_bounds['K_min_bounds']):.1f}, {np.max(face_value_bounds['K_min_bounds']):.1f}]")
            print(f"  Max feasible K: [{np.min(face_value_bounds['K_max_bounds']):.1f}, {np.max(face_value_bounds['K_max_bounds']):.1f}]")
            print(f"  Avg feasible range: {np.mean(face_value_bounds['K_max_bounds'] - face_value_bounds['K_min_bounds']):.1f}")
        
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
