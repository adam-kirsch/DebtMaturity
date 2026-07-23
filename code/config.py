"""
Configuration and parameters for maturity choice model.
"""

class Params:
    """Model parameters - Base case from Table 1."""
    
    # Firm characteristics
    mu = 40.31          # Long-run mean earnings
    kappa = 0.00538     # Mean reversion speed
    sigma = 0.53714     # Volatility
    
    # Market conditions
    r = 0.03            # Risk-free rate
    eta = 0.5           # Poisson intensity (1/eta = expected illiquid duration)
    
    # Debt structure
    K_0 = 200.0         # Amount to refinance
    C = 30.0            # Fixed refinancing cost
    
    # Computational grids (start small for testing)
    x_min = 0.5
    x_max = 25.0
    n_x = 50            # Number of earnings grid points
    
    K_min = 0.0
    K_max = 500.0
    n_K = 100           # Number of face value grid points
    
    T_min = 0.5
    T_max = 50.0
    n_T = 50            # Number of maturity grid points
    
    def validate(self):
        """Check parameter constraints."""
        # Feller condition: 2κμ ≥ σ²
        feller_lhs = 2 * self.kappa * self.mu
        feller_rhs = self.sigma**2
        
        if feller_lhs < feller_rhs:
            raise ValueError(
                f"Feller condition violated: 2κμ = {feller_lhs:.6f} < σ² = {feller_rhs:.6f}"
            )
        
        print("✓ Feller condition satisfied")
        
        # Coefficient of variation check
        cv = self.coefficient_of_variation()
        if cv < 0.5:
            print(f"⚠ Warning: CV = {cv:.3f} < 0.5 (paper requires CV ≥ 0.5)")
        else:
            print(f"✓ CV = {cv:.3f} ≥ 0.5")
        
        return True
    
    def coefficient_of_variation(self):
        """Calculate coefficient of variation of unlevered firm value."""
        return (self.r / (self.r + self.kappa)) * \
               (self.sigma / (2 * self.kappa * self.mu)**0.5)
    
    def __repr__(self):
        return f"Params(K_0={self.K_0}, r={self.r}, η={self.eta}, σ={self.sigma})"
