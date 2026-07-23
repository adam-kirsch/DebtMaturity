"""
CIR (Cox-Ingersoll-Ross) stochastic process implementation.

The CIR process: dx_t = κ(μ - x_t)dt + σ√x_t dW_t
"""

import numpy as np
from scipy.stats import ncx2


class CIRProcess:
    """
    Cox-Ingersoll-Ross mean-reverting process.
    
    Under the Feller condition (2κμ ≥ σ²), process stays non-negative.
    Conditional distribution is a scaled non-central chi-square.
    """
    
    def __init__(self, kappa, mu, sigma):
        """
        Parameters
        ----------
        kappa : float
            Mean reversion speed
        mu : float
            Long-run mean
        sigma : float
            Volatility coefficient
        """
        self.kappa = kappa
        self.mu = mu
        self.sigma = sigma
        
        # Pre-compute degrees of freedom (constant)
        self._nu = 4 * kappa * mu / (sigma**2)
        
        # Validate Feller condition
        if 2 * kappa * mu < sigma**2:
            raise ValueError(
                f"Feller condition violated: 2κμ={2*kappa*mu:.6f} < σ²={sigma**2:.6f}"
            )
    
    @property
    def nu(self):
        """Degrees of freedom for chi-square distribution."""
        return self._nu
    
    def s_tau(self, tau):
        """
        Scale parameter for chi-square distribution.
        
        s_τ = 2κ / [σ²(1 - e^(-κτ))]
        
        Parameters
        ----------
        tau : float or array
            Time horizon
            
        Returns
        -------
        float or array
            Scale parameter
        """
        return 2 * self.kappa / (self.sigma**2 * (1 - np.exp(-self.kappa * tau)))
    
    def lambda_param(self, x, tau):
        """
        Non-centrality parameter for chi-square distribution.
        
        λ(x,τ) = 2s_τ x e^(-κτ)
        
        Parameters
        ----------
        x : float or array
            Current value
        tau : float or array
            Time horizon
            
        Returns
        -------
        float or array
            Non-centrality parameter
        """
        return 2 * self.s_tau(tau) * x * np.exp(-self.kappa * tau)
    
    def mean_conditional(self, x, tau):
        """
        Conditional expectation: E[x_T | x_t = x]
        
        x̄(x,τ) = μ(1 - e^(-κτ)) + x e^(-κτ)
        
        Parameters
        ----------
        x : float or array
            Current value
        tau : float or array
            Time horizon
            
        Returns
        -------
        float or array
            Conditional mean
        """
        exp_term = np.exp(-self.kappa * tau)
        return self.mu * (1 - exp_term) + x * exp_term
    
    def Q(self, x, tau, y):
        """
        Survival probability: P(x_T ≥ y | x_t = x)
        
        Uses complementary CDF of non-central chi-square distribution.
        
        Parameters
        ----------
        x : float or array
            Current value
        tau : float
            Time horizon
        y : float
            Threshold value
            
        Returns
        -------
        float or array
            Probability x_T ≥ y
        """
        if y <= 0:
            return 1.0
        
        threshold = 2 * self.s_tau(tau) * y
        lambda_val = self.lambda_param(x, tau)
        
        return 1 - ncx2.cdf(threshold, self.nu, lambda_val)
    
    def Q_plus(self, x, tau, y):
        """
        Shifted survival probability (for truncated expectations).
        
        Q⁺(x,τ,y) = 1 - F_χ²(2s_τy; ν+2, λ)
        
        Parameters
        ----------
        x : float or array
            Current value
        tau : float
            Time horizon
        y : float
            Threshold value
            
        Returns
        -------
        float or array
            Shifted survival probability
        """
        if y <= 0:
            return 1.0
        
        threshold = 2 * self.s_tau(tau) * y
        lambda_val = self.lambda_param(x, tau)
        
        return 1 - ncx2.cdf(threshold, self.nu + 2, lambda_val)
    
    def truncated_expectation(self, x, tau, y):
        """
        Truncated first moment: E[x_T · 1_{x_T ≥ y} | x_t = x]
        
        Formula: x̄(x,τ) Q⁺(x,τ,y)
        
        Parameters
        ----------
        x : float
            Current value
        tau : float
            Time horizon
        y : float
            Threshold value
            
        Returns
        -------
        float
            Truncated expectation
        """
        return self.mean_conditional(x, tau) * self.Q_plus(x, tau, y)
    
    def __repr__(self):
        return f"CIRProcess(κ={self.kappa:.5f}, μ={self.mu:.2f}, σ={self.sigma:.5f})"
