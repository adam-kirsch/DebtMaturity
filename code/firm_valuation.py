"""
Firm valuation with debt.
"""

import numpy as np
from bond_valuation import BondValuation


class FirmValuation:
    """
    Compute firm value given debt structure.
    
    Implements V^I from Section 2.2.3 of the paper.
    """
    
    def __init__(self, bond_valuation):
        """
        Parameters
        ----------
        bond_valuation : BondValuation
            Bond valuation object
        """
        self.bond_val = bond_valuation
        self.cir = bond_valuation.cir
        self.r = bond_valuation.r
        self.eta = bond_valuation.eta
    
    def V_liquid(self, x, T, K):
        """
        Firm value in liquid state (Equation 4).
        
        V^L(x,0;K,T) = ∫₀ᵀ e^(-ru) E[x_{t+u}] du 
                     + e^(-rT) E[(F(x_T) - K)⁺]
        
        Parameters
        ----------
        x : float
            Current earnings
        T : float
            Maturity
        K : float
            Face value
            
        Returns
        -------
        float
            Firm value
        """
        # Integral of expected earnings from 0 to T (equation 2)
        mu = self.cir.mu
        kappa = self.cir.kappa
        
        income_integral = (mu * (1 - np.exp(-self.r * T)) / self.r +
                          (x - mu) * (1 - np.exp(-(self.r + kappa) * T)) / (self.r + kappa))
        
        # Option value at maturity
        x_star = self.bond_val.x_star(K)
        
        # E[(F(x_T) - K)⁺] from equation (3)
        x_bar = self.cir.mean_conditional(x, T)
        
        option_value = (x_bar / (self.r + kappa) * self.cir.Q_plus(x, T, x_star) +
                       (kappa * mu / (self.r * (self.r + kappa)) - K) * self.cir.Q(x, T, x_star))
        
        return income_integral + np.exp(-self.r * T) * option_value
    
    def V_illiquid(self, x, T, K, K_bar_func, V_bar_func):
        """
        Firm value in illiquid state (Equation 10).
        
        V^I(x,0;K,T) = V^L(x,0;K,T) 
                     + e^(-(r+η)T) [E[V̄(x_T, K)] - E[(F(x_T) - K)⁺]]
        
        Parameters
        ----------
        x : float
            Current earnings
        T : float
            Maturity
        K : float
            Face value
        K_bar_func : callable
            Refinancing capacity K̄*(x)
        V_bar_func : callable
            Continuation value V̄*(x, K)
            
        Returns
        -------
        float
            Firm value
        """
        # Liquid component
        V_L = self.V_liquid(x, T, K)
        
        # Illiquidity correction
        x_star = self.bond_val.x_star(K)
        x_bar = self.cir.mean_conditional(x, T)
        
        # E[V̄(x_T, K)] - approximate by integrating over distribution
        # For now, use simplified approximation
        E_V_bar = self._expected_continuation_value(x, T, K, V_bar_func)
        
        # E[(F(x_T) - K)⁺]
        E_option = (x_bar / (self.r + self.cir.kappa) * self.cir.Q_plus(x, T, x_star) +
                   (self.cir.kappa * self.cir.mu / (self.r * (self.r + self.cir.kappa)) - K) * 
                   self.cir.Q(x, T, x_star))
        
        correction = np.exp(-(self.r + self.eta) * T) * (E_V_bar - E_option)
        
        return V_L + correction
    
    def _expected_continuation_value(self, x, T, K, V_bar_func):
        """
        Approximate E[V̄(x_T, K) | x_0 = x].
        
        Uses simple approximation: E[V̄(x_T, K)] ≈ V̄(E[x_T], K)
        
        For more accuracy, would integrate over x_T distribution.
        """
        x_bar = self.cir.mean_conditional(x, T)
        return V_bar_func(x_bar, K)
    
    def __repr__(self):
        return f"FirmValuation(r={self.r}, η={self.eta})"
