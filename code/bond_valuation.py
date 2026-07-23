"""
Bond valuation in liquid and illiquid states.
"""

import numpy as np
from cir_process import CIRProcess


class BondValuation:
    """Compute bond values under different market conditions."""
    
    def __init__(self, cir_process, r, eta, C):
        self.cir = cir_process
        self.r = r
        self.eta = eta
        self.C = C
    
    def F_unlevered(self, x):
        """Unlevered firm value: F(x) = x/(r+κ) + κμ/(r(r+κ))"""
        return x / (self.r + self.cir.kappa) + \
               self.cir.kappa * self.cir.mu / (self.r * (self.r + self.cir.kappa))
    
    def x_star(self, K):
        """Default threshold in liquid state: x* = (r+κ)K - κμ/r"""
        x_star_val = (self.r + self.cir.kappa) * K - self.cir.kappa * self.cir.mu / self.r
        return max(x_star_val, 0)
    
    def B_liquid(self, x, T, K):
        """Bond value in liquid state: B^L = e^(-rT) K Q(x,T,x*)"""
        x_s = self.x_star(K)
        return np.exp(-self.r * T) * K * self.cir.Q(x, T, x_s)
    
    def B_illiquid(self, x, T, K, K_bar_func):
        """Bond value in illiquid state."""
        x_s = self.x_star(K)
        x_B = self._find_x_B(K, K_bar_func)
        
        term1 = K * np.exp(-(self.r + self.eta) * T) * self.cir.Q(x, T, x_B)
        term2 = K * (1 - np.exp(-self.eta * T)) * np.exp(-self.r * T) * self.cir.Q(x, T, x_s)
        
        return term1 + term2
    
    def _find_x_B(self, K, K_bar_func):
        """Find x_B such that K̄*(x_B) = K + C using bisection."""
        target = K + self.C
        
        # Check if solution exists
        x_max = 100.0
        if K_bar_func(x_max) < target:
            return np.inf
        
        # Bisection
        x_low, x_high = 0.01, x_max
        for _ in range(50):
            x_mid = (x_low + x_high) / 2
            K_bar_mid = K_bar_func(x_mid)
            
            if abs(K_bar_mid - target) < 0.01:
                return x_mid
            
            if K_bar_mid < target:
                x_low = x_mid
            else:
                x_high = x_mid
        
        return (x_low + x_high) / 2
