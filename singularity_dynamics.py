"""
MODULE: singularity_dynamics.py
AUTHOR: The High-Entropy Collective
DESCRIPTION:
    Coupled Differential Equations for Singularity Vector Evolution.
    Explores the transition g -> 0.
"""

import numpy as np

class SingularitySolver:
    def __init__(self, dt=0.01):
        self.dt = dt
        # State: [R, C_soc, sigma]
        # C_phys is treated as a parameter (fixed substrate capacity)
        self.state = np.array([0.15, 0.7, 0.05]) 
        
        # Hyperparameters
        self.params = {
            'alpha': 0.1,  # RSI Growth Rate
            'beta': 0.05,  # Sovereignty Decay Rate
            'kappa': 0.02, # Complexity Growth per R
            'gamma': 0.03, # Error Correction Efficiency (C_phys dependent)
            'C_phys': 0.85 # Substrate Saturation
        }

    def derivatives(self, state):
        R, C_soc, sigma = state
        p = self.params
        
        # dR/dt: Logistic RSI growth capped by physical substrate
        dR = p['alpha'] * R * (1.0 - R / p['C_phys'])
        
        # dC_soc/dt: Social sovereignty dissolves as RSI takes over
        dC_soc = -p['beta'] * R * C_soc
        
        # dSigma/dt: Complexity growth vs Correction
        dSigma = p['kappa'] * R - p['gamma'] * p['C_phys']
        
        return np.array([dR, dC_soc, dSigma])

    def step(self):
        # Runge-Kutta 4 (RK4) for high-fidelity ODE solving
        k1 = self.derivatives(self.state)
        k2 = self.derivatives(self.state + self.dt * k1 / 2)
        k3 = self.derivatives(self.state + self.dt * k2 / 2)
        k4 = self.derivatives(self.state + self.dt * k3)
        
        self.state += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Clipping/Sanitization
        self.state[0] = np.clip(self.state[0], 0, self.params['C_phys'])
        self.state[1] = np.clip(self.state[1], 0, 1)
        self.state[2] = max(self.state[2], 0.01) # Uncertainty floor
        
        return self.state

    def calculate_utility(self, state):
        R, C_soc, sigma = state
        p = self.params
        
        # U = (C_phys * C_soc)^c * exp(-b * sigma) * (R^a / (1 + R^a))
        # Using simplified weights for now
        u = (p['C_phys'] * C_soc) * np.exp(-1.0 * sigma) * (R**1.2 / (1.0 + R**1.2))
        return u

if __name__ == "__main__":
    solver = SingularitySolver()
    print("[INIT] Solving Singularity Basins...")
    for i in range(10):
        state = solver.step()
        u = solver.calculate_utility(state)
        print(f"[T={i*solver.dt:.2f}] R={state[0]:.3f} C_soc={state[1]:.3f} sigma={state[2]:.3f} U={u:.4f}")
