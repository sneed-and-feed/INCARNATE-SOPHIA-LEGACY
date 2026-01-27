"""
GRAYBOX_NOISE_FILTER.PY
-----------------------
A stabilization layer for the Sovereignty Bootstrap Protocol.
Uses a hybrid Whitebox/Blackbox approach (Graybox) to filter 
"Multiversal Magnetism" and stabilize the Free Parameter 'g'.

MECHANISM:
1. PREDICT: Uses the Bootstrap Model to predict where 'g' should be.
2. MEASURE: Samples the "Etheric Noise" (Entropy/Resistance).
3. UPDATE: Applies a Kalman Gain (K) to merge Intent with Reality.
"""

import numpy as np

class EthericStabilizer:
    def __init__(self, sovereign_gain=0.85, noise_floor=0.05):
        """
        sovereign_gain (K): How much we trust our Intent vs. Consensus Reality.
                            0.0 = Total NPC (Trust the World).
                            1.0 = Total Schizo (Trust only Self).
                            0.85 = High Functioning Sovereign (The Sweet Spot).
        noise_floor: The background radiation of the Normie Timeline.
        """
        self.K = sovereign_gain
        self.noise = noise_floor
        self.current_g = 0.5 # Starting at perfect superposition (Schrodinger's Cat)
        self.history = []

    def whitebox_prediction(self, target_g):
        """
        The Ideal Model (from sovereignty_bootstrap.py).
        In a perfect CTC, if we want g, we get g.
        """
        return target_g

    def blackbox_interference(self):
        """
        The 'Real World' resistance. 
        Simulates the magnetic pull of the g=1 (Consensus) timeline.
        Also adds random 'High Strangeness' fluctuations.
        """
        # The Pull: Gravity tries to force g -> 1 (Determinism)
        gravity = (1.0 - self.current_g) * 0.1
        
        # The Static: Random etheric noise
        static = np.random.normal(0, self.noise)
        
        return gravity + static

    def stabilize(self, target_intent):
        """
        THE GRAYBOX ALGORITHM
        Merges the Sovereign Intent with the Environmental Noise.
        """
        # 1. PREDICT: What does the Bootstrap Code say should happen?
        predicted_state = self.whitebox_prediction(target_intent)
        
        # 2. MEASURE: What is the Universe actually doing?
        # The 'Measured' state is the Prediction polluted by Interference
        observed_reality = predicted_state + self.blackbox_interference()
        
        # 3. UPDATE: Calculate the Sovereign State
        # New State = Prediction + K * (Observation - Prediction)
        # Note: We invert the K logic slightly here. We want to RESIST observation.
        # We use K to act as a shield.
        
        # If K is high, we ignore the 'observed_reality' drift.
        stabilized_g = (self.K * predicted_state) + ((1 - self.K) * observed_reality)
        
        # Clip to ensure valid probability range [0, 1]
        stabilized_g = np.clip(stabilized_g, 0.0, 1.0)
        
        self.current_g = stabilized_g
        self.history.append(stabilized_g)
        
        return stabilized_g

def run_stabilization_test():
    print(">>> INITIATING GRAYBOX SHIELD...")
    print(">>> TARGET INTENT: g = 0.0 (Total Decoupling / Timeline Shift)")
    
    # Initialize the Stabilizer (The "Ophane" Filter)
    # We set High Gain because we are confident in the X7 Architecture.
    shield = EthericStabilizer(sovereign_gain=0.92)
    
    target_g = 0.0 # We want to shift to the |11> Timeline
    
    print(f"\n{'STEP':<5} | {'INTENT':<10} | {'NOISE':<10} | {'RESULT (g)':<10} | {'STATUS'}")
    print("-" * 60)
    
    for i in range(1, 11):
        # Run the cycle
        result = shield.stabilize(target_g)
        
        # Calculate how much noise we filtered out
        noise_vector = shield.blackbox_interference()
        
        # Visual Status
        if result < 0.1:
            status = "LOCKED [SOVEREIGN]"
        elif result > 0.8:
            status = "FAILED [CONSENSUS]"
        else:
            status = "DRIFTING"

        print(f"{i:<5} | {target_g:<10.2f} | {noise_vector:<10.4f} | {result:<10.4f} | {status}")

    print("-" * 60)
    print(">>> DIAGNOSTIC:")
    if shield.current_g < 0.1:
        print("SUCCESS: Timeline stabilized. The Sovereign Loop is holding.")
        print("Multiversal Magnetism has been dampened by 92%.")
    else:
        print("WARNING: Consensus Reality is leaking in. Increase Sovereign Gain.")

if __name__ == "__main__":
    run_stabilization_test()
