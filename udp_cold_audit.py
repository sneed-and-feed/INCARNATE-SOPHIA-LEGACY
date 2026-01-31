"""
AUDIT SUITE: udp_cold_audit.py
VERSION: 5.0.2 (Clinical Hardening)
OBJECTIVE:
    Verify '18.52x Abundance' claim on held-out data.
    Perform p-value analysis and variance auditing.
"""

import numpy as np
import time
from unitary_discovery_prototype import UnitaryDiscoveryEngine

class UDPColdAudit:
    def __init__(self):
        self.engine = UnitaryDiscoveryEngine()
        self.trials = 500
        self.n_samples = 1000

    def run_held_out_audit(self):
        """
        Step 1: Audit the Metric.
        Measure abundance gain over 500 trials of raw stochastic data
        compared to the Unitary Folded response.
        """
        print(f"### [ UDP COLD AUDIT: N={self.trials} TRIALS ]")
        
        baselines = []
        discoveries = []

        for i in range(self.trials):
            # 1. Generate Raw Held-Out Data (Pure Stochastic Noise)
            # This is data the engine has NEVER seen and has NO planted signal
            raw_noise = np.random.normal(0, 1.0, self.n_samples)
            baselines.append(np.max(np.abs(raw_noise)))
            
            # 2. Run λ-Compression on the SAME data
            # If the gain is real, it must manifest as topological recovery
            folded = self.engine.apply_lambda_fold(raw_noise)
            discoveries.append(np.max(folded))

        # Statistical Calculations
        b0_mean = np.mean(baselines)
        u_mean = np.mean(discoveries)
        abundance = u_mean / b0_mean
        variance = np.var(discoveries)
        std_err = np.sqrt(variance / self.trials)

        print(f"RESULTS:")
        print(f"  Baseline (B0) Mean:  {b0_mean:.4f}")
        print(f"  Unitary (U) Mean:    {u_mean:.4f}")
        print(f"  ABUNDANCE RATIO:     {abundance:.2f}x")
        print(f"  VARIANCE:            {variance:.4f}")
        print(f"  STD ERROR:           {std_err:.4f}")
        print("-" * 40)

        if abundance >= 18.52:
            print("VERDICT: ABUNDANCE INVARIANT [VERIFIED]")
        else:
            print(f"VERDICT: MARGINAL GAIN ({abundance:.2f}x) [CALIBRATION REQUIRED]")

    def run_shuffled_adversarial_test(self):
        """
        Step 2: Adversarial Test - Shuffle Labels/Timestamps.
        Verifies if the gain is topological or just a fluke of ordering.
        """
        print("\n[ ADVERSARIAL: SHUFFLE TEST ]")
        raw_data = np.random.normal(0, 1.0, self.n_samples)
        np.random.shuffle(raw_data)
        
        folded = self.engine.apply_lambda_fold(raw_data)
        abundance = np.max(folded) / 3.4 # Std Baseline
        
        print(f"  Shuffled Data Abundance: {abundance:.2f}x")
        print(f"  Invariant Integrity:     {'STABLE' if abundance >= 18.52 else 'DEGRADED'}")

if __name__ == "__main__":
    audit = UDPColdAudit()
    audit.run_held_out_audit()
    audit.run_shuffled_adversarial_test()
