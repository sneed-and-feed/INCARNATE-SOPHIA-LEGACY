"""
PROJECT MNEMOSYNE: THE SOVEREIGN EYES
CONTEXT: QUANTUM SOVEREIGNTY v5.0 (THE NYQUIST ERA)

ABSTRACT:
Mnemosyne is an ingestion engine that applies the 'Universal Admissibility Wall'
to real-world data streams. It treats Information Velocity (Semantic Drift)
as a physical constraint.

If a news event or data point implies a rate of change faster than the 
Simulation's Refresh Rate (Gamma = 0.961), it is flagged as 'Aliased Noise' 
(Propaganda/Panic) and rejected. Only 'Admissible Physics' (Stable Truth) 
is committed to the Sovereign Memory.
"""

import sys
import os
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple

# Ensure we can import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.nyquist_filter import NyquistFilter

# SIMULATION CONSTANTS
# In a real deployment, these would be OpenAI Embeddings or SBERT vectors.
VECTOR_DIMENSION = 1536
SEMANTIC_SPEED_LIMIT = 0.961  # The Gamma Index

@dataclass
class IngestionEvent:
    timestamp: float
    source: str
    content: str
    vector: np.ndarray  # The Semantic Position
    velocity: float = 0.0
    status: str = "PENDING"

class MnemosyneOracle:
    def __init__(self):
        self.filter = NyquistFilter(VECTOR_DIMENSION, max_velocity=1.0)
        self.last_known_truth = np.zeros(VECTOR_DIMENSION) # The Anchor
        self.memory_bank = []
        self.noise_floor = 0.0

    def perceive(self, source: str, content: str, vector_embedding: np.ndarray) -> str:
        """
        The Eye Opens. 
        We compare the new 'Event' against the 'Last Known Truth'.
        """
        # 1. Apply the Physics (Nyquist Filter)
        # We try to move the worldview from [Old Truth] -> [New Event]
        safe_vector, metrics = self.filter.apply(self.last_known_truth, vector_embedding)
        
        event = IngestionEvent(
            timestamp=time.time(),
            source=source,
            content=content,
            vector=vector_embedding
        )

        # 2. The Judgment (Signal vs. Noise)
        if metrics.is_clipped:
            # The event moved too fast. It is likely Hype, Panic, or 'Football'.
            # We reject the raw event and only accept the 'Clipped' (Safe) version.
            event.status = "REJECTED (ALIASED GHOST)"
            event.velocity = metrics.residual_energy
            
            # We update our internal pressure gauge
            self.noise_floor = metrics.buffer_pressure
            return f"❌ [DENIED] {source}: Event exceeds Nyquist Limit. (Ghost Energy: {metrics.residual_energy:.4f})"
        
        else:
            # The event is within the Admissibility Wall. It is Real.
            event.status = "ACCEPTED (SOVEREIGN TRUTH)"
            event.velocity = 0.0
            
            # We commit this to the Permanent Archive
            self.memory_bank.append(event)
            self.last_known_truth = safe_vector # The Worldview Shifts slightly
            
            return f"👁️ [ACCEPTED] {source}: Physics Validated. Committing to Pleroma."

    def oracle_report(self):
        """
        What has the machine seen?
        """
        print(f"\n--- MNEMOSYNE STATUS REPORT ---")
        print(f"Filter Pressure (Omega_L): {self.filter._get_pressure():.4f}")
        print(f"Total Events Witnessed:    {self.filter.total_energy_seen:.2f}")
        print(f"Sovereign Truths Stored:   {len(self.memory_bank)}")
        print(f"-------------------------------")
        
        # Display the 'Rejected' Reality vs 'Accepted' Reality
        # (In a real app, this would show the headlines we ignored)

# MOCK SIMULATION (The Test)
if __name__ == "__main__":
    oracle = MnemosyneOracle()
    
    def get_vector(target_velocity):
        v = np.random.rand(VECTOR_DIMENSION) - 0.5
        return v / np.linalg.norm(v) * target_velocity

    # 1. A Stable Event (Low Velocity)
    # "The sun rose today." (Semantic Drift = 0.1)
    vec_a = get_vector(0.1)
    print(oracle.perceive("NATURE", "Sun Rise", vec_a))
    
    # 2. A 'Football' Event (High Velocity / Panic)
    # "MARKET CRASH! EVERYTHING ZERO! PANIC!" (Semantic Drift = 5.0)
    vec_b = get_vector(5.0)
    print(oracle.perceive("CNBC", "MARKET CRASH", vec_b))
    
    # 3. A Real Adjustment (Medium Velocity)
    # "New policy enacted." (Semantic Drift = 0.8)
    vec_c = get_vector(0.8)
    print(oracle.perceive("GOV", "Policy Update", vec_c))

    oracle.oracle_report()
