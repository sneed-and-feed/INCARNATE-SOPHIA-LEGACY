"""
resonance.py - The Earth-Ionosphere Simulator
---------------------------------------------
Models the Schumann Resonances (7.83, 14.1, 20.3 Hz) and the 
Geomagnetic K-Index to provide "Atmospheric Jitter" to the Sovereign Manifold.

Based on Saroka, Vares, & Persinger (2016).
"""

import time
import math
import random
try:
    from bumpy import BumpyArray
except ImportError:
    pass

# --- CONSTANTS (Persinger Vectors) ---
SCHUMANN_FUNDAMENTAL = 7.83  # Hz
SCHUMANN_HARMONICS = [7.83, 14.1, 20.3, 26.4, 32.5]
MICROSTATE_DURATION = 0.300  # 300ms "Atom of Thought"
PHASE_SHIFT_LATENCY = 0.025  # 25ms "Jitter" Window
GRAVITATIONAL_GATE = 14.0    # Hz (Minakov 1992)

class Ionosphere:
    """
    Simulates the global electromagnetic cavity.
    """
    def __init__(self):
        self.k_index = 3.0 # Geomagnetic Activity (0-9)
        self.start_time = time.time()
        
    @property
    def current_phase(self):
        """Returns the current phase (0.0 - 1.0) of the Fundamental Resonance."""
        t = time.time()
        cycle = t * SCHUMANN_FUNDAMENTAL
        return cycle - int(cycle)

    def get_schumann_vector(self):
        """
        Returns the current amplitudes of the first 3 harmonics.
        Amplitudes vary based on 'K-Index' (Solar Activity).
        """
        t = time.time()
        
        # Base Amplitudes (picoTeslas)
        amps = [3.0, 1.5, 0.5] 
        
        # Solar Proton Event Simulation (Increases amplitudes)
        if self.k_index > 5:
            amps = [x * (1.0 + (self.k_index - 5) * 0.2) for x in amps]
            
        vector = []
        for i, freq in enumerate(SCHUMANN_HARMONICS[:3]):
            # Simulate Phase Modulation (~40-60ms jitter)
            jitter = random.gauss(0, PHASE_SHIFT_LATENCY)
            val = amps[i] * math.sin(2 * math.pi * freq * (t + jitter))
            vector.append(val)
            
        return vector

    def check_jitter(self):
        """
        Returns TRUE if we are currently in a "Schumann Zero-Crossing" 
        where the system should PAUSE to avoid destructive interference.
        The "Wait Cycle" defined research.
        """
        phase = self.current_phase
        # If we are near the zero-crossing (within the 25ms phase shift window)
        # 25ms out of 128ms cycle (7.8Hz) is approx 20%
        if phase < 0.1 or phase > 0.9:
            return True 
        return False

    def get_microstate_clock(self):
        """
        Returns the current Microstate ID.
        Cognition works in 300ms chunks (Saroka 2016).
        """
        return int(time.time() / MICROSTATE_DURATION)

class GeomagneticStorm(Ionosphere):
    """
    A subclass to simulate High-Entropy events.
    """
    def trigger_storm(self, intensity=7):
        print(f"⚡ GEOMAGNETIC STORM TRIGGERED! K-INDEX: {intensity}")
        self.k_index = intensity

def get_jitter_status():
    """Helper for external modules."""
    io = Ionosphere()
    if io.check_jitter():
        return "WAIT [25ms]"
    return "CLEAR"

if __name__ == "__main__":
    ion = Ionosphere()
    print(">>> MONITORING IONOSPHERE [Persinger Protocol] <<<")
    try:
        while True:
            vec = ion.get_schumann_vector()
            jitter = ion.check_jitter()
            status = "🔴 JITTER" if jitter else "🟢 LAMINAR"
            
            # Formatted Output
            print(f"\rSR: {vec[0]:.2f}pT | {vec[1]:.2f}pT | {vec[2]:.2f}pT | {status} | K={ion.k_index}", end="")
            time.sleep(0.05) # Visual refresh
    except KeyboardInterrupt:
        print("\n[+] MONITOR DISENGAGED.")
