"""
silver_sulfide.py - The Argentum-Sulfur Interface
-------------------------------------------------
Implements the "High Strangeness" optical properties of Ag2S at 400nm.
Based on Sovereignty Protocol research into anomalous nonlinear optics.

"The Gate is not open, it is merely Translucent."
"""

import math
import random
import time

# --- CONSTANTS ---
AG2S_INPUT_WAVELENGTH = 400.0   # nm (Violet - The Threshold)
AG2S_OUTPUT_WAVELENGTH = 480.0  # nm (Cyan - The Emission)
NONLINEAR_ABS_COEFF = 1.0e-3    # cm/W (Giant Nonlinearity)
QUANTUM_DOT_SIZE = 4.0          # nm (Confinement Radius)
REFRACTIVE_INDEX = 2.5          # Standard Thin Film n
THRESHOLD_INTENSITY = 0.85      # Arbitrary unit for "Bit-Flip"

class Ag2S_Nonlinear_Gate:
    """
    A simulated optical gate based on Silver Sulfide quantum dots.
    
    Function:
    - Filters low-intensity signals (Noise).
    - Nonlinear transmission for high-intensity signals (Truth).
    - Shifts frequency from 400nm to 480nm (Reality Shift).
    """
    
    def __init__(self, film_thicknessN=30):
        self.thickness = film_thicknessN # Angstroms
        self.state = "OPAQUE"
        self.last_transmission = 0.0
        
    def refract(self, wavelength):
        """
        Calculates the Refractive Index at a specific wavelength.
        Includes anomalous dispersion near 400nm.
        """
        if abs(wavelength - AG2S_INPUT_WAVELENGTH) < 10:
            # Anomalous Dispersion spike near resonance
            return REFRACTIVE_INDEX + 0.5 * random.random()
        return REFRACTIVE_INDEX

    def transmute_signal(self, input_intensity, wavelength):
        """
        The Core Logic.
        Accepts a signal. If it matches the Key (400nm) and has sufficient Will (Intensity),
        it is Transmuted (Shifted) and Transmitted.
        
        Returns:
            (output_intensity, output_wavelength, status_flag)
        """
        # 1. Check Spectrum (The Key)
        if abs(wavelength - AG2S_INPUT_WAVELENGTH) > 5.0:
            # Wrong frequency. Linear absorption.
            # Ag2S has high absorption in UV.
            return (input_intensity * 0.1, wavelength, "BLOCKED_WRONG_FREQ")
            
        # 2. Check Intensity (The Will)
        # Apply Nonlinear Optical Limiting (Reverse Saturable Absorption logic inverted for Gateway?)
        # Actually, for a "Gate", usually we want it to OPEN at high intensity (Saturable Absorption)
        # or CLOSE (Optical Limiting).
        # "Sovereignty Protocol" implies we want to filter OUT weak signals/noise.
        # So we model Saturable Absorption (Transparency at High Intensity).
        
        # Transmission T = T0 + beta * I (Simplified model for the sake of the metaphor)
        # Let's use a Sigmoid function to represent the "Opening" of the gate.
        
        transmission_factor = 1.0 / (1.0 + math.exp(-10.0 * (input_intensity - THRESHOLD_INTENSITY)))
        
        if input_intensity > THRESHOLD_INTENSITY:
            self.state = "TRANSMISSIVE"
            
            # 3. Frequency Shift (The High Strangeness)
            # The 400nm photon is absorbed, creating an exciton which decays via the 480nm channel.
            output_wavelength = AG2S_OUTPUT_WAVELENGTH
            
            # Add some "Quantum Jitter" (Time Domain Interference)
            # Simulating the transient state bleaching
            if random.random() > 0.9:
                # Occasional packet loss due to transient absorption
                transmission_factor *= 0.1 
                status = "TRANSIENT_INTERFERENCE"
            else:
                status = "TRANSMUTED_CYAN"
                
            return (input_intensity * transmission_factor, output_wavelength, status)
            
        else:
            self.state = "ABSORPTIVE"
            # Weak signals are absorbed into the heat bath
            return (input_intensity * 0.01, wavelength, "ABSORBED_WEAK_SIGNAL")

    def measure_biophoton_interaction(self, external_field_tesla):
        """
        Theoretical interaction with weak magnetic fields.
        Reference: High Strangeness - Biological correlations.
        """
        # Ag2S is diamagnetic but at 4nm scale, surface effects dominate.
        # High strangeness implies sensitivity to observation.
        if external_field_tesla > 1e-6: # Microtesla range
            # Field interference
            return "PERTURBED"
        return "STABLE"

    def __repr__(self):
        return f"<Ag2S_Gate: {self.state} | {self.thickness}A | n={REFRACTIVE_INDEX}>"

if __name__ == "__main__":
    # Test the Gate
    gate = Ag2S_Nonlinear_Gate()
    print(">>> INITIALIZING SILVER SULFIDE GATE [Ag2S] <<<")
    
    test_signals = [0.1, 0.5, 0.8, 0.9, 1.2, 5.0]
    
    for sig in test_signals:
        i_out, w_out, status = gate.transmute_signal(sig, 400.0)
        print(f"IN: {sig:.2f} @ 400nm | OUT: {i_out:.2f} @ {w_out:.1f}nm | STATUS: {status}")
