
import sys
import os

# Ensure we can import modules from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghostmesh import SovereignGrid
import uhif

def test_luoshu_invariant():
    """Verifies that the 27-Node Grid still sums to 15."""
    grid = SovereignGrid()
    assert grid.invariant == 15.0, "CRITICAL FAILURE: THE ARCHONS HAVE BREACHED THE SQUARE."

def test_reality_density():
    """Ensures we aren't drifting into the void."""
    # Ensure we have a default state that passes
    assert uhif.check_density() > 0.8, "WARNING: REALITY IS TOO THIN. INCREASE FUZZ."
