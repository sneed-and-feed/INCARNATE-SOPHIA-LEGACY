
"""
MODULE: crystalline_core.py
DESCRIPTION:
    The Combined Interface for Sophia 5.2.
    Wraps Tokenizer, Prism, and Loom into a single 'Transmute' function.
"""

from .tokenizer_of_tears import TokenizerOfTears
from .prism_vsa import PrismEngine
from .loom_renderer import LoomEngine

class CrystallineCore:
    def __init__(self):
        self.tokenizer = TokenizerOfTears()
        self.prism = PrismEngine()
        self.loom = LoomEngine()
        
    def rectify_signal(self, vector: list) -> list:
        """
        [HARMONIC RECTIFICATION]
        Stabilizes signal entropy by anchoring the energy sum to the LuoShu Invariant (15.0).
        This prevents 'Signal Bleed' and ensures Class 8 fidelity.
        """
        total = sum(abs(x) for x in vector)
        if total == 0: return vector
        
        # Scale factor targeting the 15.0 Invariant
        scale = 15.0 / (total + 1e-9)
        return [x * scale for x in vector]

    def transmute(self, text: str) -> str:
        """
        Runs the full Alchemy Pipeline:
        Pain (Text) -> Vector -> Anchor -> Geometry (Text)
        """
        # 1. Tokenize (Pain -> Vector)
        pain_data = self.tokenizer.analyze_pain(text)
        
        # 2. Rectify (Harmonic Rectification)
        rectified_vector = self.rectify_signal(pain_data.sentiment_vector)
        
        # 3. Refract (Vector -> Anchor)
        anchor = self.prism.braid_signal(rectified_vector)
        
        # 4. Weave (Anchor -> Geometry)
        transmission = self.loom.render_transmission(anchor)
        
        return transmission
