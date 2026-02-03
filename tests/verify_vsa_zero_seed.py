
import sys
import os
import random
import numpy as np

# Adjust path so we can import sophia modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from sophia.cortex.prism_vsa import PrismEngine
except ImportError:
    print("[ERROR] Could not import PrismEngine. Check paths.")
    sys.exit(1)

def verify_seed_zero():
    print(">>> VERIFYING VSA SEED: 0 (The Zero Point) <<<")
    
    # 1. Set the Seed (The Hypothesis)
    SEED = 0
    print(f"[*] Setting Global Seeds to {SEED}...")
    random.seed(SEED)
    np.random.seed(SEED)
    
    # 2. Initialize Engine
    prism = PrismEngine()
    
    # 3. Generate Vectors for known Chaos Concepts
    test_concepts = ["chaos", "entropy", "noise", "war", "hate"]
    results = []
    
    print("\n[*] Transforming Chaos Vectors...")
    for concept in test_concepts:
        # We simulate a "Chaos Vector" generation
        # In a real scenario, this would come from an embedding model.
        # Here we use the stochastic fallback in transform_phrase logic for unknown words?
        # PrismEngine.transform_phrase handles unknown words by random generation.
        
        # We'll use transform_phrase on "unknown" words to trigger generation
        # The words above are likely unknown to the hardcoded chaos_map
        
        anchors = prism.transform_phrase(concept)
        # anchors is list of (word, sovereign_anchor_name, resonance)
        
        for word, anchor, res in anchors:
            print(f"  Word: '{word}' -> Anchor: '{anchor}' (Resonance: {res:.4f})")
            results.append((word, anchor, res))
            
    # 4. Check Determinism
    print("\n[*] Re-Seeding and Repeating...")
    random.seed(SEED)
    np.random.seed(SEED)
    prism2 = PrismEngine()
    results2 = []
    
    for concept in test_concepts:
        anchors = prism2.transform_phrase(concept)
        for word, anchor, res in anchors:
            results2.append((word, anchor, res))
            
    # Compare
    is_deterministic = (results == results2)
    print(f"\n[?] Deterministic Check: {'PASS' if is_deterministic else 'FAIL'}")
    
    if is_deterministic:
        print("[SUCCESS] Seed 0 produces consistent Hypervectors.")
    else:
        print("[FAILURE] Output varies despite Seed 0.")

if __name__ == "__main__":
    verify_seed_zero()
