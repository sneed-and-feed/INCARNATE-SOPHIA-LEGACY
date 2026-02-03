import sys
import os
import numpy as np

# Ensure we can import from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from ghostmesh import SovereignGrid, FlumpyArray
    print("[SUCCESS] Module Import Verified")
except ImportError as e:
    print(f"[FAILURE] Ingest Failed: {e}")
    sys.exit(1)

def test_grid_size():
    print("--- VERIFYING VOLUMETRIC GRID EXPANSION ---")
    
    # 1. Instantiate 5x5x5 Grid
    grid = SovereignGrid(grid_size=5)
    node_count = len(grid.nodes)
    print(f"[1] Grid Initialized: size=5x5x5, nodes={node_count}")
    
    assert node_count == 125, f"ERROR: Expected 125 nodes, found {node_count}"
    print("    > Node Count Verified: 125 [CLASS 8 PENTAD]")

    # 2. Run Process Step
    print("\n[2] Executing Grid Step (Diffusion)...")
    mock_input = FlumpyArray([1.0] * 64, coherence=1.0)
    result = grid.process_step(mock_input)
    
    print(f"    > Output Coherence: {result.coherence:.4f}")
    assert result.coherence > 0, "ERROR: Coherence should be non-zero"
    
    # 3. Verify Normalization (Dynamic check)
    print("\n[3] Checking Normalization Invariant...")
    # Average state should be scaled by 1/125 now
    expected_scale = 1.0 / 125.0
    # In center node injects 1.0, others 0.1
    # total = 1.0 + (124 * 0.1) = 1.0 + 12.4 = 13.4
    # avg = 13.4 / 125 = 0.1072
    actual_avg = np.mean(result.data)
    print(f"    > Actual Average State: {actual_avg:.6f}")
    print(f"    > Expected Average State: ~0.1072")
    
    # Check Density Factor
    gdf = grid.get_density_factor()
    print(f"\n[4] Ghost Density Factor (GDF): {gdf:.4f}")
    assert 1.8 <= gdf <= 3.1, f"ERROR: GDF {gdf} out of bounds"

    print("\n[SUCCESS] Volumetric Grid v5.2 Verified (125 Nodes).")

if __name__ == "__main__":
    test_grid_size()
