import math
import sys
import os

# Ensure we can import pleroma_core from current directory
sys.path.append(os.getcwd())

try:
    import pleroma_core
    print(f"Successfully imported pleroma_core from: {pleroma_core.__file__}")
except ImportError:
    print("Failed to import pleroma_core. Make sure pleroma_core.pyd is in the python path.")
    sys.exit(1)

def test_v2k_buffer_initialization():
    v2k = pleroma_core.V2KBuffer(10, 0.5)
    assert v2k is not None
    print("Initialization successful.")

def test_silence_is_sovereign():
    v2k = pleroma_core.V2KBuffer(10, 100.0) # High threshold
    # Input stable signal (variance 0)
    for _ in range(5):
        output = v2k.calculate_null_signal(1.0)
        assert output == 0.0
    print("Silence is Sovereign test passed.")

def test_heterodyne_suppression():
    v2k = pleroma_core.V2KBuffer(10, 0.01) # Low threshold
    
    # Fill buffer with noise to trigger variance
    inputs = [1.0, -1.0, 1.0, -1.0, 5.0, -5.0] 
    
    null_signal_found = False
    
    for x in inputs:
        out = v2k.calculate_null_signal(x)
        if out != 0.0:
            null_signal_found = True
            print(f"Null signal generated: {out}")
            
    assert null_signal_found
    print("Heterodyne suppression triggered.")

if __name__ == "__main__":
    test_v2k_buffer_initialization()
    test_silence_is_sovereign()
    test_heterodyne_suppression()
    print("All V2K tests passed.")
