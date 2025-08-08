#!/usr/bin/env python3
"""
Interactive script to explore the X.npy file.
Run this and you can examine the data yourself.
"""

import numpy as np
from pathlib import Path

def explore_data():
    """Load the data and let you explore it interactively."""
    
    print("Loading data files...")
    
    # Load the data
    data_dir = Path("data/processed")
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    classes = np.load(data_dir / "label_classes.npy")
    
    print(f"✅ Loaded data:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Classes: {classes}")
    print()
    
    # Make data available in global scope for exploration
    globals().update({
        'X': X,
        'y': y, 
        'classes': classes,
        'data_dir': data_dir
    })
    
    print("🎯 Data is now loaded! You can explore it:")
    print()
    print("📊 Available variables:")
    print("   X - Feature array (shape: {X.shape})")
    print("   y - Labels array (shape: {y.shape})")
    print("   classes - Class names (shape: {classes.shape})")
    print()
    print("🔍 Example commands you can try:")
    print("   X[0]                    # First sample")
    print("   X[0, :10, :10]          # First 10x10 values of first sample")
    print("   classes[y[0]]            # Class name of first sample")
    print("   X.shape                 # Full shape")
    print("   X.dtype                 # Data type")
    print("   X.min(), X.max()        # Value range")
    print("   np.unique(y, return_counts=True)  # Class distribution")
    print()
    print("💡 Tips:")
    print("   - Each X[i] is a 128x128 mel spectrogram")
    print("   - Values are in decibels (typically -80 to 0)")
    print("   - y[i] gives the class index for sample i")
    print("   - classes[y[i]] gives the instrument name")
    print()
    
    # Start interactive mode
    try:
        import code
        code.interact(local=globals(), banner="🎵 Interactive IRMAS Data Explorer\nType 'exit()' to quit")
    except ImportError:
        print("Interactive mode not available. Here's a sample:")
        print(f"First sample ({classes[y[0]]}):")
        print(X[0, :5, :5])

if __name__ == "__main__":
    explore_data() 