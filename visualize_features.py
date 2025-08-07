#!/usr/bin/env python3
"""
Visualize the mel spectrogram features from the IRMAS dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_features():
    """Visualize sample mel spectrograms from the dataset."""
    
    # Load the data
    data_dir = Path("data/processed")
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    classes = np.load(data_dir / "label_classes.npy")
    
    print(f"Loaded {len(X)} samples with {len(classes)} instrument classes")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('IRMAS Dataset: Mel Spectrograms of Different Instruments', fontsize=16)
    
    # Show one example from each instrument class
    for i, instrument in enumerate(classes):
        if i >= 12:  # Only show first 12 instruments
            break
            
        # Find first sample of this instrument
        instrument_samples = np.where(y == i)[0]
        if len(instrument_samples) > 0:
            sample_idx = instrument_samples[0]
            
            # Get the mel spectrogram
            mel_spec = X[sample_idx]
            
            # Plot
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            
            # Create the spectrogram plot
            im = ax.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f'{instrument}', fontsize=12)
            ax.set_xlabel('Time Frames')
            ax.set_ylabel('Mel Frequency Bands')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='dB')
    
    # Hide unused subplots
    for i in range(len(classes), 12):
        row = i // 4
        col = i % 4
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Also show detailed view of one sample
    print(f"\nDetailed view of first sample ({classes[y[0]]}):")
    print(f"Shape: {X[0].shape}")
    print(f"Value range: [{X[0].min():.2f}, {X[0].max():.2f}]")
    print(f"Mean value: {X[0].mean():.2f}")
    
    # Show the actual numerical values (small section)
    print(f"\nFirst 10x10 values of the mel spectrogram:")
    print(X[0, :10, :10])
    
    # Show what different instruments look like
    print(f"\nSample instruments and their shapes:")
    for i, instrument in enumerate(classes[:5]):  # Show first 5
        instrument_samples = np.where(y == i)[0]
        if len(instrument_samples) > 0:
            sample_idx = instrument_samples[0]
            print(f"  {instrument}: shape {X[sample_idx].shape}, range [{X[sample_idx].min():.1f}, {X[sample_idx].max():.1f}]")

if __name__ == "__main__":
    visualize_features() 