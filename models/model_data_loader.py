#!/usr/bin/env python3
"""
Comprehensive data loader for IRMAS dataset.
Handles both training (single-label) and test (multi-label) data.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config import LABEL_MAP

class IRMASDataset(Dataset):
    """
    IRMAS dataset loader that handles both training and test data.
    Converts single-label training data to multi-label format.
    """
    
    def __init__(self, data_file, label_file, is_multi_label=False):
        """
        Initialize dataset.
        
        Args:
            data_file: Path to feature file (.npy)
            label_file: Path to label file (.npy)
            is_multi_label: Whether the data is already multi-label
        """
        self.data = torch.from_numpy(np.load(data_file)).float()
        self.labels = np.load(label_file)
        self.is_multi_label = is_multi_label
        
        # Convert single-label to multi-label if needed
        if not is_multi_label:
            self.labels = self._convert_to_multi_label(self.labels)
        
        # Convert to tensor
        self.labels = torch.from_numpy(self.labels).float()
        
        print(f"Loaded dataset: {len(self.data)} samples")
        print(f"Data shape: {self.data.shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Multi-label: {self.is_multi_label}")
    
    def _convert_to_multi_label(self, single_labels):
        """
        Convert single-label indices to multi-label binary vectors.
        
        Args:
            single_labels: Array of single-label indices
            
        Returns:
            Multi-label binary matrix
        """
        num_samples = len(single_labels)
        num_classes = 11  # Number of instrument classes
        
        # Create binary matrix
        multi_labels = np.zeros((num_samples, num_classes), dtype=np.float32)
        
        # Set 1 for each sample's class
        for i, label_idx in enumerate(single_labels):
            multi_labels[i, label_idx] = 1.0
        
        return multi_labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get data and ensure correct shape
        data = self.data[idx]
        labels = self.labels[idx]
        
        # Add channel dimension if needed (batch, 128, 128) -> (batch, 1, 128, 128)
        if data.dim() == 2:
            data = data.unsqueeze(0)
        
        return data, labels

def create_data_loaders(batch_size=32, train_val_split=0.2, random_state=42):
    """
    Create training and validation data loaders.
    
    Args:
        batch_size: Batch size for training
        train_val_split: Fraction of data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load training data (single-label, will be converted to multi-label)
    train_dataset = IRMASDataset(
        "data/processed/X_train.npy", 
        "data/processed/y_train.npy",
        is_multi_label=False
    )
    
    # Load validation data (single-label, will be converted to multi-label)
    val_dataset = IRMASDataset(
        "data/processed/X_val.npy", 
        "data/processed/y_val.npy",
        is_multi_label=False
    )
    
    # Load test data (already multi-label)
    test_dataset = IRMASDataset(
        "data/test_processed/X_test.npy", 
        "data/test_processed/y_test.npy",
        is_multi_label=True
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_dataset_info():
    """Get information about all datasets."""
    
    # Training data info
    train_data = np.load("data/processed/X_train.npy")
    train_labels = np.load("data/processed/y_train.npy")
    
    # Validation data info
    val_data = np.load("data/processed/X_val.npy")
    val_labels = np.load("data/processed/y_val.npy")
    
    # Test data info
    test_data = np.load("data/test_processed/X_test.npy")
    test_labels = np.load("data/test_processed/y_test.npy")
    
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    
    print(f"\nTRAINING DATA:")
    print(f"   Samples: {len(train_data)}")
    print(f"   Features: {train_data.shape[1:]} (mel spectrograms)")
    print(f"   Labels: Single-label indices (0-10)")
    print(f"   Classes: {len(np.unique(train_labels))}")
    
    print(f"\nVALIDATION DATA:")
    print(f"   Samples: {len(val_data)}")
    print(f"   Features: {val_data.shape[1:]} (mel spectrograms)")
    print(f"   Labels: Single-label indices (0-10)")
    print(f"   Classes: {len(np.unique(val_labels))}")
    
    print(f"\nTEST DATA:")
    print(f"   Samples: {len(test_data)}")
    print(f"   Features: {test_data.shape[1:]} (mel spectrograms)")
    print(f"   Labels: Multi-label binary vectors (11 classes)")
    print(f"   Multi-label samples: {(test_labels.sum(axis=1) > 1).sum()}")
    print(f"   Single-label samples: {(test_labels.sum(axis=1) == 1).sum()}")
    
    print(f"\nCONVERSION:")
    print(f"   Training/Val: Single-label → Multi-label binary")
    print(f"   Test: Already multi-label (no conversion needed)")
    print(f"   All datasets: Compatible with multi-label models")

if __name__ == "__main__":
    # Test the data loader
    get_dataset_info()
    
    # Test loading
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=4)
    
    print(f"\nData loaders created successfully!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Test a batch
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"\nSample batch {batch_idx + 1}:")
        print(f"   Data shape: {data.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Labels sum: {labels.sum(dim=1)} (should be 1 for training data)")
        break 