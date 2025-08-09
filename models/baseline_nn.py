#!/usr/bin/env python3
"""
Simple baseline neural network for IRMAS multi-label classification.
Fast training for testing the pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineNN(nn.Module):
    """
    Simple 3-layer fully connected network for multi-label instrument classification.
    
    Architecture:
    - Flatten: 128×128 → 16384
    - Linear: 16384 → 128 + ReLU
    - Linear: 128 → 32 + ReLU  
    - Linear: 32 → 11 + Sigmoid (multi-label)
    """
    
    def __init__(self, input_size=128*128, hidden1=128, hidden2=32, num_classes=11, dropout_rate=0.3):
        super(BaselineNN, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Network layers
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(in_features=hidden1, out_features=hidden2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear3 = nn.Linear(in_features=hidden2, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x_raw):
        """
        Forward pass through baseline network.
        
        Args:
            x_raw: Input tensor of shape (batch_size, 1, 128, 128) or (batch_size, 128, 128)
            
        Returns:
            Output tensor of shape (batch_size, num_classes) with sigmoid activations
        """
        # Handle different input shapes
        if x_raw.dim() == 4:  # (batch, 1, 128, 128)
            x = x_raw.view(-1, self.input_size)
        elif x_raw.dim() == 3:  # (batch, 128, 128)
            x = x_raw.view(-1, self.input_size)
        else:
            x = x_raw
        
        # Forward pass
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)
        x = self.linear3(x)
        y = self.sigmoid(x)  # Multi-label output
        
        return y
    
    def get_model_info(self):
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024
        }

def create_baseline_model(hidden1=128, hidden2=32, dropout_rate=0.3):
    """
    Factory function to create baseline neural network.
    
    Args:
        hidden1: Size of first hidden layer
        hidden2: Size of second hidden layer
        dropout_rate: Dropout rate for regularization
        
    Returns:
        BaselineNN model instance
    """
    return BaselineNN(
        input_size=128*128,
        hidden1=hidden1,
        hidden2=hidden2,
        num_classes=11,
        dropout_rate=dropout_rate
    )

# Test the model
if __name__ == "__main__":
    print("Testing Baseline Neural Network...")
    
    # Create model
    model = create_baseline_model()
    info = model.get_model_info()
    
    print(f"Model: BaselineNN")
    print(f"Parameters: {info['total_params']:,}")
    print(f"Model size: {info['model_size_mb']:.2f} MB")
    
    # Test with different input shapes
    batch_size = 4
    
    # Test with 4D input (batch, channel, height, width)
    x1 = torch.randn(batch_size, 1, 128, 128)
    output1 = model(x1)
    print(f"Input 4D: {x1.shape} → Output: {output1.shape}")
    
    # Test with 3D input (batch, height, width)
    x2 = torch.randn(batch_size, 128, 128)
    output2 = model(x2)
    print(f"Input 3D: {x2.shape} → Output: {output2.shape}")
    
    print(f"Output range: [{output1.min():.3f}, {output1.max():.3f}] (should be [0,1] for sigmoid)")
    print("SUCCESS: Baseline model working correctly!") 