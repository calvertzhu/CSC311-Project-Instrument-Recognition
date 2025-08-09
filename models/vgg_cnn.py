#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

global device
print("Cuda available: ", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

class IRMASBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers=2, use_batch_norm=True):
        super(IRMASBlock, self).__init__()
        
        layers = []
        for i in range(num_conv_layers):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            layers.append(nn.ReLU(inplace=True))
        
        self.conv_block = nn.Sequential(*layers)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv_block(x)
        x = self.maxpool(x)
        return x

class IRMASNN(nn.Module):
    """
    IRMAS Neural Network for multi-label music instrument classification.
    
    Architecture:
    - 3 blocks: 32 -> 64 -> 128 channels
    - 6 conv layers total (2 per block)
    - Global average pooling
    - Single linear classifier
    - Multi-label output with sigmoid
    """
    
    def __init__(self, 
                 input_channels=1,
                 num_classes=11,
                 use_batch_norm=True,
                 dropout_rate=0.5):
        """
        Initialize VGG CNN model.
        
        Args:
            input_channels: Number of input channels (1 for mel spectrograms)
            num_classes: Number of instrument classes (11 for IRMAS)
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
        """
        super(IRMASNN, self).__init__()
        
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        
        # IRMASNN configuration - 6 layers, 3 blocks
        channels = [32, 64, 128]               # 3 blocks
        conv_layers = [2, 2, 2]                # 6 conv layers total (2 per block)
        
        # IRMAS Blocks
        self.irmas_blocks = nn.ModuleList()
        in_channels = input_channels
        
        for i, (out_channels, num_conv) in enumerate(zip(channels, conv_layers)):
            block = IRMASBlock(in_channels, out_channels, num_conv, use_batch_norm)
            self.irmas_blocks.append(block)
            in_channels = out_channels
        
        # Calculate feature map size after IRMAS blocks
        # Input: (batch, 1, 128, 128)
        # After 3 blocks with pool_size=2: (batch, 128, 16, 16)
        self.feature_size = 128 // (2 ** len(channels))  # 16
        self.feature_channels = channels[-1]  # 128
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Simplified classifier - single layer for ultra-fast training
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_channels, num_classes),
            nn.Sigmoid()  # Multi-label classification
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through IRMAS NN.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 128, 128)
            
        Returns:
            Output tensor of shape (batch_size, num_classes) with sigmoid activations
        """
        # IRMAS feature extraction
        for irmas_block in self.irmas_blocks:
            x = irmas_block(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classification
        output = self.classifier(x)
        
        return output
    
    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of feature maps at different layers
        """
        feature_maps = {}
        
        for i, irmas_block in enumerate(self.irmas_blocks):
            x = irmas_block(x)
            feature_maps[f'irmas_block_{i+1}'] = x.clone()
        
        return feature_maps

class IRMASConfig:
    """Configuration class for IRMAS NN model."""
    
    def __init__(self):
        self.input_channels = 1
        self.num_classes = 11
        self.use_batch_norm = True
        self.dropout_rate = 0.5
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 50
        self.weight_decay = 1e-4
        
    def get_model(self):
        """Create IRMAS NN model with current configuration."""
        return IRMASNN(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate
        )

def create_irmas_model(config=None):
    """
    Factory function to create IRMAS NN model.
    
    Args:
        config: IRMASConfig object or None for default config
        
    Returns:
        IRMASNN model instance
    """
    if config is None:
        config = IRMASConfig()
    
    return config.get_model()

# Example usage and testing
if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"Testing IRMAS NN")
    print(f"{'='*50}")
    
    # Create model
    config = IRMASConfig()
    model = create_irmas_model(config)
    if torch.cuda.is_available():
        model = model.to(device)
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 128)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test feature maps
    feature_maps = model.get_feature_maps(x)
    for name, feature_map in feature_maps.items():
        print(f"{name}: {feature_map.shape}")
    
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}] (should be [0,1] for sigmoid)") 