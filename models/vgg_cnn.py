#!/usr/bin/env python3
"""
VGG-style CNN for Music Instrument Recognition
Based on VGG architecture adapted for mel spectrogram input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VGGBlock(nn.Module):
    """
    VGG Block: Multiple convolutional layers followed by max pooling.
    """
    def __init__(self, in_channels, out_channels, num_conv_layers=2, use_batch_norm=True):
        super(VGGBlock, self).__init__()
        
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

class VGGCNN(nn.Module):
    """
    VGG-style CNN for multi-label music instrument classification.
    
    Architecture:
    - VGG blocks with increasing channel sizes
    - Global average pooling
    - Dense classification layers
    - Multi-label output with sigmoid
    """
    
    def __init__(self, 
                 input_channels=1,
                 num_classes=11,
                 vgg_config='A',  # 'A', 'B', 'C', 'D', 'E'
                 use_batch_norm=True,
                 dropout_rate=0.5):
        """
        Initialize VGG CNN model.
        
        Args:
            input_channels: Number of input channels (1 for mel spectrograms)
            num_classes: Number of instrument classes (11 for IRMAS)
            vgg_config: VGG configuration ('A', 'B', 'C', 'D', 'E')
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
        """
        super(VGGCNN, self).__init__()
        
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        
        # VGG configurations (channels per block)
        vgg_configs = {
            'A': [64, 128, 256, 512, 512],      # VGG-11
            'B': [64, 128, 256, 512, 512],      # VGG-13
            'C': [64, 128, 256, 512, 512],      # VGG-16
            'D': [64, 128, 256, 512, 512],      # VGG-16
            'E': [64, 128, 256, 512, 512]       # VGG-19
        }
        
        # Number of conv layers per block
        conv_layers_per_block = {
            'A': [1, 1, 2, 2, 2],  # VGG-11
            'B': [2, 2, 2, 2, 2],  # VGG-13
            'C': [2, 2, 3, 3, 3],  # VGG-16
            'D': [2, 2, 3, 3, 3],  # VGG-16
            'E': [2, 2, 4, 4, 4]   # VGG-19
        }
        
        channels = vgg_configs[vgg_config]
        conv_layers = conv_layers_per_block[vgg_config]
        
        # VGG Blocks
        self.vgg_blocks = nn.ModuleList()
        in_channels = input_channels
        
        for i, (out_channels, num_conv) in enumerate(zip(channels, conv_layers)):
            block = VGGBlock(in_channels, out_channels, num_conv, use_batch_norm)
            self.vgg_blocks.append(block)
            in_channels = out_channels
        
        # Calculate feature map size after VGG blocks
        # Input: (batch, 1, 128, 128)
        # After 5 blocks with pool_size=2: (batch, 512, 4, 4)
        self.feature_size = 128 // (2 ** len(channels))  # 4
        self.feature_channels = channels[-1]  # 512
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
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
        Forward pass through VGG CNN.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 128, 128)
            
        Returns:
            Output tensor of shape (batch_size, num_classes) with sigmoid activations
        """
        # VGG feature extraction
        for vgg_block in self.vgg_blocks:
            x = vgg_block(x)
        
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
        
        for i, vgg_block in enumerate(self.vgg_blocks):
            x = vgg_block(x)
            feature_maps[f'vgg_block_{i+1}'] = x.clone()
        
        return feature_maps

class VGGConfig:
    """Configuration class for VGG CNN model."""
    
    def __init__(self):
        self.input_channels = 1
        self.num_classes = 11
        self.vgg_config = 'C'  # VGG-16 style
        self.use_batch_norm = True
        self.dropout_rate = 0.5
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 50
        self.weight_decay = 1e-4
        
    def get_model(self):
        """Create VGG CNN model with current configuration."""
        return VGGCNN(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            vgg_config=self.vgg_config,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate
        )

def create_vgg_model(config=None):
    """
    Factory function to create VGG CNN model.
    
    Args:
        config: VGGConfig object or None for default config
        
    Returns:
        VGGCNN model instance
    """
    if config is None:
        config = VGGConfig()
    
    return config.get_model()

# Example usage and testing
if __name__ == "__main__":
    # Test different VGG configurations
    configs = ['A', 'B', 'C', 'D', 'E']
    
    for config_name in configs:
        print(f"\n{'='*50}")
        print(f"Testing VGG-{config_name}")
        print(f"{'='*50}")
        
        # Create model
        config = VGGConfig()
        config.vgg_config = config_name
        model = create_vgg_model(config)
        
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