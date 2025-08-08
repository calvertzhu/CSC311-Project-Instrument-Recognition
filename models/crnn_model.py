#!/usr/bin/env python3
"""
CRNN (Convolutional Recurrent Neural Network) for Music Instrument Recognition
Combines CNN for spatial feature extraction with RNN for temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for multi-label music instrument classification.
    
    Architecture:
    1. CNN layers: Extract spatial features from mel spectrograms
    2. RNN layers: Model temporal dependencies
    3. Attention mechanism: Focus on important time steps
    4. Classification layers: Multi-label output
    """
    
    def __init__(self, 
                 input_channels=1,
                 num_classes=11,
                 cnn_channels=[32, 64, 128, 256],
                 cnn_kernel_sizes=[3, 3, 3, 3],
                 cnn_pool_sizes=[2, 2, 2, 2],
                 rnn_hidden_size=256,
                 rnn_num_layers=2,
                 dropout_rate=0.3,
                 use_attention=True):
        """
        Initialize CRNN model.
        
        Args:
            input_channels: Number of input channels (1 for mel spectrograms)
            num_classes: Number of instrument classes (11 for IRMAS)
            cnn_channels: List of channel sizes for CNN layers
            cnn_kernel_sizes: List of kernel sizes for CNN layers
            cnn_pool_sizes: List of pool sizes for CNN layers
            rnn_hidden_size: Hidden size for RNN layers
            rnn_num_layers: Number of RNN layers
            dropout_rate: Dropout rate for regularization
            use_attention: Whether to use attention mechanism
        """
        super(CRNN, self).__init__()
        
        self.num_classes = num_classes
        self.rnn_hidden_size = rnn_hidden_size
        self.use_attention = use_attention
        
        # CNN Feature Extraction Layers
        self.cnn_layers = nn.ModuleList()
        in_channels = input_channels
        
        for i, (out_channels, kernel_size, pool_size) in enumerate(zip(cnn_channels, cnn_kernel_sizes, cnn_pool_sizes)):
            # Convolutional block
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
                nn.Dropout2d(dropout_rate * 0.5)  # Less dropout for CNN
            )
            self.cnn_layers.append(conv_block)
            in_channels = out_channels
        
        # Calculate CNN output size
        # Input: (batch, 1, 128, 128)
        # After 4 CNN layers with pool_size=2: (batch, 256, 8, 8)
        self.cnn_output_height = 128 // (2 ** len(cnn_pool_sizes))  # 8
        self.cnn_output_width = 128 // (2 ** len(cnn_pool_sizes))   # 8
        self.cnn_output_channels = cnn_channels[-1]  # 256
        
        # RNN layers for temporal modeling
        self.rnn_input_size = self.cnn_output_channels * self.cnn_output_height  # 256 * 8 = 2048
        self.lstm = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=dropout_rate if rnn_num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=rnn_hidden_size * 2,  # Bidirectional
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(rnn_hidden_size * 2)
        
        # Classification layers
        classifier_input_size = rnn_hidden_size * 2  # Bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, rnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(rnn_hidden_size, rnn_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(rnn_hidden_size // 2, num_classes),
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
        Forward pass through CRNN.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 128, 128)
            
        Returns:
            Output tensor of shape (batch_size, num_classes) with sigmoid activations
        """
        batch_size = x.size(0)
        
        # CNN Feature Extraction
        # x: (batch, 1, 128, 128)
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        # x: (batch, 256, 8, 8)
        
        # Reshape for RNN: (batch, time_steps, features)
        # Transpose to get time steps as sequence
        x = x.permute(0, 3, 1, 2)  # (batch, 8, 256, 8)
        x = x.reshape(batch_size, self.cnn_output_width, -1)  # (batch, 8, 2048)
        
        # RNN Processing
        # x: (batch, 8, 2048)
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (batch, 8, 512) - bidirectional
        
        # Attention mechanism
        if self.use_attention:
            # Self-attention on time steps
            attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.attention_norm(lstm_out + attn_out)  # Residual connection
        
        # Global average pooling over time dimension
        # Use mean pooling to get fixed-size representation
        pooled = torch.mean(lstm_out, dim=1)  # (batch, 512)
        
        # Classification
        output = self.classifier(pooled)  # (batch, num_classes)
        
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
        
        # CNN features
        for i, cnn_layer in enumerate(self.cnn_layers):
            x = cnn_layer(x)
            feature_maps[f'cnn_layer_{i}'] = x.clone()
        
        return feature_maps

class CRNNConfig:
    """Configuration class for CRNN model."""
    
    def __init__(self):
        self.input_channels = 1
        self.num_classes = 11
        self.cnn_channels = [32, 64, 128, 256]
        self.cnn_kernel_sizes = [3, 3, 3, 3]
        self.cnn_pool_sizes = [2, 2, 2, 2]
        self.rnn_hidden_size = 256
        self.rnn_num_layers = 2
        self.dropout_rate = 0.3
        self.use_attention = True
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 50
        self.weight_decay = 1e-4
        
    def get_model(self):
        """Create CRNN model with current configuration."""
        return CRNN(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            cnn_channels=self.cnn_channels,
            cnn_kernel_sizes=self.cnn_kernel_sizes,
            cnn_pool_sizes=self.cnn_pool_sizes,
            rnn_hidden_size=self.rnn_hidden_size,
            rnn_num_layers=self.rnn_num_layers,
            dropout_rate=self.dropout_rate,
            use_attention=self.use_attention
        )

def create_crnn_model(config=None):
    """
    Factory function to create CRNN model.
    
    Args:
        config: CRNNConfig object or None for default config
        
    Returns:
        CRNN model instance
    """
    if config is None:
        config = CRNNConfig()
    
    return config.get_model()

# Example usage and testing
if __name__ == "__main__":
    # Test the model
    model = create_crnn_model()
    
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