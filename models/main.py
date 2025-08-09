#!/usr/bin/env python3
"""
Concise training script for IRMAS music instrument recognition.
"""

import torch
import argparse
import sys
from pathlib import Path
import sys
import argparse
import json
import time
from datetime import datetime
import warnings

global device
print("Cuda available: ", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from models.vgg_cnn import create_irmas_model, IRMASConfig
from models.baseline_nn import create_baseline_model
from models.trainer import IRMASTrainer

def create_model(model_type, args):
    """Create and return the specified model."""
    if model_type == 'irmas':
        config = IRMASConfig()
        config.batch_size = args.batch_size
        config.learning_rate = args.lr
        config.num_epochs = args.epochs
        config.dropout_rate = args.dropout
        model = create_irmas_model(config)
        model_name = "irmas_nn"
    elif model_type == 'baseline':
        model = create_baseline_model(
            hidden1=128,
            hidden2=32,
            dropout_rate=args.dropout
        )
        model_name = "baseline_nn_irmas"
    else:
        raise ValueError(f"Model '{model_type}' not supported")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type.upper()}")
    print(f"Parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f}MB)")
    
    return model, model_name

def train_model(args):
    """Main training function."""
    print("IRMAS TRAINING")
    print("=" * 30)
    
    # Create model
    model, model_name = create_model(args.model, args)
    if torch.cuda.is_available():
        model = model.to(device)
    
    # Setup trainer
    trainer = IRMASTrainer(
        model=model,
        model_name=model_name,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Train
    print(f"Training {model_name} for {args.epochs} epochs...")
    trainer.train(args.epochs)
    
    # Evaluate if requested
    if not args.no_test:
        print("Evaluating on test set...")
        trainer.evaluate_test()
    
    print("Training completed!")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train IRMAS music instrument classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, default='irmas', choices=['irmas', 'baseline'], 
                       help='Model architecture to use')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Evaluation arguments
    parser.add_argument('--no-test', action='store_true', help='Skip test evaluation')
    
    # Quick test mode
    parser.add_argument('--quick', action='store_true', help='Quick test with 5 epochs and batch size 16')
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        args.epochs = 5
        args.batch_size = 16
        print("QUICK MODE: 5 epochs, batch size 16")
    
    # Run training
    train_model(args)

if __name__ == "__main__":
    main() 