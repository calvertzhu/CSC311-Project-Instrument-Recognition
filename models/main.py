#!/usr/bin/env python3
"""
Enhanced main training script for IRMAS music instrument recognition.
Clean, robust pipeline with comprehensive progress tracking and error handling.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import json
import time
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from models.vgg_cnn import create_vgg_model, VGGConfig
from models.baseline_nn import create_baseline_model
from models.trainer import IRMASTrainer
from models.model_data_loader import create_data_loaders

class IRMASTrainingPipeline:
    """Complete training pipeline for IRMAS dataset."""
    
    def __init__(self, args):
        self.args = args
        self.model = None
        self.trainer = None
        self.model_name = None
        self.start_time = None
        
    def verify_data_files(self):
        """Check if all required data files exist."""
        print("Verifying data files...")
        
        required_files = [
            "data/processed/X_train.npy",
            "data/processed/y_train.npy", 
            "data/processed/X_val.npy",
            "data/processed/y_val.npy",
            "data/test_processed/X_test.npy",
            "data/test_processed/y_test.npy"
        ]
        
        missing_files = [f for f in required_files if not Path(f).exists()]
        
        if missing_files:
            print("ERROR: Missing data files:")
            for f in missing_files:
                print(f"   - {f}")
            print("\nPlease run data processing first:")
            print("   python3 data/feature_extractor.py")
            print("   python3 data/split_dataset.py") 
            print("   python3 data/test_feature_extractor.py")
            return False
        
        print("SUCCESS: All data files found!")
        return True
    
    def display_dataset_info(self):
        """Display dataset information."""
        print("\nDataset Information:")
        
        try:
            # Load and display dataset stats
            X_train = np.load("data/processed/X_train.npy")
            y_train = np.load("data/processed/y_train.npy")
            X_val = np.load("data/processed/X_val.npy") 
            y_val = np.load("data/processed/y_val.npy")
            X_test = np.load("data/test_processed/X_test.npy")
            y_test = np.load("data/test_processed/y_test.npy")
            
            print(f"   Training:   {X_train.shape[0]:,} samples, shape {X_train.shape}")
            print(f"   Validation: {X_val.shape[0]:,} samples, shape {X_val.shape}")
            print(f"   Test:       {X_test.shape[0]:,} samples, shape {X_test.shape}")
            print(f"   Total:      {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]:,} samples")
            
        except Exception as e:
            print(f"ERROR: Error loading dataset info: {e}")
            return False
        
        return True
    
    def create_model(self):
        """Create and configure the model."""
        print(f"\nCreating {self.args.model.upper()} model...")
        
        try:
            if self.args.model == 'vgg':
                config = VGGConfig()
                config.vgg_config = self.args.config
                config.batch_size = self.args.batch_size
                config.learning_rate = self.args.lr
                config.num_epochs = self.args.epochs
                config.dropout_rate = self.args.dropout
                
                self.model = create_vgg_model(config)
                self.model_name = f"vgg_{self.args.config}_irmas"
                
            elif self.args.model == 'baseline':
                self.model = create_baseline_model(
                    hidden1=128,
                    hidden2=32,
                    dropout_rate=self.args.dropout
                )
                self.model_name = "baseline_nn_irmas"
                
            else:
                raise ValueError(f"Model '{self.args.model}' not supported")
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"SUCCESS: Model created successfully!")
            if self.args.model == 'vgg':
                print(f"   - Architecture: VGG-{self.args.config}")
            else:
                print(f"   - Architecture: {self.args.model.upper()}")
            print(f"   - Total parameters: {total_params:,}")
            print(f"   - Trainable parameters: {trainable_params:,}")
            print(f"   - Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Model creation failed: {e}")
            return False
    
    def setup_trainer(self):
        """Set up the trainer with data loaders."""
        print(f"\nSetting up trainer...")
        
        try:
            self.trainer = IRMASTrainer(
                model=self.model,
                model_name=self.model_name,
                batch_size=self.args.batch_size,
                learning_rate=self.args.lr,
                weight_decay=self.args.weight_decay
            )
            
            print(f"SUCCESS: Trainer configured successfully!")
            print(f"   - Training batches: {len(self.trainer.train_loader)}")
            print(f"   - Validation batches: {len(self.trainer.val_loader)}")
            print(f"   - Test batches: {len(self.trainer.test_loader)}")
            print(f"   - Optimizer: Adam (lr={self.args.lr}, wd={self.args.weight_decay})")
            print(f"   - Loss function: BCELoss (multi-label)")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Trainer setup failed: {e}")
            return False
    
    def train_model(self):
        """Train the model with progress tracking."""
        print(f"\nStarting training...")
        print(f"   - Epochs: {self.args.epochs}")
        print(f"   - Batch size: {self.args.batch_size}")
        print(f"   - Learning rate: {self.args.lr}")
        print(f"   - Device: {next(self.model.parameters()).device}")
        print(f"   - SpecAugment: Enabled (training only)")
        
        self.start_time = time.time()
        
        try:
            # Train the model
            history = self.trainer.train(self.args.epochs, save_best=True)
            
            training_time = time.time() - self.start_time
            
            print(f"\nSUCCESS: Training completed!")
            print(f"   - Total time: {training_time/60:.1f} minutes")
            print(f"   - Best validation loss: {min(self.trainer.val_losses):.4f}")
            print(f"   - Final training loss: {self.trainer.train_losses[-1]:.4f}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Training failed: {e}")
            return False
    
    def evaluate_model(self):
        """Evaluate the model on test data."""
        if self.args.no_test:
            print("\nSkipping test evaluation")
            return True
        
        print(f"\nEvaluating on test data...")
        
        try:
            test_results = self.trainer.evaluate_test()
            
            print(f"SUCCESS: Test evaluation completed!")
            print(f"   - Test Loss: {test_results['test_loss']:.4f}")
            print(f"   - Test Accuracy: {test_results['test_accuracy']:.4f}")
            print(f"   - F1-Micro: {test_results.get('f1_micro', 'N/A'):.4f}")
            print(f"   - F1-Macro: {test_results.get('f1_macro', 'N/A'):.4f}")
            
            # Display per-instrument results
            if 'instrument_metrics' in test_results:
                print(f"\nPer-Instrument Performance:")
                instruments = ['acoustic guitar', 'cello', 'clarinet', 'electric guitar', 
                              'flute', 'organ', 'piano', 'saxophone', 'trumpet', 'violin', 'voice']
                
                print(f"{'Instrument':<15} {'F1':>6} {'Precision':>9} {'Recall':>8} {'Support':>8}")
                print("-" * 55)
                
                for instrument in instruments:
                    if instrument in test_results['instrument_metrics']:
                        metrics = test_results['instrument_metrics'][instrument]
                        print(f"{instrument:<15} {metrics['f1']:>6.3f} {metrics['precision']:>9.3f} "
                              f"{metrics['recall']:>8.3f} {metrics.get('support', 0):>8}")
            
            return test_results
            
        except Exception as e:
            print(f"ERROR: Test evaluation failed: {e}")
            return None
    
    def save_results(self, test_results=None):
        """Save training results and history."""
        if not self.args.save_results:
            return
        
        print(f"\nSaving results...")
        
        try:
            results_dir = Path("models/results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"{self.model_name}_{timestamp}_results.json"
            
            training_time = time.time() - self.start_time if self.start_time else 0
            
            results = {
                'model_name': self.model_name,
                'training_config': {
                    'model': self.args.model,
                    'config': getattr(self.args, 'config', 'N/A'),
                    'epochs': self.args.epochs,
                    'batch_size': self.args.batch_size,
                    'learning_rate': self.args.lr,
                    'weight_decay': self.args.weight_decay,
                    'dropout': self.args.dropout
                },
                'training_time_seconds': training_time,
                'training_time_minutes': training_time / 60,
                'training_history': {
                    'train_losses': [float(x) for x in self.trainer.train_losses],
                    'val_losses': [float(x) for x in self.trainer.val_losses],
                    'train_accuracies': [float(x) for x in self.trainer.train_accuracies],
                    'val_accuracies': [float(x) for x in self.trainer.val_accuracies]
                }
            }
            
            if test_results:
                results['test_results'] = test_results
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"SUCCESS: Results saved to: {results_file}")
            
        except Exception as e:
            print(f"ERROR: Failed to save results: {e}")
    
    def plot_training_history(self):
        """Plot and save training history."""
        print(f"\nGenerating training plots...")
        
        try:
            self.trainer.plot_training_history()
            print(f"SUCCESS: Training plots generated!")
            
        except Exception as e:
            print(f"ERROR: Failed to generate plots: {e}")
    
    def run(self):
        """Run the complete training pipeline."""
        print("=" * 70)
        print("IRMAS MUSIC INSTRUMENT RECOGNITION TRAINING")
        print("=" * 70)
        print(f"Configuration:")
        print(f"   - Model: {self.args.model.upper()}")
        if hasattr(self.args, 'config'):
            print(f"   - VGG Config: {self.args.config}")
        print(f"   - Epochs: {self.args.epochs}")
        print(f"   - Batch Size: {self.args.batch_size}")
        print(f"   - Learning Rate: {self.args.lr}")
        print(f"   - Weight Decay: {self.args.weight_decay}")
        print(f"   - Dropout: {self.args.dropout}")
        print(f"   - Test Evaluation: {'No' if self.args.no_test else 'Yes'}")
        print(f"   - Save Results: {'Yes' if self.args.save_results else 'No'}")
        print("=" * 70)
        
        # Run pipeline steps
        if not self.verify_data_files():
            return False
        
        if not self.display_dataset_info():
            return False
        
        if not self.create_model():
            return False
        
        if not self.setup_trainer():
            return False
        
        if not self.train_model():
            return False
        
        test_results = self.evaluate_model()
        
        self.save_results(test_results)
        self.plot_training_history()
        
        print(f"\nSUCCESS: TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"   - Best model: models/saved_models/{self.model_name}_best.pth")
        print(f"   - Final model: models/saved_models/{self.model_name}_final.pth")
        print("=" * 70)
        
        return True

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Train IRMAS music instrument recognition model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, default='vgg', choices=['vgg', 'baseline'], 
                       help='Model architecture to use')
    parser.add_argument('--config', type=str, default='C', choices=['A', 'B', 'C', 'D', 'E'],
                       help='VGG configuration (A=VGG11, B=VGG13, C=VGG16, D=VGG16, E=VGG19)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='L2 weight decay')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Evaluation arguments
    parser.add_argument('--no-test', action='store_true', help='Skip test evaluation')
    parser.add_argument('--save-results', action='store_true', help='Save detailed results to JSON')
    
    # Quick options
    parser.add_argument('--quick', action='store_true', help='Quick training (5 epochs for testing)')
    
    args = parser.parse_args()
    
    # Quick training mode
    if args.quick:
        args.epochs = 5
        args.batch_size = 16
        print("Quick training mode: 5 epochs, batch size 16")
    
    # Create and run pipeline
    pipeline = IRMASTrainingPipeline(args)
    success = pipeline.run()
    
    if not success:
        print("ERROR: Training pipeline failed!")
        sys.exit(1)

def quick_test():
    """Quick test to verify all components work."""
    print("QUICK COMPONENT TEST")
    print("=" * 40)
    
    try:
        # Test data loading
        print("1. Testing data loading...")
        train_loader, val_loader, test_loader = create_data_loaders(batch_size=4)
        print(f"   SUCCESS: Data loaders created successfully")
        
        # Test model creation
        print("2. Testing model creation...")
        model = create_vgg_model()
        params = sum(p.numel() for p in model.parameters())
        print(f"   SUCCESS: VGG model created ({params:,} parameters)")
        
        # Test forward pass
        print("3. Testing forward pass...")
        batch = next(iter(train_loader))
        data, labels = batch
        with torch.no_grad():
            output = model(data)
        print(f"   SUCCESS: Forward pass successful ({data.shape} -> {output.shape})")
        
        print("\nSUCCESS: All components working correctly!")
        print("\nTo start training:")
        print("   python3 models/main.py                    # Default training")
        print("   python3 models/main.py --quick            # Quick test (5 epochs)")
        print("   python3 models/main.py --config D --epochs 50  # Custom config")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Component test failed: {e}")
        return False

if __name__ == "__main__":
    # If no arguments provided, run quick test
    if len(sys.argv) == 1:
        quick_test()
    else:
        main() 