#!/usr/bin/env python3
"""
Main training script for IRMAS music instrument recognition.
Complete pipeline without requiring Jupyter notebooks.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from models.vgg_cnn import create_vgg_model, VGGConfig
from models.trainer import IRMASTrainer, train_model
from models.model_data_loader import get_dataset_info

def main():
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train IRMAS music instrument recognition model')
    parser.add_argument('--model', type=str, default='vgg', choices=['vgg', 'crnn'], 
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--config', type=str, default='C', choices=['A', 'B', 'C', 'D', 'E'],
                       help='VGG configuration (only for VGG model)')
    parser.add_argument('--no-test', action='store_true', help='Skip test evaluation')
    parser.add_argument('--save-results', action='store_true', help='Save detailed results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("IRMAS MUSIC INSTRUMENT RECOGNITION TRAINING")
    print("=" * 80)
    print(f"Model: {args.model.upper()}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    if args.model == 'vgg':
        print(f"VGG Config: {args.config}")
    print(f"Test Evaluation: {'No' if args.no_test else 'Yes'}")
    print(f"Save Results: {'Yes' if args.save_results else 'No'}")
    print("=" * 80)
    
    # Check if data exists
    data_files = [
        "data/processed/X_train.npy",
        "data/processed/y_train.npy", 
        "data/processed/X_val.npy",
        "data/processed/y_val.npy",
        "data/test_processed/X_test.npy",
        "data/test_processed/y_test.npy"
    ]
    
    missing_files = [f for f in data_files if not Path(f).exists()]
    if missing_files:
        print("Missing data files:")
        for f in missing_files:
            print(f"   {f}")
        print("\nPlease run the data processing scripts first:")
        print("   python3 data/feature_extractor.py")
        print("   python3 data/split_dataset.py") 
        print("   python3 data/test_feature_extractor.py")
        return
    
    # Display dataset information
    print("\nDATASET INFORMATION:")
    get_dataset_info()
    
    # Create model
    print(f"\nCREATING {args.model.upper()} MODEL:")
    
    if args.model == 'vgg':
        config = VGGConfig()
        config.vgg_config = args.config
        config.batch_size = args.batch_size
        config.learning_rate = args.lr
        config.num_epochs = args.epochs
        
        model = create_vgg_model(config)
        model_name = f"vgg_{args.config}_irmas"
        
    elif args.model == 'crnn':
        from models.crnn_model import create_crnn_model, CRNNConfig
        config = CRNNConfig()
        config.batch_size = args.batch_size
        config.learning_rate = args.lr
        config.num_epochs = args.epochs
        
        model = create_crnn_model(config)
        model_name = "crnn_irmas"
    
    print(f"   Model: {model_name}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Architecture: {type(model).__name__}")
    
    # Create trainer
    print(f"\nSETTING UP TRAINER:")
    trainer = IRMASTrainer(
        model=model,
        model_name=model_name,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    print(f"   Train batches: {len(trainer.train_loader)}")
    print(f"   Val batches: {len(trainer.val_loader)}")
    print(f"   Test batches: {len(trainer.test_loader)}")
    
    # Train model
    print(f"\nSTARTING TRAINING:")
    start_time = datetime.now()
    
    history = trainer.train(args.epochs, save_best=True)
    
    training_time = datetime.now() - start_time
    print(f"\nTraining completed in: {training_time}")
    
    # Evaluate on test data
    if not args.no_test:
        print(f"\nEVALUATING ON TEST DATA:")
        test_results = trainer.evaluate_test()
        
        # Print per-instrument results
        print(f"\nPER-INSTRUMENT PERFORMANCE:")
        instruments = ['acoustic guitar', 'cello', 'clarinet', 'electric guitar', 
                      'flute', 'organ', 'piano', 'saxophone', 'trumpet', 'violin', 'voice']
        
        for i, instrument in enumerate(instruments):
            metrics = test_results['instrument_metrics'][instrument]
            print(f"   {instrument:15s}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
    
    # Save results if requested
    if args.save_results:
        results_dir = Path("models/results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{model_name}_{timestamp}_results.json"
        
        results = {
            'model_name': model_name,
            'training_time': str(training_time),
            'training_history': {
                'train_losses': [float(x) for x in trainer.train_losses],
                'val_losses': [float(x) for x in trainer.val_losses],
                'train_accuracies': [float(x) for x in trainer.train_accuracies],
                'val_accuracies': [float(x) for x in trainer.val_accuracies]
            }
        }
        
        if not args.no_test:
            results['test_results'] = {
                'test_loss': float(test_results['test_loss']),
                'test_accuracy': float(test_results['test_accuracy']),
                'instrument_metrics': test_results['instrument_metrics']
            }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    # Plot training history
    print(f"\nPLOTTING TRAINING HISTORY:")
    trainer.plot_training_history()
    
    print(f"\nTRAINING COMPLETED SUCCESSFULLY!")
    print(f"   Best model saved: models/saved_models/{model_name}_best.pth")
    print(f"   Final model saved: models/saved_models/{model_name}_final.pth")

def quick_test():
    """Quick test to verify everything works."""
    print("QUICK TEST - Verifying all components...")
    
    # Test data loading
    try:
        get_dataset_info()
        print("Data loading: OK")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return False
    
    # Test model creation
    try:
        model = create_vgg_model()
        print(f"Model creation: OK ({sum(p.numel() for p in model.parameters()):,} params)")
    except Exception as e:
        print(f"Model creation failed: {e}")
        return False
    
    # Test trainer creation
    try:
        trainer = IRMASTrainer(model, "test", batch_size=4)
        print("Trainer creation: OK")
    except Exception as e:
        print(f"Trainer creation failed: {e}")
        return False
    
    print("All components working correctly!")
    return True

if __name__ == "__main__":
    # Check if this is a test run
    if len(sys.argv) == 1:
        print("No arguments provided. Running quick test...")
        if quick_test():
            print("\nTo train a model, use:")
            print("   python3 models/main.py --model vgg --epochs 30")
            print("   python3 models/main.py --model crnn --epochs 30")
            print("\nFor help:")
            print("   python3 models/main.py --help")
    else:
        main() 