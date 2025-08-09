#!/usr/bin/env python3
"""
Comprehensive trainer for IRMAS music instrument recognition.
Works with both training (single-label) and test (multi-label) data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from models.model_data_loader import create_data_loaders, get_dataset_info

class IRMASTrainer:
    """
    Trainer for IRMAS music instrument recognition models.
    Handles both training and evaluation on test data.
    """
    
    def __init__(self, model, model_name, batch_size=32, learning_rate=0.001, weight_decay=1e-4, device='auto'):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            model_name: Name for saving models
            batch_size: Training batch size
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model = model
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(batch_size=batch_size)
        
        # Setup training components
        self.criterion = nn.BCELoss()  # Binary Cross Entropy for multi-label
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Create save directory
        self.save_dir = Path("models/saved_models")
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data, labels) in enumerate(self.train_loader):
            # Move data to device
            data, labels = data.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Calculate accuracy (threshold at 0.5)
            predictions = (outputs > 0.5).float()
            correct = (predictions == labels).float().sum()
            total_correct += correct.item()
            total_samples += labels.numel()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, labels in self.val_loader:
                # Move data to device
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = (outputs > 0.5).float()
                correct = (predictions == labels).float().sum()
                total_correct += correct.item()
                total_samples += labels.numel()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def train(self, num_epochs, save_best=True):
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            save_best: Whether to save best model
            
        Returns:
            Training history
        """
        print(f"   Starting training for {self.model_name}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate_epoch()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f"{self.model_name}_best.pth")
                print(f"   Saved best model (val_loss: {val_loss:.4f})")
        
        # Save final model
        self.save_model(f"{self.model_name}_final.pth")
        print(f"Training completed! Final model saved.")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def evaluate_test(self):
        """
        Evaluate model on test data.
        
        Returns:
            Test metrics
        """
        print(f"Evaluating {self.model_name} on test data...")
        
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in self.test_loader:
                # Move data to device
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = (outputs > 0.5).float()
                correct = (predictions == labels).float().sum()
                total_correct += correct.item()
                total_samples += labels.numel()
                
                # Store for detailed analysis
                all_predictions.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate metrics
        test_loss = total_loss / len(self.test_loader)
        test_accuracy = total_correct / total_samples
        
        # Concatenate all predictions and labels
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Calculate per-instrument metrics and F1 scores
        instrument_metrics = self._calculate_instrument_metrics(all_predictions, all_labels)
        f1_scores = self._calculate_f1_scores(all_predictions, all_labels)
        
        print(f"Test Results:")
        print(f"   Loss: {test_loss:.4f}")
        print(f"   Accuracy: {test_accuracy:.4f}")
        print(f"   F1-Micro: {f1_scores['f1_micro']:.4f}")
        print(f"   F1-Macro: {f1_scores['f1_macro']:.4f}")
        print(f"   F1-Weighted: {f1_scores['f1_weighted']:.4f}")
        print(f"   Multi-label samples: {(all_labels.sum(axis=1) > 1).sum()}")
        print(f"   Single-label samples: {(all_labels.sum(axis=1) == 1).sum()}")
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'f1_micro': f1_scores['f1_micro'],
            'f1_macro': f1_scores['f1_macro'],
            'f1_weighted': f1_scores['f1_weighted'],
            'predictions': all_predictions,
            'labels': all_labels,
            'instrument_metrics': instrument_metrics
        }
    
    def _calculate_instrument_metrics(self, predictions, labels):
        """Calculate per-instrument precision, recall, and F1."""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Threshold predictions
        pred_binary = (predictions > 0.5).astype(int)
        
        # Calculate metrics for each instrument
        instruments = ['acoustic guitar', 'cello', 'clarinet', 'electric guitar', 
                      'flute', 'organ', 'piano', 'saxophone', 'trumpet', 'violin', 'voice']
        
        metrics = {}
        for i, instrument in enumerate(instruments):
            precision = precision_score(labels[:, i], pred_binary[:, i], zero_division=0)
            recall = recall_score(labels[:, i], pred_binary[:, i], zero_division=0)
            f1 = f1_score(labels[:, i], pred_binary[:, i], zero_division=0)
            
            metrics[instrument] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return metrics
    
    def _calculate_f1_scores(self, predictions, labels):
        """Calculate F1 scores (micro, macro, weighted) for multi-label classification."""
        from sklearn.metrics import f1_score
        
        # Threshold predictions
        pred_binary = (predictions > 0.5).astype(int)
        
        # Calculate F1 scores
        f1_micro = f1_score(labels, pred_binary, average='micro', zero_division=0)
        f1_macro = f1_score(labels, pred_binary, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, pred_binary, average='weighted', zero_division=0)
        
        return {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }
    
    def save_model(self, filename):
        """Save model to file."""
        save_path = self.save_dir / filename
        torch.save(self.model.state_dict(), save_path)
    
    def load_model(self, filename):
        """Load model from file."""
        load_path = self.save_dir / filename
        self.model.load_state_dict(torch.load(load_path))
        print(f"Loaded model from {load_path}")
    
    def plot_training_history(self):
        """Plot training history and save to root directory."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        

        ax1.plot(self.train_losses, label='Train Loss', linewidth=2, marker='o', markersize=4)
        ax1.plot(self.val_losses, label='Val Loss', linewidth=2, marker='s', markersize=4)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{self.model_name} - Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        

        ax2.plot(self.train_accuracies, label='Train Accuracy', linewidth=2, marker='o', markersize=4)
        ax2.plot(self.val_accuracies, label='Val Accuracy', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{self.model_name} - Training Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()

        save_path = f"{self.model_name}_training_history.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved as: {save_path}")
        
        plt.show()

def train_model(model, model_name, num_epochs=30, batch_size=32, learning_rate=0.001):
    """
    Convenience function to train a model.
    
    Args:
        model: PyTorch model
        model_name: Model name
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Trainer object with training history
    """
    trainer = IRMASTrainer(
        model=model,
        model_name=model_name,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Train the model
    history = trainer.train(num_epochs)
    
    # Evaluate on test data
    test_results = trainer.evaluate_test()
    
    # Plot training history
    trainer.plot_training_history()
    
    return trainer, history, test_results

if __name__ == "__main__":
    # Test the trainer
    get_dataset_info()
    
    print("\n" + "="*60)
    print("TESTING TRAINER")
    print("="*60)
    
    # Create a simple test model
    from vgg_cnn import create_vgg_model
    
    model = create_vgg_model()
    trainer = IRMASTrainer(model, "test_vgg", batch_size=4)
    
    print(f"Trainer created successfully!")
    print(f"   Train batches: {len(trainer.train_loader)}")
    print(f"   Val batches: {len(trainer.val_loader)}")
    print(f"   Test batches: {len(trainer.test_loader)}") 