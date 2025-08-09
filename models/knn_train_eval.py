#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt

print("="*60)
print("MULTI-LABEL INSTRUMENT CLASSIFICATION WITH kNN (k=1) - TRAIN SET EVALUATION")
print("="*60)

# Load training data
X_train = np.load("data/processed/X_train.npy")  # shape: (5364, 128, 128)
y_train = np.load("data/processed/y_train.npy")  # shape: (5364,) - single labels
X_val = np.load("data/processed/X_val.npy")      # shape: (1341, 128, 128)  
y_val = np.load("data/processed/y_val.npy")      # shape: (1341,) - single labels

print(f"Training data: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Validation data: X_val {X_val.shape}, y_val {y_val.shape}")  

# Flatten features for kNN
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# Combine training and validation for more data
X_combined = np.vstack([X_train_flat, X_val_flat])
y_combined = np.hstack([y_train, y_val])

print(f"Combined training data: {X_combined.shape}, {y_combined.shape}")

# Feature scaling
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)

# Convert single-label training data to multi-label format (one-hot encoding)
lb = LabelBinarizer()
y_combined_multilabel = lb.fit_transform(y_combined)
print(f"Multi-label training data shape: {y_combined_multilabel.shape}")

print("\n" + "="*50)
print("TRAINING kNN CLASSIFIER WITH k=1")
print("="*50)

# Train kNN with k=1 (optimal value from previous testing)
knn_model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=1))
print("Training multi-label kNN classifier...")
knn_model.fit(X_combined_scaled, y_combined_multilabel)

# Make predictions ON THE TRAINING SET (same data used for training)
print("Making predictions on TRAINING data...")
y_pred = knn_model.predict(X_combined_scaled)

# Class names
class_names = [
    "acoustic guitar", "cello", "clarinet", "electric guitar", "flute",
    "organ", "piano", "saxophone", "trumpet", "violin", "voice"
]

print(f"\n{'='*50}")
print("TRAINING SET EVALUATION RESULTS (OVERFITTING CHECK)")
print("="*50)

# Calculate main metrics
exact_match = accuracy_score(y_combined_multilabel, y_pred)
f1_micro = f1_score(y_combined_multilabel, y_pred, average='micro')
f1_macro = f1_score(y_combined_multilabel, y_pred, average='macro')
f1_weighted = f1_score(y_combined_multilabel, y_pred, average='weighted')

print("OVERALL PERFORMANCE ON TRAINING SET:")
print(f"Exact Match Ratio: {exact_match:.4f} ({exact_match*100:.2f}%)")
print(f"F1 Score (micro):  {f1_micro:.4f} ({f1_micro*100:.2f}%)")
print(f"F1 Score (macro):  {f1_macro:.4f} ({f1_macro*100:.2f}%)")
print(f"F1 Score (weighted): {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")

# Per-instrument performance
print(f"\n{'='*50}")
print("PER-INSTRUMENT PERFORMANCE ON TRAINING SET")
print("="*50)

instrument_results = []
for i, instrument in enumerate(class_names):
    f1 = f1_score(y_combined_multilabel[:, i], y_pred[:, i])
    precision = np.sum((y_pred[:, i] == 1) & (y_combined_multilabel[:, i] == 1)) / max(1, np.sum(y_pred[:, i] == 1))
    recall = np.sum((y_pred[:, i] == 1) & (y_combined_multilabel[:, i] == 1)) / max(1, np.sum(y_combined_multilabel[:, i] == 1))
    support = np.sum(y_combined_multilabel[:, i] == 1)
    
    instrument_results.append({
        'name': instrument,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'support': support
    })
    
    print(f"{instrument:15s}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Support={support}")

# Prediction analysis
print(f"\n{'='*50}")
print("PREDICTION ANALYSIS ON TRAINING SET")
print("="*50)

ground_truth_avg = y_combined_multilabel.sum(axis=1).mean()
predicted_avg = y_pred.sum(axis=1).mean()
zero_predictions = (y_pred.sum(axis=1) == 0).sum()
single_predictions = (y_pred.sum(axis=1) == 1).sum()
multi_predictions = (y_pred.sum(axis=1) >= 2).sum()

print(f"Average instruments per sample (ground truth): {ground_truth_avg:.2f}")
print(f"Average instruments per sample (predicted):    {predicted_avg:.2f}")
print(f"Samples with 0 predictions: {zero_predictions}")
print(f"Samples with 1 prediction:  {single_predictions}")
print(f"Samples with 2+ predictions: {multi_predictions}")

# Summary
print(f"\n{'='*60}")
print("TRAINING SET EVALUATION SUMMARY")
print("="*60)
print(f"✓ Used k=1 for kNN")
print(f"✓ Training samples: {X_combined.shape[0]:,}")
print(f"✓ Feature dimensions: {X_combined.shape[1]:,}")
print(f"✓ Number of instruments: {len(class_names)}")

print(f"\n🎯 TRAINING SET PERFORMANCE:")
print(f"   • F1-Micro score: {f1_micro:.3f}")
print(f"   • Exact Match Ratio: {exact_match:.3f}")
print(f"   • Average predictions per sample: {predicted_avg:.1f}")

if exact_match > 0.95:
    print(f"\n✅ HIGH TRAINING ACCURACY: Model memorizes training data well (expected for k=1)")
else:
    print(f"\n⚠️  LOWER TRAINING ACCURACY: Unexpected for k=1 kNN - check data processing")

print(f"\n{'='*60}")
print("kNN TRAINING SET EVALUATION COMPLETE")
print("="*60) 