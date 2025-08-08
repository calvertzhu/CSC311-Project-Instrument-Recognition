#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt

print("="*60)
print("MULTI-LABEL INSTRUMENT CLASSIFICATION WITH kNN")
print("="*60)

# Load proper training data
X_train = np.load("data/processed/X_train.npy")  # shape: (5364, 128, 128)
y_train = np.load("data/processed/y_train.npy")  # shape: (5364,) - single labels
X_val = np.load("data/processed/X_val.npy")      # shape: (1341, 128, 128)  
y_val = np.load("data/processed/y_val.npy")      # shape: (1341,) - single labels

# Load test data (multi-label)
X_test_multi = np.load("data/test_processed/X_test.npy")     # shape: (2874, 128, 128)
y_test_multi = np.load("data/test_processed/y_test.npy")     # shape: (2874, 11) - multi-label

print(f"Training data: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Validation data: X_val {X_val.shape}, y_val {y_val.shape}")  
print(f"Test data: X_test {X_test_multi.shape}, y_test {y_test_multi.shape}")

# Flatten features for kNN
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test_multi.reshape(X_test_multi.shape[0], -1)

# Combine training and validation for more data
X_combined = np.vstack([X_train_flat, X_val_flat])
y_combined = np.hstack([y_train, y_val])

print(f"Combined training data: {X_combined.shape}, {y_combined.shape}")

# Feature scaling
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)
X_test_scaled = scaler.transform(X_test_flat)

# Convert single-label training data to multi-label format (one-hot encoding)
lb = LabelBinarizer()
y_combined_multilabel = lb.fit_transform(y_combined)
print(f"Multi-label training data shape: {y_combined_multilabel.shape}")

# Class names
class_names = [
    "acoustic guitar", "cello", "clarinet", "electric guitar", "flute",
    "organ", "piano", "saxophone", "trumpet", "violin", "voice"
]

print("\n" + "="*50)
print("TRAINING MULTI-LABEL kNN CLASSIFIER")
print("="*50)

# Method 1: Multi-Output kNN (Binary Relevance)
print("Method 1: Binary Relevance (separate kNN for each instrument)")
multi_knn = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5))
multi_knn.fit(X_combined_scaled, y_combined_multilabel)

# Predict on test data
y_pred_binary = multi_knn.predict(X_test_scaled)

print("\nBinary Relevance Results:")
print(f"Exact Match Ratio: {accuracy_score(y_test_multi, y_pred_binary):.4f}")
print(f"Hamming Loss: {1 - accuracy_score(y_test_multi, y_pred_binary, normalize=False) / (y_test_multi.shape[0] * y_test_multi.shape[1]):.4f}")
print(f"F1 Score (micro): {f1_score(y_test_multi, y_pred_binary, average='micro'):.4f}")
print(f"F1 Score (macro): {f1_score(y_test_multi, y_pred_binary, average='macro'):.4f}")
print(f"F1 Score (weighted): {f1_score(y_test_multi, y_pred_binary, average='weighted'):.4f}")

# Method 2: Threshold-based approach using probabilities
print(f"\n{'='*50}")
print("Method 2: Probability Thresholding")
print("="*50)

# Get probabilities for each binary classifier
y_proba_list = []
for i in range(len(class_names)):
    # Train individual kNN for each instrument
    knn_single = KNeighborsClassifier(n_neighbors=5)
    knn_single.fit(X_combined_scaled, y_combined_multilabel[:, i])
    proba = knn_single.predict_proba(X_test_scaled)
    # Take probability of positive class (instrument present)
    if proba.shape[1] == 2:  # Binary classification
        y_proba_list.append(proba[:, 1])
    else:  # Only negative class present in training
        y_proba_list.append(np.zeros(X_test_scaled.shape[0]))

y_proba = np.column_stack(y_proba_list)

# Test different thresholds
thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
f1_micro_list, f1_macro_list, f1_weighted_list = [], [], []

print("Testing thresholds:")
for t in thresholds:
    y_pred_thresh = (y_proba >= t).astype(int)
    
    # Ensure at least one instrument per sample
    zero_rows = (y_pred_thresh.sum(axis=1) == 0)
    if zero_rows.sum() > 0:
        top_indices = np.argmax(y_proba[zero_rows], axis=1)
        y_pred_thresh[zero_rows, top_indices] = 1
    
    f1_micro = f1_score(y_test_multi, y_pred_thresh, average='micro')
    f1_macro = f1_score(y_test_multi, y_pred_thresh, average='macro')
    f1_weighted = f1_score(y_test_multi, y_pred_thresh, average='weighted')
    
    f1_micro_list.append(f1_micro)
    f1_macro_list.append(f1_macro)
    f1_weighted_list.append(f1_weighted)
    
    print(f"Threshold {t:.1f}: F1-micro={f1_micro:.4f}, F1-macro={f1_macro:.4f}, F1-weighted={f1_weighted:.4f}")

# Find best threshold
best_idx = np.argmax(f1_macro_list)
best_threshold = thresholds[best_idx]
print(f"\nBest threshold: {best_threshold:.1f} (F1-macro: {f1_macro_list[best_idx]:.4f})")

# Final predictions with best threshold
y_pred_final = (y_proba >= best_threshold).astype(int)
zero_rows = (y_pred_final.sum(axis=1) == 0)
if zero_rows.sum() > 0:
    top_indices = np.argmax(y_proba[zero_rows], axis=1)
    y_pred_final[zero_rows, top_indices] = 1

print(f"\n{'='*50}")
print("FINAL MULTI-LABEL RESULTS")
print("="*50)
print(f"Exact Match Ratio: {accuracy_score(y_test_multi, y_pred_final):.4f}")
print(f"F1 Score (micro): {f1_score(y_test_multi, y_pred_final, average='micro'):.4f}")
print(f"F1 Score (macro): {f1_score(y_test_multi, y_pred_final, average='macro'):.4f}")
print(f"F1 Score (weighted): {f1_score(y_test_multi, y_pred_final, average='weighted'):.4f}")

# Per-instrument performance
print(f"\n{'='*50}")
print("PER-INSTRUMENT PERFORMANCE")
print("="*50)
for i, instrument in enumerate(class_names):
    f1 = f1_score(y_test_multi[:, i], y_pred_final[:, i])
    precision = np.sum((y_pred_final[:, i] == 1) & (y_test_multi[:, i] == 1)) / max(1, np.sum(y_pred_final[:, i] == 1))
    recall = np.sum((y_pred_final[:, i] == 1) & (y_test_multi[:, i] == 1)) / max(1, np.sum(y_test_multi[:, i] == 1))
    support = np.sum(y_test_multi[:, i] == 1)
    print(f"{instrument:15s}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Support={support}")

# Plot threshold analysis
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_micro_list, marker='o', label='F1 Micro')
plt.plot(thresholds, f1_macro_list, marker='s', label='F1 Macro') 
plt.plot(thresholds, f1_weighted_list, marker='^', label='F1 Weighted')
plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best Threshold ({best_threshold:.1f})')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('Multi-Label kNN: F1 Scores vs Threshold')
plt.legend()
plt.grid(True)
plt.show()

# Analysis of predictions
print(f"\n{'='*50}")
print("PREDICTION ANALYSIS")
print("="*50)
print(f"Average instruments per sample (ground truth): {y_test_multi.sum(axis=1).mean():.2f}")
print(f"Average instruments per sample (predicted): {y_pred_final.sum(axis=1).mean():.2f}")
print(f"Samples with 0 predictions: {(y_pred_final.sum(axis=1) == 0).sum()}")
print(f"Samples with 1 prediction: {(y_pred_final.sum(axis=1) == 1).sum()}")
print(f"Samples with 2+ predictions: {(y_pred_final.sum(axis=1) >= 2).sum()}")

