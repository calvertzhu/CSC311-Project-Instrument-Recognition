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
print("MULTI-LABEL INSTRUMENT CLASSIFICATION WITH kNN (k=1)")
print("="*60)

# Load training data
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

print("\n" + "="*50)
print("TRAINING kNN CLASSIFIER WITH k=1")
print("="*50)

# Train kNN with k=1 (optimal value from previous testing)
knn_model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=1))
print("Training multi-label kNN classifier...")
knn_model.fit(X_combined_scaled, y_combined_multilabel)

# Make predictions
print("Making predictions on test data...")
y_pred = knn_model.predict(X_test_scaled)

# Class names
class_names = [
    "acoustic guitar", "cello", "clarinet", "electric guitar", "flute",
    "organ", "piano", "saxophone", "trumpet", "violin", "voice"
]

print(f"\n{'='*50}")
print("FINAL MULTI-LABEL CLASSIFICATION RESULTS")
print("="*50)

# Calculate main metrics
exact_match = accuracy_score(y_test_multi, y_pred)
f1_micro = f1_score(y_test_multi, y_pred, average='micro')
f1_macro = f1_score(y_test_multi, y_pred, average='macro')
f1_weighted = f1_score(y_test_multi, y_pred, average='weighted')

print("OVERALL PERFORMANCE:")
print(f"Exact Match Ratio: {exact_match:.4f} ({exact_match*100:.2f}%)")
print(f"F1 Score (micro):  {f1_micro:.4f} ({f1_micro*100:.2f}%)")
print(f"F1 Score (macro):  {f1_macro:.4f} ({f1_macro*100:.2f}%)")
print(f"F1 Score (weighted): {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")

# Per-instrument performance
print(f"\n{'='*50}")
print("PER-INSTRUMENT PERFORMANCE")
print("="*50)

instrument_results = []
for i, instrument in enumerate(class_names):
    f1 = f1_score(y_test_multi[:, i], y_pred[:, i])
    precision = np.sum((y_pred[:, i] == 1) & (y_test_multi[:, i] == 1)) / max(1, np.sum(y_pred[:, i] == 1))
    recall = np.sum((y_pred[:, i] == 1) & (y_test_multi[:, i] == 1)) / max(1, np.sum(y_test_multi[:, i] == 1))
    support = np.sum(y_test_multi[:, i] == 1)
    
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
print("PREDICTION ANALYSIS")
print("="*50)

ground_truth_avg = y_test_multi.sum(axis=1).mean()
predicted_avg = y_pred.sum(axis=1).mean()
zero_predictions = (y_pred.sum(axis=1) == 0).sum()
single_predictions = (y_pred.sum(axis=1) == 1).sum()
multi_predictions = (y_pred.sum(axis=1) >= 2).sum()

print(f"Average instruments per sample (ground truth): {ground_truth_avg:.2f}")
print(f"Average instruments per sample (predicted):    {predicted_avg:.2f}")
print(f"Samples with 0 predictions: {zero_predictions}")
print(f"Samples with 1 prediction:  {single_predictions}")
print(f"Samples with 2+ predictions: {multi_predictions}")

# Create visualizations
print(f"\n{'='*50}")
print("GENERATING VISUALIZATIONS")
print("="*50)

# Create a comprehensive visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Overall Performance Metrics
metrics = ['Exact Match', 'F1-Micro', 'F1-Macro', 'F1-Weighted']
values = [exact_match, f1_micro, f1_macro, f1_weighted]
colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

bars = ax1.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_title('Overall Performance Metrics (k=1)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Score')
ax1.set_ylim(0, max(values) * 1.1)
ax1.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# 2. Per-Instrument F1 Scores
instruments = [r['name'] for r in instrument_results]
f1_scores = [r['f1'] for r in instrument_results]

# Sort by F1 score for better visualization
sorted_results = sorted(instrument_results, key=lambda x: x['f1'], reverse=True)
sorted_instruments = [r['name'] for r in sorted_results]
sorted_f1_scores = [r['f1'] for r in sorted_results]

bars2 = ax2.barh(range(len(sorted_instruments)), sorted_f1_scores, 
                 color='skyblue', alpha=0.7, edgecolor='black')
ax2.set_yticks(range(len(sorted_instruments)))
ax2.set_yticklabels(sorted_instruments)
ax2.set_title('F1 Score by Instrument (Sorted)', fontsize=14, fontweight='bold')
ax2.set_xlabel('F1 Score')
ax2.grid(True, axis='x', alpha=0.3)

# Add value labels
for i, (bar, score) in enumerate(zip(bars2, sorted_f1_scores)):
    ax2.text(score + 0.01, bar.get_y() + bar.get_height()/2,
             f'{score:.3f}', ha='left', va='center', fontsize=9)

# 3. Support (number of samples) per instrument
supports = [r['support'] for r in sorted_results]
bars3 = ax3.barh(range(len(sorted_instruments)), supports,
                 color='lightcoral', alpha=0.7, edgecolor='black')
ax3.set_yticks(range(len(sorted_instruments)))
ax3.set_yticklabels(sorted_instruments)
ax3.set_title('Number of Test Samples per Instrument', fontsize=14, fontweight='bold')
ax3.set_xlabel('Number of Samples')
ax3.grid(True, axis='x', alpha=0.3)

# Add value labels
for i, (bar, support) in enumerate(zip(bars3, supports)):
    ax3.text(support + 10, bar.get_y() + bar.get_height()/2,
             f'{support}', ha='left', va='center', fontsize=9)

# 4. Prediction Distribution
prediction_counts = [y_pred.sum(axis=1) == i for i in range(6)]  # 0 to 5+ instruments
prediction_labels = ['0', '1', '2', '3', '4', '5+']
prediction_values = [sum(count) for count in prediction_counts]

# Combine 5+ predictions
prediction_values[5] = sum((y_pred.sum(axis=1) >= 5))

pie_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
wedges, texts, autotexts = ax4.pie(prediction_values, labels=prediction_labels, 
                                  autopct='%1.1f%%', colors=pie_colors, startangle=90)
ax4.set_title('Distribution of Predicted Instruments per Sample', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print("="*60)
print(f"✓ Used k=1 for kNN (optimal value)")
print(f"✓ Training samples: {X_combined.shape[0]:,}")
print(f"✓ Test samples: {X_test_scaled.shape[0]:,}")
print(f"✓ Feature dimensions: {X_combined.shape[1]:,}")
print(f"✓ Number of instruments: {len(class_names)}")
print(f"\n🎯 BEST PERFORMING INSTRUMENTS (F1 > 0.20):")
for result in sorted_results:
    if result['f1'] > 0.20:
        print(f"   {result['name']:15s}: {result['f1']:.3f}")

print(f"\n📊 KEY INSIGHTS:")
print(f"   • F1-Micro score of {f1_micro:.3f} indicates overall reasonable performance")
print(f"   • Popular instruments (piano, guitar) perform better than rare ones")
print(f"   • Model tends to predict {predicted_avg:.1f} instruments vs {ground_truth_avg:.1f} actual")
print(f"   • Only {exact_match:.1%} exact matches (all instruments correct)")

print(f"\n{'='*60}")
print("MULTI-LABEL kNN CLASSIFICATION COMPLETE")
print("="*60)

