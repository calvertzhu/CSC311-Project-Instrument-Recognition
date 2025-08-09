# IRMAS Music Instrument Recognition Models

This directory contains the complete model training pipeline for the IRMAS dataset.

- **Architecture**: VGG-style convolutional neural network
- **Parameters**: ~15.1M (VGG-16 style)
- **Input**: Mel spectrograms (128×128)
- **Output**: Multi-label classification (11 instruments)
- **Configurations**: A, B, C, D, E (VGG-11 to VGG-19 style)


## Quick Start

### 1. Test Everything Works
```bash
python3 models/main.py
```

### 2. Train VGG Model
```bash
# Basic training (30 epochs)
python3 models/main.py --model vgg --epochs 30

# Custom configuration
python3 models/main.py --model vgg --epochs 50 --batch-size 16 --lr 0.0005 --config C

# Save detailed results
python3 models/main.py --model vgg --epochs 30 --save-results
```
### 3. Skip Test Evaluation (faster training)
```bash
python3 models/main.py --model vgg --epochs 30 --no-test
```

## 🔧 Data Requirements

Before training, ensure you have processed data:
- `data/processed/X_train.npy` - Training features
- `data/processed/y_train.npy` - Training labels
- `data/processed/X_val.npy` - Validation features
- `data/processed/y_val.npy` - Validation labels
- `data/test_processed/X_test.npy` - Test features
- `data/test_processed/y_test.npy` - Test labels

## Training Features

### Automatic Data Handling
- **Training Data**: Single-label → Multi-label conversion
- **Test Data**: Already multi-label (no conversion needed)
- **All Datasets**: Compatible with multi-label models

### Training Components
- **Loss Function**: BCELoss (Binary Cross Entropy)
- **Optimizer**: Adam with weight decay
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Visualization**: Training curves and results

### Model Saving
- **Best Model**: Saved based on validation loss
- **Final Model**: Saved after all epochs
- **Results**: JSON file with detailed metrics

## Output Files

### Models
- `models/saved_models/{model_name}_best.pth` - Best model
- `models/saved_models/{model_name}_final.pth` - Final model

### Results (if --save-results)
- `models/results/{model_name}_{timestamp}_results.json` - Detailed results

## Performance Metrics

The system provides:
- **Overall Accuracy**: Percentage of correct predictions
- **Per-Instrument Metrics**: Precision, Recall, F1 for each instrument
- **Training History**: Loss and accuracy curves
- **Test Evaluation**: Comprehensive evaluation on test data

### Training Parameters
```python
trainer = IRMASTrainer(
    model=model,
    model_name="my_model",
    batch_size=32,
    learning_rate=0.001,
    weight_decay=1e-4
)
```

## Instrument Classes

The model recognizes 11 instruments:
1. acoustic guitar
2. cello
3. clarinet
4. electric guitar
5. flute
6. organ
7. piano
8. saxophone
9. trumpet
10. violin
11. voice
