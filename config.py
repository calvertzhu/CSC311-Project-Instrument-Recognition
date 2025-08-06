import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data configuration
DATA_CONFIG = {
    "sample_rate": 22050,
    "duration": 3.0,
    "feature_type": "mel",
    "n_mels": 128,
    "target_shape": [128, 128],
    "raw_data_dir": PROJECT_ROOT / "datasets" / "IRMAS" / "IRMAS-TrainingData",
    "processed_data_dir": PROJECT_ROOT / "data" / "processed"
}

# Model configuration
MODEL_CONFIG = {
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "random_state": 42
}

# Instrument label mapping
LABEL_MAP = {
    'cel': 'cello',
    'cla': 'clarinet', 
    'flu': 'flute',
    'gac': 'acoustic guitar',
    'gel': 'electric guitar',
    'org': 'organ',
    'pia': 'piano',
    'sax': 'saxophone',
    'tru': 'trumpet',
    'vio': 'violin',
    'voi': 'voice'
}

# Paths
PATHS = {
    "models": PROJECT_ROOT / "models" / "saved_models",
    "notebooks": PROJECT_ROOT / "notebooks",
    "data": PROJECT_ROOT / "data",
    "datasets": PROJECT_ROOT / "datasets"
}

# Ensure directories exist
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)
    
DATA_CONFIG["processed_data_dir"].mkdir(parents=True, exist_ok=True) 