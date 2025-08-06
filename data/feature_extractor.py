import numpy as np
import librosa
import librosa.util
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CONFIG
from data.data_loader import load_file_paths_from_folders

def extract_features(path, feature_type='mel', duration=3.0, sr=22050, n_mels=128):
    """
    Extract audio features from a file.
    
    Args:
        path: Path to audio file
        feature_type: Type of features to extract ('mel' or 'mfcc')
        duration: Duration to load in seconds
        sr: Sample rate
        n_mels: Number of mel bands for mel spectrogram
        
    Returns:
        Extracted features as numpy array
    """
    y, _ = librosa.load(path, sr=sr, duration=duration)
    
    if feature_type == 'mel':
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        return librosa.util.fix_length(spec_db, size=128, axis=1)
    elif feature_type == 'mfcc':
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        return librosa.util.fix_length(mfcc, size=128, axis=1)
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

def build_dataset():
    """
    Build dataset by extracting features from all audio files.
    
    Returns:
        Tuple of (X, y) where X is feature array and y is label array
    """
    data = load_file_paths_from_folders()
    X = []
    y = []
    
    print(f"Processing {len(data)} audio files...")
    
    for path, label in tqdm(data, desc="Extracting features"):
        try:
            feat = extract_features(
                path,
                feature_type=DATA_CONFIG["feature_type"],
                duration=DATA_CONFIG["duration"],
                sr=DATA_CONFIG["sample_rate"],
                n_mels=DATA_CONFIG["n_mels"]
            )
            X.append(feat)
            y.append(label)
        except Exception as e:
            print(f"Error extracting from {path}: {e}")
    
    return np.array(X), np.array(y)

def save_dataset(X, y, label_encoder, out_dir=None):
    """
    Save processed dataset to files.
    
    Args:
        X: Feature array
        y: Label array (raw strings)
        label_encoder: Fitted LabelEncoder
        out_dir: Output directory. If None, uses config default.
    """
    if out_dir is None:
        out_dir = DATA_CONFIG["processed_data_dir"]
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", label_encoder.transform(y))
    np.save(out_dir / "label_classes.npy", label_encoder.classes_)
    
    print(f"Saved dataset to {out_dir}/")
    print(f"  - X.npy: {X.shape}")
    print(f"  - y.npy: {label_encoder.transform(y).shape}")
    print(f"  - label_classes.npy: {label_encoder.classes_.shape}")

def load_processed_dataset(data_dir=None):
    """
    Load previously processed dataset.
    
    Args:
        data_dir: Directory containing processed data. If None, uses config default.
        
    Returns:
        Tuple of (X, y, label_classes)
    """
    if data_dir is None:
        data_dir = DATA_CONFIG["processed_data_dir"]
    
    data_dir = Path(data_dir)
    
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    label_classes = np.load(data_dir / "label_classes.npy")
    
    return X, y, label_classes

if __name__ == "__main__":
    from sklearn.preprocessing import LabelEncoder

    print("Building dataset...")
    X, y_raw = build_dataset()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {len(y_raw)}")
    print(f"Unique labels: {set(y_raw)}")
    
    le = LabelEncoder()
    le.fit(y_raw)
    
    save_dataset(X, y_raw, le) 