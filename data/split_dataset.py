import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CONFIG

def split_dataset(val_size=0.2, random_state=42):
    """
    Split the processed dataset into train/validation sets.
    
    Args:
        val_size: Proportion of data for validation set (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    # Load processed dataset
    data_dir = DATA_CONFIG["processed_data_dir"]
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    label_classes = np.load(data_dir / "label_classes.npy")
    
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]}x{X.shape[2]} features")
    print(f"Number of classes: {len(label_classes)}")
    print(f"Classes: {label_classes}")
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=random_state
    )
    
    return X_train, X_val, y_train, y_val

def save_splits(X_train, X_val, y_train, y_val, out_dir=None):
    """
    Save the dataset splits to files.
    
    Args:
        X_train, X_val: Feature arrays for each split
        y_train, y_val: Label arrays for each split
        out_dir: Output directory. If None, uses config default.
    """
    if out_dir is None:
        out_dir = DATA_CONFIG["processed_data_dir"]
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_val.npy", y_val)
    
    print(f"\nSaved dataset splits to {out_dir}/")
    print(f"  Training set:   {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"  Total:          {X_train.shape[0] + X_val.shape[0]} samples")

def load_splits(data_dir=None):
    """
    Load previously saved dataset splits.
    
    Args:
        data_dir: Directory containing split data. If None, uses config default.
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    if data_dir is None:
        data_dir = DATA_CONFIG["processed_data_dir"]
    
    data_dir = Path(data_dir)
    
    X_train = np.load(data_dir / "X_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_train = np.load(data_dir / "y_train.npy")
    y_val = np.load(data_dir / "y_val.npy")
    
    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split IRMAS dataset into train/val sets')
    parser.add_argument('--val-size', type=float, default=0.2,
                       help='Proportion of data for validation set (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("IRMAS Dataset Splitting")
    print("=" * 50)
    
    try:
        # Split dataset
        X_train, X_val, y_train, y_val = split_dataset(
            val_size=args.val_size,
            random_state=args.random_state
        )
        
        # Save splits
        save_splits(X_train, X_val, y_train, y_val)
        
        print("\n✅ Dataset splitting completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run the feature extractor first to generate processed data.")
        print("Run: python data/feature_extractor.py")
    except Exception as e:
        print(f"❌ Error: {e}")
