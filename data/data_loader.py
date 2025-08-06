import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CONFIG, LABEL_MAP

def load_file_paths_from_folders(base_dir=None):
    """
    Walk through instrument folders and return list of (file_path, label) tuples.
    
    """
    if base_dir is None:
        base_dir = DATA_CONFIG["raw_data_dir"]
    
    base_dir = Path(base_dir)
    data = []
    
    if not base_dir.exists():
        print(f"Warning: Data directory {base_dir} does not exist!")
        print("Please place your IRMAS dataset in the datasets/IRMAS/ directory")
        return data
    
    for code in os.listdir(base_dir):
        folder_path = base_dir / code
        if folder_path.is_dir() and code in LABEL_MAP:
            label = LABEL_MAP[code]
            for fname in os.listdir(folder_path):
                if fname.endswith('.wav'):
                    full_path = folder_path / fname
                    data.append((str(full_path), label))
    return data

if __name__ == "__main__":
    print("Using configuration:")
    for k, v in DATA_CONFIG.items():
        print(f"{k}: {v}")

    file_label_pairs = load_file_paths_from_folders()
    print(f"Found {len(file_label_pairs)} audio files.")

    # Example: print the first 5 entries
    for path, label in file_label_pairs[:5]:
        print(f"{label}: {path}") 