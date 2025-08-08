#!/usr/bin/env python3
"""
Test feature extractor for the IRMAS test dataset.
Handles variable-length audio files (5-20 seconds) and multi-label annotations.
"""

import numpy as np
import librosa
import librosa.util
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CONFIG, LABEL_MAP
from data.test_data_loader import load_test_dataset_info, create_binary_labels

def extract_test_features(audio_path: str, target_duration: float = None, single_segment: bool = True) -> np.ndarray:
    """
    Extract features from test audio file with variable length handling.
    
    Args:
        audio_path: Path to the audio file
        target_duration: If specified, segments will be extracted at this duration
                        If None, uses the full audio length
        single_segment: If True, extract only one segment per file (default: True)
        
    Returns:
        Feature array of shape (n_mels, time_frames) or (n_segments, n_mels, time_frames)
    """
    try:
        # Load audio file
        y, sr = librosa.load(
            audio_path, 
            sr=DATA_CONFIG["sample_rate"],
            duration=None  # Load full file for test data
        )
        
        duration = len(y) / sr
        n_mels = DATA_CONFIG["n_mels"]
        
        if target_duration is None:
            # Extract features from full audio
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=n_mels,
                hop_length=512,
                win_length=2048
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Resize to target shape if specified
            if "target_shape" in DATA_CONFIG:
                target_shape = DATA_CONFIG["target_shape"]
                mel_spec_db = librosa.util.fix_length(
                    mel_spec_db, 
                    size=target_shape[1], 
                    axis=1
                )
            
            return mel_spec_db
            
        else:
            # Extract segment(s) at fixed duration
            segment_samples = int(target_duration * sr)
            
            if len(y) <= segment_samples:
                # Audio is shorter than target, pad it
                y_padded = librosa.util.fix_length(y, size=segment_samples)
                
                mel_spec = librosa.feature.melspectrogram(
                    y=y_padded, 
                    sr=sr, 
                    n_mels=n_mels,
                    hop_length=512,
                    win_length=2048
                )
                
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                if "target_shape" in DATA_CONFIG:
                    target_shape = DATA_CONFIG["target_shape"]
                    mel_spec_db = librosa.util.fix_length(
                        mel_spec_db, 
                        size=target_shape[1], 
                        axis=1
                    )
                
                return mel_spec_db
                
            else:
                if single_segment:
                    # Extract only one segment from the middle of the audio
                    middle_start = (len(y) - segment_samples) // 2
                    y_segment = y[middle_start:middle_start + segment_samples]
                    
                    mel_spec = librosa.feature.melspectrogram(
                        y=y_segment, 
                        sr=sr, 
                        n_mels=n_mels,
                        hop_length=512,
                        win_length=2048
                    )
                    
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    if "target_shape" in DATA_CONFIG:
                        target_shape = DATA_CONFIG["target_shape"]
                        mel_spec_db = librosa.util.fix_length(
                            mel_spec_db, 
                            size=target_shape[1], 
                            axis=1
                        )
                    
                    return mel_spec_db
                    
                else:
                    # Extract multiple overlapping segments (original behavior)
                    hop_samples = segment_samples // 2  # 50% overlap
                    segments = []
                    
                    for start in range(0, len(y) - segment_samples + 1, hop_samples):
                        y_segment = y[start:start + segment_samples]
                        
                        mel_spec = librosa.feature.melspectrogram(
                            y=y_segment, 
                            sr=sr, 
                            n_mels=n_mels,
                            hop_length=512,
                            win_length=2048
                        )
                        
                        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                        
                        if "target_shape" in DATA_CONFIG:
                            target_shape = DATA_CONFIG["target_shape"]
                            mel_spec_db = librosa.util.fix_length(
                                mel_spec_db, 
                                size=target_shape[1], 
                                axis=1
                            )
                        
                        segments.append(mel_spec_db)
                    
                    return np.array(segments)
                
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

def build_test_dataset(segment_duration: float = None, max_files: int = None, max_segments: int = None):
    """
    Build test dataset with multi-label annotations.
    
    Args:
        segment_duration: If specified, extract segments of this duration
                         If None, use variable-length features
        max_files: Maximum number of files to process (for testing)
        max_segments: Maximum number of segments to generate (for controlling dataset size)
        
    Returns:
        Tuple of (X, y_binary, metadata) where:
        - X: Feature array 
        - y_binary: Binary label matrix (n_samples, n_instruments)
        - metadata: List of dictionaries with file information
    """
    print("Loading test dataset information...")
    dataset_info = load_test_dataset_info()
    
    if max_files:
        dataset_info = dataset_info[:max_files]
        print(f"Processing subset: {len(dataset_info)} files")
    else:
        print(f"Processing all {len(dataset_info)} test files...")
    
    # Get all instruments for binary encoding
    all_instruments = sorted(list(LABEL_MAP.values()))
    print(f"Instruments: {all_instruments}")
    
    X = []
    y_binary = []
    metadata = []
    
    print("\nExtracting features...")
    failed_files = 0
    segments_generated = 0
    
    for i, item in enumerate(dataset_info):
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(dataset_info)} files...")
        
        # Check if we've reached the segment limit
        if max_segments and segments_generated >= max_segments:
            print(f"Reached segment limit of {max_segments}, stopping...")
            break
        
        try:
            # Extract features (single segment per file)
            features = extract_test_features(
                item['wav_path'], 
                target_duration=segment_duration,
                single_segment=True
            )
            
            if features is not None:
                # Create binary labels
                binary_labels = create_binary_labels(item['instruments'], all_instruments)
                
                # Single segment per file (always)
                if not max_segments or segments_generated < max_segments:
                    X.append(features)
                    y_binary.append(binary_labels)
                    metadata.append({
                        'filename': item['filename'],
                        'part': item['part'],
                        'instruments': item['instruments'],
                        'segment_idx': 0
                    })
                    segments_generated += 1
            else:
                failed_files += 1
                
        except Exception as e:
            print(f"Error processing {item['filename']}: {e}")
            failed_files += 1
    
    print(f"\nFeature extraction completed!")
    print(f"Successfully processed: {len(X)} samples")
    print(f"Segments generated: {segments_generated}")
    print(f"Failed files: {failed_files}")
    
    return np.array(X), np.array(y_binary), metadata

def save_test_dataset(X, y_binary, metadata, output_dir=None):
    """
    Save processed test dataset.
    
    Args:
        X: Feature array
        y_binary: Binary label matrix
        metadata: Metadata list
        output_dir: Output directory (default: config test_processed_dir)
    """
    if output_dir is None:
        output_dir = DATA_CONFIG["test_processed_dir"]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving test dataset to {output_dir}...")
    
    # Save features and labels
    np.save(output_dir / "X_test.npy", X)
    np.save(output_dir / "y_test.npy", y_binary)
    
    # Save metadata as numpy array of dictionaries
    np.save(output_dir / "test_metadata.npy", metadata)
    
    # Save instrument names for reference
    all_instruments = sorted(list(LABEL_MAP.values()))
    np.save(output_dir / "instrument_classes.npy", all_instruments)
    
    # Save summary statistics
    stats = {
        'total_samples': len(X),
        'feature_shape': X.shape[1:],
        'num_instruments': len(all_instruments),
        'instruments': all_instruments,
        'label_distribution': y_binary.sum(axis=0).tolist(),
        'multi_label_samples': (y_binary.sum(axis=1) > 1).sum(),
        'single_label_samples': (y_binary.sum(axis=1) == 1).sum()
    }
    
    np.save(output_dir / "test_stats.npy", stats)
    
    print(f"   Saved test dataset:")
    print(f"   Features: {X.shape}")
    print(f"   Labels: {y_binary.shape}")
    print(f"   Multi-label samples: {stats['multi_label_samples']}")
    print(f"   Single-label samples: {stats['single_label_samples']}")

def load_test_dataset(data_dir=None):
    """
    Load processed test dataset.
    
    Args:
        data_dir: Data directory (default: config test_processed_dir)
        
    Returns:
        Tuple of (X, y_binary, metadata, stats)
    """
    if data_dir is None:
        data_dir = DATA_CONFIG["test_processed_dir"]
    
    data_dir = Path(data_dir)
    
    X = np.load(data_dir / "X_test.npy")
    y_binary = np.load(data_dir / "y_test.npy")
    metadata = np.load(data_dir / "test_metadata.npy", allow_pickle=True)
    stats = np.load(data_dir / "test_stats.npy", allow_pickle=True).item()
    
    return X, y_binary, metadata, stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from IRMAS test dataset')
    parser.add_argument('--segment-duration', type=float, default=None,
                        help='Extract segments of fixed duration (seconds). If None, use variable length.')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process (for testing)')
    parser.add_argument('--max-segments', type=int, default=None,
                        help='Maximum number of segments to generate (for controlling dataset size)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("IRMAS Test Dataset Feature Extraction")
    print("=" * 60)
    
    if args.segment_duration:
        print(f"Mode: Fixed segments ({args.segment_duration}s)")
    else:
        print("Mode: Variable-length features")
    
    try:
        # Extract features
        X, y_binary, metadata = build_test_dataset(
            segment_duration=args.segment_duration,
            max_files=args.max_files,
            max_segments=args.max_segments
        )
        
        # Save dataset
        save_test_dataset(X, y_binary, metadata)
        
        print("\nTest dataset processing completed")
        
        # Load and verify
        X_loaded, y_loaded, meta_loaded, stats = load_test_dataset()
        print(f"\nFinal dataset statistics:")
        print(f"   Shape: {X_loaded.shape}")
        print(f"   Instruments: {len(stats['instruments'])}")
        print(f"   Multi-label ratio: {stats['multi_label_samples']/stats['total_samples']:.2%}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 