"""
Test data loader for the IRMAS test dataset.
Handles multi-label annotations and variable-length audio files across 3 parts.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_CONFIG, LABEL_MAP

def load_test_file_paths_from_parts() -> List[Tuple[str, str]]:
    """
    Load file paths from all three test parts.
    
    Returns:
        List of (wav_file_path, txt_file_path) tuples
    """
    file_pairs = []
    
    # Load from all three parts
    test_parts = [
        DATA_CONFIG["test_data_part1"],
        DATA_CONFIG["test_data_part2"], 
        DATA_CONFIG["test_data_part3"]
    ]
    
    for part_dir in test_parts:
        part_dir = Path(part_dir)
        
        if not part_dir.exists():
            print(f"Warning: Test data directory {part_dir} does not exist!")
            continue
            
        # Find all WAV files
        for wav_file in part_dir.glob("*.wav"):
            # Get corresponding annotation file
            txt_file = wav_file.with_suffix('.txt')
            
            if txt_file.exists():
                file_pairs.append((str(wav_file), str(txt_file)))
            else:
                print(f"Warning: No annotation file found for {wav_file}")
    
    return file_pairs

def parse_annotation_file(txt_file_path: str) -> List[str]:
    """
    Parse the annotation file to get instrument labels.
    
    Args:
        txt_file_path: Path to the .txt annotation file
        
    Returns:
        List of instrument names (e.g., ['piano', 'violin'])
    """
    instruments = []
    
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                instrument_code = line.strip()
                if instrument_code in LABEL_MAP:
                    instruments.append(LABEL_MAP[instrument_code])
                elif instrument_code:  # Non-empty line
                    print(f"Warning: Unknown instrument code '{instrument_code}' in {txt_file_path}")
    except Exception as e:
        print(f"Error reading {txt_file_path}: {e}")
    
    return instruments

def load_test_dataset_info() -> List[Dict]:
    """
    Load complete test dataset information.
    
    Returns:
        List of dictionaries with keys:
        - 'wav_path': Path to audio file
        - 'txt_path': Path to annotation file  
        - 'instruments': List of instrument names
        - 'part': Which part the file comes from (1, 2, or 3)
        - 'filename': Base filename without extension
    """
    dataset_info = []
    
    # Load from all three parts
    test_parts = [
        (DATA_CONFIG["test_data_part1"], 1),
        (DATA_CONFIG["test_data_part2"], 2),
        (DATA_CONFIG["test_data_part3"], 3)
    ]
    
    for part_dir, part_num in test_parts:
        part_dir = Path(part_dir)
        
        if not part_dir.exists():
            print(f"Warning: Test data directory {part_dir} does not exist!")
            continue
            
        print(f"Loading from Part {part_num}: {part_dir}")
        
        # Find all WAV files
        wav_files = list(part_dir.glob("*.wav"))
        
        for wav_file in wav_files:
            # Get corresponding annotation file
            txt_file = wav_file.with_suffix('.txt')
            
            if txt_file.exists():
                # Parse instruments
                instruments = parse_annotation_file(str(txt_file))
                
                if instruments:  # Only include files with valid annotations
                    dataset_info.append({
                        'wav_path': str(wav_file),
                        'txt_path': str(txt_file),
                        'instruments': instruments,
                        'part': part_num,
                        'filename': wav_file.stem
                    })
                else:
                    print(f"Warning: No valid instruments found in {txt_file}")
            else:
                print(f"Warning: No annotation file found for {wav_file}")
    
    return dataset_info

def get_test_dataset_statistics() -> Dict:
    """
    Get statistics about the test dataset.
    
    Returns:
        Dictionary with dataset statistics
    """
    dataset_info = load_test_dataset_info()
    
    # Count files per part
    part_counts = {1: 0, 2: 0, 3: 0}
    instrument_counts = {}
    multi_label_count = 0
    
    for item in dataset_info:
        # Count files per part
        part_counts[item['part']] += 1
        
        # Count instruments
        for instrument in item['instruments']:
            instrument_counts[instrument] = instrument_counts.get(instrument, 0) + 1
        
        # Count multi-label samples
        if len(item['instruments']) > 1:
            multi_label_count += 1
    
    stats = {
        'total_files': len(dataset_info),
        'part_counts': part_counts,
        'instrument_counts': instrument_counts,
        'multi_label_files': multi_label_count,
        'single_label_files': len(dataset_info) - multi_label_count,
        'unique_instruments': len(instrument_counts),
        'available_instruments': sorted(instrument_counts.keys())
    }
    
    return stats

def create_binary_labels(instruments: List[str], all_instruments: List[str]) -> List[int]:
    """
    Create binary label vector for multi-label classification.
    
    Args:
        instruments: List of instruments present in this sample
        all_instruments: List of all possible instruments (ordered)
        
    Returns:
        Binary vector where 1 indicates presence of instrument
    """
    binary_labels = [0] * len(all_instruments)
    
    for instrument in instruments:
        if instrument in all_instruments:
            idx = all_instruments.index(instrument)
            binary_labels[idx] = 1
    
    return binary_labels

if __name__ == "__main__":
    print("=" * 60)
    print("IRMAS Test Dataset Analysis")
    print("=" * 60)
    
    # Load and display statistics
    stats = get_test_dataset_statistics()
    
    print(f"\nDataset Overview:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Part 1: {stats['part_counts'][1]} files")
    print(f"  Part 2: {stats['part_counts'][2]} files") 
    print(f"  Part 3: {stats['part_counts'][3]} files")
    
    print(f"\nLabel Information:")
    print(f"  Single-label files: {stats['single_label_files']}")
    print(f"  Multi-label files: {stats['multi_label_files']}")
    print(f"  Unique instruments: {stats['unique_instruments']}")
    
    print(f"\nInstrument Distribution:")
    for instrument, count in sorted(stats['instrument_counts'].items()):
        print(f"  {instrument}: {count} files")
    
    print(f"\nSample Files:")
    dataset_info = load_test_dataset_info()
    
    # Show some examples
    print("  Single-label examples:")
    single_examples = [item for item in dataset_info if len(item['instruments']) == 1][:3]
    for item in single_examples:
        print(f"    {item['filename']}: {item['instruments'][0]} (Part {item['part']})")
    
    print("  Multi-label examples:")
    multi_examples = [item for item in dataset_info if len(item['instruments']) > 1][:3]
    for item in multi_examples:
        instruments_str = ', '.join(item['instruments'])
        print(f"    {item['filename']}: {instruments_str} (Part {item['part']})")
    
    print(f"\nTest data loader ready for {stats['total_files']} files!") 