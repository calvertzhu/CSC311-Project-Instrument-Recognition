"""
Data loading and preprocessing utilities for the IRMAS music instrument recognition project.
"""

from .data_loader import load_file_paths_from_folders
from .feature_extractor import extract_features, build_dataset, save_dataset, load_processed_dataset
from .split_dataset import split_dataset, save_splits, load_splits
from .test_data_loader import (
    load_test_file_paths_from_parts, 
    load_test_dataset_info, 
    get_test_dataset_statistics,
    parse_annotation_file,
    create_binary_labels
)
from .test_feature_extractor import (
    extract_test_features,
    build_test_dataset,
    save_test_dataset,
    load_test_dataset
)

__all__ = [
    'load_file_paths_from_folders',
    'extract_features', 
    'build_dataset',
    'save_dataset',
    'load_processed_dataset',
    'split_dataset',
    'save_splits',
    'load_splits',
    'load_test_file_paths_from_parts',
    'load_test_dataset_info',
    'get_test_dataset_statistics',
    'parse_annotation_file',
    'create_binary_labels',
    'extract_test_features',
    'build_test_dataset',
    'save_test_dataset',
    'load_test_dataset'
] 