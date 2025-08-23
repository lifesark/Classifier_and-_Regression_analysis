"""
PyTorch DataLoader implementations for violence detection.

This module provides custom Dataset classes and DataLoader utilities
for efficient training and evaluation of violence detection models.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import json
import random
from .preprocessing import VideoFrameExtractor, VideoPreprocessor, VideoAugmentation
import logging

logger = logging.getLogger(__name__)

class ViolenceDetectionDataset(Dataset):
    """Dataset class for violence detection."""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = "train",
                 frames_per_clip: int = 16,
                 target_size: Tuple[int, int] = (224, 224),
                 augment: bool = True,
                 cache_data: bool = False):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing preprocessed data
            split: Dataset split ('train', 'val', 'test')
            frames_per_clip: Number of frames per video clip
            target_size: Target frame size
            augment: Whether to apply data augmentation
            cache_data: Whether to cache data in memory
        """
        self.data_dir = data_dir
        self.split = split
        self.frames_per_clip = frames_per_clip
        self.target_size = target_size
        self.augment = augment and split == "train"
        self.cache_data = cache_data
        
        # Initialize preprocessor and augmentation
        self.preprocessor = VideoPreprocessor(target_size=target_size)
        self.augmentation = VideoAugmentation() if self.augment else None
        
        # Load data paths and labels
        self.samples = self._load_samples()
        
        # Cache for loaded data
        self.data_cache = {} if cache_data else None
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load sample paths and labels."""
        samples = []
        
        # Define class mapping
        class_to_idx = {"non_violence": 0, "violence": 1}
        
        for class_name, class_idx in class_to_idx.items():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Class directory not found: {class_dir}")
                continue
                
            # Get all .pt files in the class directory
            pt_files = [f for f in os.listdir(class_dir) if f.endswith('.pt')]
            
            for pt_file in pt_files:
                file_path = os.path.join(class_dir, pt_file)
                samples.append((file_path, class_idx))
        
        # Shuffle samples for training
        if self.split == "train":
            random.shuffle(samples)
            
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (frames, label)
        """
        file_path, label = self.samples[idx]
        
        # Check cache first
        if self.data_cache is not None and file_path in self.data_cache:
            frames = self.data_cache[file_path].clone()
        else:
            # Load preprocessed frames
            try:
                frames = torch.load(file_path)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                # Return dummy data if loading fails
                frames = torch.zeros(self.frames_per_clip, 3, *self.target_size)
            
            # Cache if enabled
            if self.data_cache is not None:
                self.data_cache[file_path] = frames.clone()
        
        # Apply augmentation if enabled
        if self.augmentation is not None:
            frames = self.augmentation.augment_sequence(frames)
        
        return frames, label

class VideoDataset(Dataset):
    """Dataset class for raw video files."""
    
    def __init__(self,
                 video_paths: List[str],
                 labels: List[int],
                 frames_per_clip: int = 16,
                 target_size: Tuple[int, int] = (224, 224),
                 augment: bool = False):
        """
        Initialize the dataset with raw video files.
        
        Args:
            video_paths: List of video file paths
            labels: List of corresponding labels
            frames_per_clip: Number of frames per video clip
            target_size: Target frame size
            augment: Whether to apply data augmentation
        """
        self.video_paths = video_paths
        self.labels = labels
        self.frames_per_clip = frames_per_clip
        self.target_size = target_size
        self.augment = augment
        
        # Initialize components
        self.extractor = VideoFrameExtractor(frames_per_clip=frames_per_clip)
        self.preprocessor = VideoPreprocessor(target_size=target_size)
        self.augmentation = VideoAugmentation() if augment else None
        
        assert len(video_paths) == len(labels), "Number of videos and labels must match"
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (frames, label)
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract frames from video
        frames = self.extractor.extract_frames(video_path)
        
        # Preprocess frames
        processed_frames = self.preprocessor.preprocess_frames(frames)
        
        # Apply augmentation if enabled
        if self.augmentation is not None:
            processed_frames = self.augmentation.augment_sequence(processed_frames)
        
        return processed_frames, label

def create_data_loaders(data_dir: str,
                       batch_size: int = 16,
                       frames_per_clip: int = 16,
                       target_size: Tuple[int, int] = (224, 224),
                       num_workers: int = 4,
                       cache_data: bool = False,
                       train_split: float = 0.7,
                       val_split: float = 0.15) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Directory containing preprocessed data
        batch_size: Batch size for training
        frames_per_clip: Number of frames per video clip
        target_size: Target frame size
        num_workers: Number of worker threads for data loading
        cache_data: Whether to cache data in memory
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        
    Returns:
        Dictionary containing train, val, and test data loaders
    """
    # Load all samples
    all_samples = []
    class_to_idx = {"non_violence": 0, "violence": 1}
    
    for class_name, class_idx in class_to_idx.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        pt_files = [f for f in os.listdir(class_dir) if f.endswith('.pt')]
        for pt_file in pt_files:
            file_path = os.path.join(class_dir, pt_file)
            all_samples.append((file_path, class_idx))
    
    # Shuffle samples
    random.shuffle(all_samples)
    
    # Split data
    total_samples = len(all_samples)
    train_end = int(total_samples * train_split)
    val_end = int(total_samples * (train_split + val_split))
    
    train_samples = all_samples[:train_end]
    val_samples = all_samples[train_end:val_end]
    test_samples = all_samples[val_end:]
    
    # Create temporary directories for splits (in-memory split)
    datasets = {}
    
    # Create datasets
    for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        dataset = CustomSplitDataset(
            samples=samples,
            frames_per_clip=frames_per_clip,
            target_size=target_size,
            augment=(split_name == "train"),
            cache_data=cache_data
        )
        datasets[split_name] = dataset
    
    # Create data loaders
    data_loaders = {}
    for split_name, dataset in datasets.items():
        shuffle = (split_name == "train")
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split_name == "train")
        )
        data_loaders[split_name] = data_loader
    
    logger.info(f"Created data loaders: Train={len(datasets['train'])}, "
                f"Val={len(datasets['val'])}, Test={len(datasets['test'])}")
    
    return data_loaders

class CustomSplitDataset(Dataset):
    """Custom dataset for handling pre-split samples."""
    
    def __init__(self,
                 samples: List[Tuple[str, int]],
                 frames_per_clip: int = 16,
                 target_size: Tuple[int, int] = (224, 224),
                 augment: bool = False,
                 cache_data: bool = False):
        """
        Initialize dataset with pre-split samples.
        
        Args:
            samples: List of (file_path, label) tuples
            frames_per_clip: Number of frames per clip
            target_size: Target frame size
            augment: Whether to apply augmentation
            cache_data: Whether to cache data
        """
        self.samples = samples
        self.frames_per_clip = frames_per_clip
        self.target_size = target_size
        self.augment = augment
        self.cache_data = cache_data
        
        # Initialize components
        self.preprocessor = VideoPreprocessor(target_size=target_size)
        self.augmentation = VideoAugmentation() if augment else None
        
        # Cache for loaded data
        self.data_cache = {} if cache_data else None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.samples[idx]
        
        # Check cache first
        if self.data_cache is not None and file_path in self.data_cache:
            frames = self.data_cache[file_path].clone()
        else:
            # Load preprocessed frames
            try:
                frames = torch.load(file_path)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                frames = torch.zeros(self.frames_per_clip, 3, *self.target_size)
            
            # Cache if enabled
            if self.data_cache is not None:
                self.data_cache[file_path] = frames.clone()
        
        # Apply augmentation if enabled
        if self.augmentation is not None:
            frames = self.augmentation.augment_sequence(frames)
        
        return frames, label

def get_dataset_stats(data_loader: DataLoader) -> Dict[str, Any]:
    """
    Compute dataset statistics.
    
    Args:
        data_loader: DataLoader to analyze
        
    Returns:
        Dictionary containing dataset statistics
    """
    total_samples = 0
    class_counts = {}
    
    for _, labels in data_loader:
        total_samples += len(labels)
        for label in labels:
            label_item = label.item()
            class_counts[label_item] = class_counts.get(label_item, 0) + 1
    
    stats = {
        "total_samples": total_samples,
        "class_distribution": class_counts,
        "class_balance": {k: v/total_samples for k, v in class_counts.items()}
    }
    
    return stats

if __name__ == "__main__":
    # Example usage
    data_dir = "/path/to/preprocessed/data"
    
    # Create data loaders
    data_loaders = create_data_loaders(
        data_dir=data_dir,
        batch_size=16,
        frames_per_clip=16
    )
    
    # Print dataset statistics
    for split, loader in data_loaders.items():
        stats = get_dataset_stats(loader)
        print(f"{split.upper()} Dataset Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Class distribution: {stats['class_distribution']}")
        print(f"  Class balance: {stats['class_balance']}")
        print()