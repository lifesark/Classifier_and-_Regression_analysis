"""
Data preprocessing utilities for violence detection.

This module handles video frame extraction, preprocessing, and data augmentation
for the violence detection dataset.
"""

import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import decord
from PIL import Image
import random
from typing import List, Tuple, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoFrameExtractor:
    """Extract frames from video files efficiently."""
    
    def __init__(self, frames_per_clip: int = 16, target_fps: Optional[int] = None):
        """
        Initialize the frame extractor.
        
        Args:
            frames_per_clip: Number of frames to extract per video clip
            target_fps: Target FPS for frame extraction (None for original FPS)
        """
        self.frames_per_clip = frames_per_clip
        self.target_fps = target_fps
        
    def extract_frames(self, video_path: str, output_dir: Optional[str] = None) -> np.ndarray:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames (optional)
            
        Returns:
            Array of extracted frames with shape (frames, height, width, channels)
        """
        try:
            # Use decord for efficient video reading
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)
            
            # Calculate frame indices to extract
            if total_frames <= self.frames_per_clip:
                # If video has fewer frames than required, repeat frames
                indices = np.linspace(0, total_frames - 1, self.frames_per_clip, dtype=int)
            else:
                # Uniformly sample frames
                indices = np.linspace(0, total_frames - 1, self.frames_per_clip, dtype=int)
            
            # Extract frames
            frames = vr.get_batch(indices).asnumpy()
            
            # Save frames if output directory is specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            # Return dummy frames if extraction fails
            return np.zeros((self.frames_per_clip, 224, 224, 3), dtype=np.uint8)

class VideoPreprocessor:
    """Preprocess video frames for model training."""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225]):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target frame size (height, width)
            normalize: Whether to normalize frames
            mean: Normalization mean values
            std: Normalization standard deviation values
        """
        self.target_size = target_size
        self.normalize = normalize
        self.mean = mean
        self.std = std
        
        # Define preprocessing transforms
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor()
        ]
        
        if normalize:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
            
        self.transform = transforms.Compose(transform_list)
        
    def preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        Preprocess a sequence of frames.
        
        Args:
            frames: Input frames with shape (frames, height, width, channels)
            
        Returns:
            Preprocessed frames as tensor with shape (frames, channels, height, width)
        """
        preprocessed_frames = []
        
        for frame in frames:
            # Convert to PIL Image and apply transforms
            processed_frame = self.transform(frame)
            preprocessed_frames.append(processed_frame)
        
        # Stack frames into a single tensor
        return torch.stack(preprocessed_frames)

class VideoAugmentation:
    """Data augmentation for video sequences."""
    
    def __init__(self, 
                 random_crop: bool = True,
                 horizontal_flip: bool = True,
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 temporal_shift: bool = True):
        """
        Initialize augmentation parameters.
        
        Args:
            random_crop: Apply random cropping
            horizontal_flip: Apply random horizontal flipping
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            temporal_shift: Apply temporal shifting
        """
        self.random_crop = random_crop
        self.horizontal_flip = horizontal_flip
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.temporal_shift = temporal_shift
        
    def augment_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to a video sequence.
        
        Args:
            frames: Input frames tensor with shape (frames, channels, height, width)
            
        Returns:
            Augmented frames tensor
        """
        # Random horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            frames = torch.flip(frames, dims=[3])  # Flip width dimension
        
        # Random brightness and contrast adjustment
        if random.random() > 0.5:
            brightness_factor = random.uniform(*self.brightness_range)
            frames = frames * brightness_factor
            
        if random.random() > 0.5:
            contrast_factor = random.uniform(*self.contrast_range)
            frames = (frames - 0.5) * contrast_factor + 0.5
        
        # Temporal shifting
        if self.temporal_shift and random.random() > 0.5:
            shift = random.randint(-2, 2)
            if shift != 0:
                frames = torch.roll(frames, shift, dims=0)
        
        # Clamp values to valid range
        frames = torch.clamp(frames, 0, 1)
        
        return frames

def preprocess_godseye_dataset(dataset_path: str, 
                             output_dir: str, 
                             frames_per_clip: int = 16,
                             target_size: Tuple[int, int] = (224, 224)) -> None:
    """
    Preprocess the Godseye Violence Detection Dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        output_dir: Directory to save preprocessed data
        frames_per_clip: Number of frames to extract per video
        target_size: Target frame size
    """
    logger.info("Starting Godseye dataset preprocessing...")
    
    extractor = VideoFrameExtractor(frames_per_clip=frames_per_clip)
    preprocessor = VideoPreprocessor(target_size=target_size)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "violence"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "non_violence"), exist_ok=True)
    
    # Process videos in each category
    categories = ["violence", "non_violence"]
    
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if not os.path.exists(category_path):
            logger.warning(f"Category directory not found: {category_path}")
            continue
            
        video_files = [f for f in os.listdir(category_path) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        logger.info(f"Processing {len(video_files)} videos in {category} category...")
        
        for i, video_file in enumerate(video_files):
            video_path = os.path.join(category_path, video_file)
            
            # Extract frames
            frames = extractor.extract_frames(video_path)
            
            # Preprocess frames
            processed_frames = preprocessor.preprocess_frames(frames)
            
            # Save preprocessed data
            output_path = os.path.join(output_dir, category, f"{os.path.splitext(video_file)[0]}.pt")
            torch.save(processed_frames, output_path)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(video_files)} videos in {category}")
    
    logger.info("Dataset preprocessing completed!")

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Godseye Violence Detection Dataset")
    parser.add_argument("--dataset_path", type=str, required=True, 
                       help="Path to the Godseye dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save preprocessed data")
    parser.add_argument("--frames_per_clip", type=int, default=16,
                       help="Number of frames per video clip")
    
    args = parser.parse_args()
    
    preprocess_godseye_dataset(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        frames_per_clip=args.frames_per_clip
    )