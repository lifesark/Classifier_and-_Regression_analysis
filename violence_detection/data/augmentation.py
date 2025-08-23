"""
Data augmentation utilities for violence detection.

This module provides comprehensive data augmentation techniques
specifically designed for video data in violence detection tasks.
"""

import torch
import torchvision.transforms as T
import random
import numpy as np
from typing import Tuple, List, Optional
import cv2

class TemporalAugmentation:
    """Temporal augmentation techniques for video sequences."""
    
    def __init__(self, 
                 temporal_jitter: bool = True,
                 frame_skip: bool = True,
                 reverse_sequence: bool = True):
        """
        Initialize temporal augmentation.
        
        Args:
            temporal_jitter: Apply temporal jittering
            frame_skip: Apply random frame skipping
            reverse_sequence: Apply sequence reversal
        """
        self.temporal_jitter = temporal_jitter
        self.frame_skip = frame_skip
        self.reverse_sequence = reverse_sequence
    
    def apply_temporal_jitter(self, frames: torch.Tensor, max_shift: int = 2) -> torch.Tensor:
        """Apply temporal jittering by shifting frames."""
        if not self.temporal_jitter or random.random() > 0.5:
            return frames
            
        shift = random.randint(-max_shift, max_shift)
        if shift != 0:
            frames = torch.roll(frames, shift, dims=0)
        return frames
    
    def apply_frame_skip(self, frames: torch.Tensor, skip_prob: float = 0.1) -> torch.Tensor:
        """Apply random frame skipping."""
        if not self.frame_skip or random.random() > 0.3:
            return frames
            
        num_frames = frames.shape[0]
        keep_mask = torch.rand(num_frames) > skip_prob
        
        # Ensure we keep at least half the frames
        if keep_mask.sum() < num_frames // 2:
            return frames
            
        # Interpolate to maintain sequence length
        kept_frames = frames[keep_mask]
        if len(kept_frames) < num_frames:
            # Simple duplication to maintain length
            indices = torch.linspace(0, len(kept_frames) - 1, num_frames).long()
            frames = kept_frames[indices]
            
        return frames
    
    def apply_reverse_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply sequence reversal."""
        if self.reverse_sequence and random.random() > 0.8:
            frames = torch.flip(frames, dims=[0])
        return frames
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply all temporal augmentations."""
        frames = self.apply_temporal_jitter(frames)
        frames = self.apply_frame_skip(frames)
        frames = self.apply_reverse_sequence(frames)
        return frames

class SpatialAugmentation:
    """Spatial augmentation techniques for video frames."""
    
    def __init__(self,
                 random_crop: bool = True,
                 random_resize: bool = True,
                 horizontal_flip: bool = True,
                 rotation: bool = True,
                 color_jitter: bool = True,
                 gaussian_blur: bool = True):
        """
        Initialize spatial augmentation.
        
        Args:
            random_crop: Apply random cropping
            random_resize: Apply random resizing
            horizontal_flip: Apply horizontal flipping
            rotation: Apply random rotation
            color_jitter: Apply color jittering
            gaussian_blur: Apply gaussian blur
        """
        self.random_crop = random_crop
        self.random_resize = random_resize
        self.horizontal_flip = horizontal_flip
        self.rotation = rotation
        self.color_jitter = color_jitter
        self.gaussian_blur = gaussian_blur
        
        # Define transforms
        self.color_jitter_transform = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
        self.gaussian_blur_transform = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    
    def apply_random_crop(self, frames: torch.Tensor, crop_ratio: float = 0.8) -> torch.Tensor:
        """Apply random cropping to frames."""
        if not self.random_crop or random.random() > 0.5:
            return frames
            
        _, _, h, w = frames.shape
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        
        frames = frames[:, :, top:top+crop_h, left:left+crop_w]
        
        # Resize back to original size
        frames = torch.nn.functional.interpolate(
            frames, size=(h, w), mode='bilinear', align_corners=False
        )
        
        return frames
    
    def apply_random_resize(self, frames: torch.Tensor, scale_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        """Apply random resizing."""
        if not self.random_resize or random.random() > 0.3:
            return frames
            
        scale = random.uniform(*scale_range)
        _, _, h, w = frames.shape
        new_h, new_w = int(h * scale), int(w * scale)
        
        frames = torch.nn.functional.interpolate(
            frames, size=(new_h, new_w), mode='bilinear', align_corners=False
        )
        
        # Resize back to original size
        frames = torch.nn.functional.interpolate(
            frames, size=(h, w), mode='bilinear', align_corners=False
        )
        
        return frames
    
    def apply_horizontal_flip(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply horizontal flipping."""
        if self.horizontal_flip and random.random() > 0.5:
            frames = torch.flip(frames, dims=[3])
        return frames
    
    def apply_rotation(self, frames: torch.Tensor, max_angle: float = 10.0) -> torch.Tensor:
        """Apply random rotation."""
        if not self.rotation or random.random() > 0.3:
            return frames
            
        angle = random.uniform(-max_angle, max_angle)
        
        # Apply rotation to each frame
        rotated_frames = []
        for frame in frames:
            # Convert to PIL format for rotation
            frame_pil = T.ToPILImage()(frame)
            rotated_frame = T.functional.rotate(frame_pil, angle)
            rotated_frames.append(T.ToTensor()(rotated_frame))
        
        return torch.stack(rotated_frames)
    
    def apply_color_jitter(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply color jittering."""
        if not self.color_jitter or random.random() > 0.5:
            return frames
            
        jittered_frames = []
        for frame in frames:
            jittered_frame = self.color_jitter_transform(frame)
            jittered_frames.append(jittered_frame)
        
        return torch.stack(jittered_frames)
    
    def apply_gaussian_blur(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply gaussian blur."""
        if not self.gaussian_blur or random.random() > 0.2:
            return frames
            
        blurred_frames = []
        for frame in frames:
            blurred_frame = self.gaussian_blur_transform(frame)
            blurred_frames.append(blurred_frame)
        
        return torch.stack(blurred_frames)
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply all spatial augmentations."""
        frames = self.apply_random_crop(frames)
        frames = self.apply_random_resize(frames)
        frames = self.apply_horizontal_flip(frames)
        frames = self.apply_rotation(frames)
        frames = self.apply_color_jitter(frames)
        frames = self.apply_gaussian_blur(frames)
        return frames

class AdvancedVideoAugmentation:
    """Advanced video augmentation combining temporal and spatial techniques."""
    
    def __init__(self,
                 temporal_aug: bool = True,
                 spatial_aug: bool = True,
                 mixup: bool = False,
                 cutmix: bool = False,
                 augmentation_prob: float = 0.8):
        """
        Initialize advanced video augmentation.
        
        Args:
            temporal_aug: Enable temporal augmentation
            spatial_aug: Enable spatial augmentation
            mixup: Enable MixUp augmentation
            cutmix: Enable CutMix augmentation
            augmentation_prob: Probability of applying augmentation
        """
        self.temporal_aug = TemporalAugmentation() if temporal_aug else None
        self.spatial_aug = SpatialAugmentation() if spatial_aug else None
        self.mixup = mixup
        self.cutmix = cutmix
        self.augmentation_prob = augmentation_prob
    
    def apply_mixup(self, frames1: torch.Tensor, frames2: torch.Tensor, 
                   label1: int, label2: int, alpha: float = 0.2) -> Tuple[torch.Tensor, float]:
        """Apply MixUp augmentation."""
        if not self.mixup or random.random() > 0.1:
            return frames1, label1
            
        lam = np.random.beta(alpha, alpha)
        mixed_frames = lam * frames1 + (1 - lam) * frames2
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_frames, mixed_label
    
    def apply_cutmix(self, frames1: torch.Tensor, frames2: torch.Tensor,
                    label1: int, label2: int, alpha: float = 1.0) -> Tuple[torch.Tensor, float]:
        """Apply CutMix augmentation."""
        if not self.cutmix or random.random() > 0.1:
            return frames1, label1
            
        lam = np.random.beta(alpha, alpha)
        
        # Get frame dimensions
        _, _, h, w = frames1.shape
        
        # Generate random box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cutmix
        mixed_frames = frames1.clone()
        mixed_frames[:, :, bby1:bby2, bbx1:bbx2] = frames2[:, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_frames, mixed_label
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply augmentation pipeline."""
        if random.random() > self.augmentation_prob:
            return frames
            
        # Apply temporal augmentation
        if self.temporal_aug is not None:
            frames = self.temporal_aug(frames)
        
        # Apply spatial augmentation
        if self.spatial_aug is not None:
            frames = self.spatial_aug(frames)
        
        return frames

class VideoNormalization:
    """Video-specific normalization techniques."""
    
    def __init__(self,
                 per_frame_norm: bool = True,
                 temporal_norm: bool = False,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225]):
        """
        Initialize video normalization.
        
        Args:
            per_frame_norm: Apply per-frame normalization
            temporal_norm: Apply temporal normalization
            mean: Normalization mean values
            std: Normalization standard deviation values
        """
        self.per_frame_norm = per_frame_norm
        self.temporal_norm = temporal_norm
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def apply_per_frame_normalization(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply per-frame normalization."""
        if not self.per_frame_norm:
            return frames
            
        normalized_frames = []
        for frame in frames:
            frame = frame.unsqueeze(0)  # Add batch dimension
            normalized_frame = (frame - self.mean) / self.std
            normalized_frames.append(normalized_frame.squeeze(0))
        
        return torch.stack(normalized_frames)
    
    def apply_temporal_normalization(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply temporal normalization across the sequence."""
        if not self.temporal_norm:
            return frames
            
        # Compute temporal statistics
        temporal_mean = frames.mean(dim=0, keepdim=True)
        temporal_std = frames.std(dim=0, keepdim=True) + 1e-8
        
        # Normalize
        frames = (frames - temporal_mean) / temporal_std
        
        return frames
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply normalization."""
        frames = self.apply_per_frame_normalization(frames)
        frames = self.apply_temporal_normalization(frames)
        return frames

# Convenience class combining all augmentation techniques
class VideoAugmentation:
    """Comprehensive video augmentation pipeline."""
    
    def __init__(self,
                 temporal_augmentation: bool = True,
                 spatial_augmentation: bool = True,
                 advanced_augmentation: bool = False,
                 normalization: bool = True,
                 augmentation_prob: float = 0.8):
        """
        Initialize comprehensive video augmentation.
        
        Args:
            temporal_augmentation: Enable temporal augmentation
            spatial_augmentation: Enable spatial augmentation
            advanced_augmentation: Enable advanced augmentation techniques
            normalization: Enable video normalization
            augmentation_prob: Overall augmentation probability
        """
        self.temporal_aug = TemporalAugmentation() if temporal_augmentation else None
        self.spatial_aug = SpatialAugmentation() if spatial_augmentation else None
        self.advanced_aug = AdvancedVideoAugmentation() if advanced_augmentation else None
        self.normalization = VideoNormalization() if normalization else None
        self.augmentation_prob = augmentation_prob
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply complete augmentation pipeline."""
        if random.random() > self.augmentation_prob:
            if self.normalization is not None:
                frames = self.normalization(frames)
            return frames
        
        # Apply augmentations
        if self.temporal_aug is not None:
            frames = self.temporal_aug(frames)
        
        if self.spatial_aug is not None:
            frames = self.spatial_aug(frames)
        
        if self.advanced_aug is not None:
            frames = self.advanced_aug(frames)
        
        # Apply normalization last
        if self.normalization is not None:
            frames = self.normalization(frames)
        
        return frames

if __name__ == "__main__":
    # Example usage
    import torch
    
    # Create dummy video frames (16 frames, 3 channels, 224x224)
    frames = torch.rand(16, 3, 224, 224)
    
    # Initialize augmentation
    augmentation = VideoAugmentation()
    
    # Apply augmentation
    augmented_frames = augmentation(frames)
    
    print(f"Original shape: {frames.shape}")
    print(f"Augmented shape: {augmented_frames.shape}")
    print(f"Value range: [{augmented_frames.min():.3f}, {augmented_frames.max():.3f}]")