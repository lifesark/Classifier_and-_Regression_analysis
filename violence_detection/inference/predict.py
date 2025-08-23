"""
Single video inference for violence detection.

This module provides functionality to predict violence in individual
video files using trained models.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from typing import Dict, Any, List, Tuple, Optional
import logging
import time
from pathlib import Path

# Import local modules
from ..models import ModelFactory, load_model_checkpoint
from ..data import VideoFrameExtractor, VideoPreprocessor

logger = logging.getLogger(__name__)

class VideoInference:
    """Single video inference class for violence detection."""
    
    def __init__(self, 
                 model_path: str,
                 model_type: str,
                 model_config: Dict[str, Any],
                 device: Optional[torch.device] = None,
                 confidence_threshold: float = 0.5):
        """
        Initialize video inference.
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: Type of model architecture
            model_config: Model configuration dictionary
            device: Device to run inference on
            confidence_threshold: Confidence threshold for classification
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model_config = model_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.model = None
        self.extractor = None
        self.preprocessor = None
        
        # Class labels
        self.class_labels = ["Non-Violence", "Violence"]
        
        # Load model
        self._load_model()
        self._setup_preprocessing()
        
        logger.info(f"VideoInference initialized with {model_type} model on {self.device}")
    
    def _load_model(self) -> None:
        """Load the trained model."""
        # Create model
        self.model = ModelFactory.create_model(self.model_type, self.model_config)
        
        # Load checkpoint
        checkpoint_info = load_model_checkpoint(
            model=self.model,
            checkpoint_path=self.model_path,
            device=self.device
        )
        
        # Move to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded model from {self.model_path}")
        logger.info(f"Model trained for {checkpoint_info['epoch']} epochs")
        logger.info(f"Best validation metrics: {checkpoint_info['metrics']}")
    
    def _setup_preprocessing(self) -> None:
        """Set up video preprocessing components."""
        frames_per_clip = self.model_config.get('frames_per_clip', 16)
        
        self.extractor = VideoFrameExtractor(frames_per_clip=frames_per_clip)
        self.preprocessor = VideoPreprocessor(target_size=(224, 224))
        
        logger.info("Video preprocessing components initialized")
    
    def predict_video(self, video_path: str) -> Dict[str, Any]:
        """
        Predict violence in a single video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing prediction results
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        start_time = time.time()
        
        try:
            # Extract frames
            frames = self.extractor.extract_frames(video_path)
            
            # Preprocess frames
            processed_frames = self.preprocessor.preprocess_frames(frames)
            
            # Add batch dimension
            input_tensor = processed_frames.unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Prepare results
            prediction_time = time.time() - start_time
            
            results = {
                'video_path': video_path,
                'predicted_class': predicted_class,
                'predicted_label': self.class_labels[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    self.class_labels[i]: probabilities[0][i].item() 
                    for i in range(len(self.class_labels))
                },
                'is_violence': predicted_class == 1 and confidence >= self.confidence_threshold,
                'prediction_time': prediction_time,
                'frames_processed': len(frames)
            }
            
            logger.info(f"Prediction completed for {video_path}")
            logger.info(f"Result: {results['predicted_label']} (confidence: {confidence:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return {
                'video_path': video_path,
                'error': str(e),
                'prediction_time': time.time() - start_time
            }
    
    def predict_batch(self, video_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Predict violence in multiple videos.
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        logger.info(f"Processing batch of {len(video_paths)} videos")
        
        for i, video_path in enumerate(video_paths):
            logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
            result = self.predict_video(video_path)
            results.append(result)
        
        return results
    
    def predict_video_frames(self, frames: np.ndarray) -> Dict[str, Any]:
        """
        Predict violence from preprocessed video frames.
        
        Args:
            frames: Video frames array with shape (frames, height, width, channels)
            
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        try:
            # Preprocess frames
            processed_frames = self.preprocessor.preprocess_frames(frames)
            
            # Add batch dimension
            input_tensor = processed_frames.unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Prepare results
            prediction_time = time.time() - start_time
            
            results = {
                'predicted_class': predicted_class,
                'predicted_label': self.class_labels[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    self.class_labels[i]: probabilities[0][i].item() 
                    for i in range(len(self.class_labels))
                },
                'is_violence': predicted_class == 1 and confidence >= self.confidence_threshold,
                'prediction_time': prediction_time,
                'frames_processed': len(frames)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing frames: {str(e)}")
            return {
                'error': str(e),
                'prediction_time': time.time() - start_time
            }
    
    def analyze_video_segments(self, 
                             video_path: str, 
                             segment_duration: float = 5.0,
                             overlap: float = 0.5) -> List[Dict[str, Any]]:
        """
        Analyze video in segments for temporal analysis.
        
        Args:
            video_path: Path to the video file
            segment_duration: Duration of each segment in seconds
            overlap: Overlap between segments (0-1)
            
        Returns:
            List of results for each segment
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        segment_frames = int(segment_duration * fps)
        step_frames = int(segment_frames * (1 - overlap))
        
        results = []
        
        logger.info(f"Analyzing video segments: {video_path}")
        logger.info(f"Total duration: {duration:.2f}s, Segment duration: {segment_duration}s")
        
        for start_frame in range(0, total_frames - segment_frames, step_frames):
            end_frame = min(start_frame + segment_frames, total_frames)
            start_time = start_frame / fps
            end_time = end_frame / fps
            
            # Extract segment frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if len(frames) < self.extractor.frames_per_clip:
                continue
            
            # Sample frames to match model requirements
            sampled_frames = self._sample_frames(frames, self.extractor.frames_per_clip)
            
            # Predict
            result = self.predict_video_frames(np.array(sampled_frames))
            result.update({
                'segment_start': start_time,
                'segment_end': end_time,
                'segment_duration': end_time - start_time
            })
            
            results.append(result)
            
            logger.info(f"Segment {start_time:.1f}-{end_time:.1f}s: "
                       f"{result['predicted_label']} ({result['confidence']:.3f})")
        
        cap.release()
        
        return results
    
    def _sample_frames(self, frames: List[np.ndarray], target_count: int) -> List[np.ndarray]:
        """Sample frames to match target count."""
        if len(frames) <= target_count:
            return frames
        
        # Uniform sampling
        indices = np.linspace(0, len(frames) - 1, target_count, dtype=int)
        return [frames[i] for i in indices]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'class_labels': self.class_labels,
            'confidence_threshold': self.confidence_threshold
        }

def create_inference_from_checkpoint(checkpoint_path: str,
                                   device: Optional[torch.device] = None) -> VideoInference:
    """
    Create VideoInference instance from checkpoint file.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to run inference on
        
    Returns:
        VideoInference instance
    """
    # Load checkpoint to get model config
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_config' not in checkpoint:
        raise ValueError("Checkpoint does not contain model configuration")
    
    model_config = checkpoint['model_config']
    
    # Infer model type from config or checkpoint
    if 'cnn_backbone' in model_config:
        model_type = 'cnn_lstm'
    elif 'embed_dim' in model_config and 'patch_size' in model_config:
        model_type = 'timesformer'
    elif 'depths' in model_config:
        model_type = 'video_swin'
    else:
        # Default to C3D for other cases
        model_type = 'c3d'
    
    return VideoInference(
        model_path=checkpoint_path,
        model_type=model_type,
        model_config=model_config,
        device=device
    )

def main():
    """Main function for command-line inference."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Violence detection inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--video_path", type=str, required=True,
                       help="Path to video file")
    parser.add_argument("--model_type", type=str, default="cnn_lstm",
                       help="Model type")
    parser.add_argument("--config_path", type=str,
                       help="Path to model configuration file")
    parser.add_argument("--output_path", type=str,
                       help="Path to save results")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                       help="Confidence threshold")
    parser.add_argument("--analyze_segments", action="store_true",
                       help="Analyze video in segments")
    parser.add_argument("--segment_duration", type=float, default=5.0,
                       help="Segment duration in seconds")
    
    args = parser.parse_args()
    
    # Load model configuration
    if args.config_path:
        with open(args.config_path, 'r') as f:
            model_config = json.load(f)
    else:
        # Use default configuration
        model_config = {
            "num_classes": 2,
            "cnn_backbone": "resnet18",
            "cnn_feature_dim": 512,
            "lstm_hidden_dim": 256,
            "lstm_num_layers": 2,
            "use_attention": True,
            "dropout_rate": 0.3
        }
    
    # Create inference instance
    inference = VideoInference(
        model_path=args.model_path,
        model_type=args.model_type,
        model_config=model_config,
        confidence_threshold=args.confidence_threshold
    )
    
    # Print model info
    model_info = inference.get_model_info()
    print("Model Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Perform inference
    if args.analyze_segments:
        results = inference.analyze_video_segments(
            args.video_path,
            segment_duration=args.segment_duration
        )
        print(f"Analyzed {len(results)} segments")
        for i, result in enumerate(results):
            print(f"Segment {i+1}: {result['segment_start']:.1f}-{result['segment_end']:.1f}s "
                  f"-> {result['predicted_label']} ({result['confidence']:.3f})")
    else:
        result = inference.predict_video(args.video_path)
        print("Prediction Results:")
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"  Video: {result['video_path']}")
            print(f"  Prediction: {result['predicted_label']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Is Violence: {result['is_violence']}")
            print(f"  Processing Time: {result['prediction_time']:.3f}s")
            print(f"  Frames Processed: {result['frames_processed']}")
            
            print("\nProbabilities:")
            for label, prob in result['probabilities'].items():
                print(f"  {label}: {prob:.3f}")
    
    # Save results if output path is specified
    if args.output_path:
        if args.analyze_segments:
            output_data = {'segments': results}
        else:
            output_data = result
            
        with open(args.output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output_path}")

if __name__ == "__main__":
    main()