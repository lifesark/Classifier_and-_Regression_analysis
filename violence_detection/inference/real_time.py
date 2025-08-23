"""
Real-time violence detection from webcam or video stream.

This module provides functionality for real-time violence detection
using webcam input or video streams.
"""

import torch
import cv2
import numpy as np
import time
from collections import deque
from typing import Dict, Any, Optional, Callable
import threading
import logging
from queue import Queue
import argparse

from .predict import VideoInference
from ..data import VideoPreprocessor

logger = logging.getLogger(__name__)

class RealTimeViolenceDetector:
    """Real-time violence detection from video streams."""
    
    def __init__(self,
                 model_path: str,
                 model_type: str,
                 model_config: Dict[str, Any],
                 frames_per_clip: int = 16,
                 detection_interval: float = 1.0,
                 confidence_threshold: float = 0.5,
                 device: Optional[torch.device] = None):
        """
        Initialize real-time detector.
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: Type of model architecture
            model_config: Model configuration dictionary
            frames_per_clip: Number of frames for each prediction
            detection_interval: Time interval between predictions (seconds)
            confidence_threshold: Confidence threshold for violence detection
            device: Device to run inference on
        """
        self.frames_per_clip = frames_per_clip
        self.detection_interval = detection_interval
        self.confidence_threshold = confidence_threshold
        
        # Initialize inference engine
        self.inference = VideoInference(
            model_path=model_path,
            model_type=model_type,
            model_config=model_config,
            device=device,
            confidence_threshold=confidence_threshold
        )
        
        # Frame buffer for temporal modeling
        self.frame_buffer = deque(maxlen=frames_per_clip * 2)  # Keep extra frames
        
        # Detection state
        self.last_detection_time = 0
        self.current_prediction = None
        self.detection_history = deque(maxlen=10)  # Keep last 10 predictions
        
        # Threading for async processing
        self.processing_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=5)
        self.processing_thread = None
        self.is_processing = False
        
        # Preprocessing
        self.preprocessor = VideoPreprocessor(target_size=(224, 224))
        
        logger.info("Real-time violence detector initialized")
    
    def start_processing_thread(self) -> None:
        """Start the background processing thread."""
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Background processing thread started")
    
    def stop_processing_thread(self) -> None:
        """Stop the background processing thread."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        logger.info("Background processing thread stopped")
    
    def _processing_worker(self) -> None:
        """Worker function for background processing."""
        while self.is_processing:
            try:
                if not self.processing_queue.empty():
                    frames = self.processing_queue.get(timeout=0.1)
                    result = self.inference.predict_video_frames(frames)
                    
                    if not self.result_queue.full():
                        self.result_queue.put(result)
                    
                    self.processing_queue.task_done()
                else:
                    time.sleep(0.01)  # Small sleep to prevent busy waiting
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
    
    def add_frame(self, frame: np.ndarray) -> None:
        """
        Add a new frame to the buffer.
        
        Args:
            frame: RGB frame array with shape (height, width, 3)
        """
        # Resize frame to standard size
        frame_resized = cv2.resize(frame, (224, 224))
        self.frame_buffer.append(frame_resized)
    
    def should_run_detection(self) -> bool:
        """
        Check if detection should be run based on timing and buffer status.
        
        Returns:
            True if detection should be run
        """
        current_time = time.time()
        time_elapsed = current_time - self.last_detection_time
        
        return (len(self.frame_buffer) >= self.frames_per_clip and 
                time_elapsed >= self.detection_interval)
    
    def run_detection(self) -> Optional[Dict[str, Any]]:
        """
        Run violence detection on current frame buffer.
        
        Returns:
            Detection result dictionary or None if not ready
        """
        if not self.should_run_detection():
            return None
        
        # Sample frames from buffer
        frames = self._sample_frames_from_buffer()
        
        if len(frames) < self.frames_per_clip:
            return None
        
        # Run inference
        try:
            result = self.inference.predict_video_frames(np.array(frames))
            self.last_detection_time = time.time()
            self.current_prediction = result
            self.detection_history.append(result)
            
            return result
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return None
    
    def run_detection_async(self) -> None:
        """
        Run detection asynchronously using background thread.
        """
        if not self.should_run_detection():
            return
        
        # Sample frames from buffer
        frames = self._sample_frames_from_buffer()
        
        if len(frames) < self.frames_per_clip:
            return
        
        # Add to processing queue
        if not self.processing_queue.full():
            self.processing_queue.put(np.array(frames))
            self.last_detection_time = time.time()
    
    def get_async_result(self) -> Optional[Dict[str, Any]]:
        """
        Get result from async processing.
        
        Returns:
            Detection result or None if no result available
        """
        if not self.result_queue.empty():
            result = self.result_queue.get()
            self.current_prediction = result
            self.detection_history.append(result)
            return result
        return None
    
    def _sample_frames_from_buffer(self) -> List[np.ndarray]:
        """Sample frames from the frame buffer."""
        if len(self.frame_buffer) < self.frames_per_clip:
            return list(self.frame_buffer)
        
        # Sample uniformly from buffer
        buffer_list = list(self.frame_buffer)
        indices = np.linspace(0, len(buffer_list) - 1, self.frames_per_clip, dtype=int)
        return [buffer_list[i] for i in indices]
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Get current detection status.
        
        Returns:
            Status dictionary
        """
        status = {
            'frames_in_buffer': len(self.frame_buffer),
            'last_detection_time': self.last_detection_time,
            'current_prediction': self.current_prediction,
            'detection_count': len(self.detection_history)
        }
        
        if self.current_prediction:
            status.update({
                'is_violence_detected': self.current_prediction.get('is_violence', False),
                'confidence': self.current_prediction.get('confidence', 0.0),
                'predicted_label': self.current_prediction.get('predicted_label', 'Unknown')
            })
        
        return status
    
    def get_violence_probability_trend(self, window_size: int = 5) -> float:
        """
        Get trend of violence probability over recent detections.
        
        Args:
            window_size: Number of recent detections to consider
            
        Returns:
            Average violence probability
        """
        if not self.detection_history:
            return 0.0
        
        recent_detections = list(self.detection_history)[-window_size:]
        violence_probs = []
        
        for detection in recent_detections:
            if 'probabilities' in detection and 'Violence' in detection['probabilities']:
                violence_probs.append(detection['probabilities']['Violence'])
        
        return np.mean(violence_probs) if violence_probs else 0.0

class WebcamViolenceDetector:
    """Webcam-based real-time violence detection with GUI."""
    
    def __init__(self, detector: RealTimeViolenceDetector):
        """
        Initialize webcam detector.
        
        Args:
            detector: RealTimeViolenceDetector instance
        """
        self.detector = detector
        self.cap = None
        self.is_running = False
        
        # Display settings
        self.window_name = "Real-time Violence Detection"
        self.alert_active = False
        self.alert_start_time = 0
        self.alert_duration = 2.0  # seconds
    
    def start_webcam(self, camera_index: int = 0) -> bool:
        """
        Start webcam capture.
        
        Args:
            camera_index: Camera device index
            
        Returns:
            True if successful
        """
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            logger.error(f"Could not open camera {camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info(f"Webcam started (camera {camera_index})")
        return True
    
    def stop_webcam(self) -> None:
        """Stop webcam capture."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam stopped")
    
    def run(self, camera_index: int = 0, use_async: bool = True) -> None:
        """
        Run real-time detection loop.
        
        Args:
            camera_index: Camera device index
            use_async: Whether to use asynchronous processing
        """
        if not self.start_webcam(camera_index):
            return
        
        if use_async:
            self.detector.start_processing_thread()
        
        self.is_running = True
        fps_counter = 0
        fps_start_time = time.time()
        
        logger.info("Starting real-time detection...")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Add frame to detector
                self.detector.add_frame(frame_rgb)
                
                # Run detection
                if use_async:
                    self.detector.run_detection_async()
                    result = self.detector.get_async_result()
                else:
                    result = self.detector.run_detection()
                
                # Update alert status
                if result and result.get('is_violence', False):
                    self.alert_active = True
                    self.alert_start_time = time.time()
                
                # Check if alert should be deactivated
                if self.alert_active:
                    if time.time() - self.alert_start_time > self.alert_duration:
                        self.alert_active = False
                
                # Draw detection results on frame
                self._draw_detection_info(frame, self.detector.get_current_status())
                
                # Display frame
                cv2.imshow(self.window_name, frame)
                
                # FPS calculation
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    logger.info(f"FPS: {fps:.1f}")
                    fps_start_time = time.time()
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):  # 's' for screenshot
                    self._save_screenshot(frame)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            if use_async:
                self.detector.stop_processing_thread()
            self.stop_webcam()
            self.is_running = False
    
    def _draw_detection_info(self, frame: np.ndarray, status: Dict[str, Any]) -> None:
        """
        Draw detection information on frame.
        
        Args:
            frame: Video frame to draw on
            status: Detection status dictionary
        """
        height, width = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Detection status
        if status.get('current_prediction'):
            label = status['predicted_label']
            confidence = status['confidence']
            is_violence = status.get('is_violence_detected', False)
            
            # Color based on prediction
            color = (0, 0, 255) if is_violence else (0, 255, 0)  # Red for violence, green for non-violence
            
            # Main prediction text
            text = f"Prediction: {label}"
            cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Confidence text
            conf_text = f"Confidence: {confidence:.3f}"
            cv2.putText(frame, conf_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Violence trend
            trend = self.detector.get_violence_probability_trend()
            trend_text = f"Violence Trend: {trend:.3f}"
            cv2.putText(frame, trend_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        else:
            cv2.putText(frame, "Initializing...", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Buffer status
        buffer_text = f"Buffer: {status['frames_in_buffer']}/{self.detector.frames_per_clip}"
        cv2.putText(frame, buffer_text, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Alert indicator
        if self.alert_active:
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 5)
            cv2.putText(frame, "VIOLENCE DETECTED!", (width//2 - 100, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 's' for screenshot", 
                   (10, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _save_screenshot(self, frame: np.ndarray) -> None:
        """Save screenshot with timestamp."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"violence_detection_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        logger.info(f"Screenshot saved: {filename}")

def main():
    """Main function for real-time detection."""
    parser = argparse.ArgumentParser(description="Real-time violence detection")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default="cnn_lstm",
                       help="Model type")
    parser.add_argument("--config_path", type=str,
                       help="Path to model configuration file")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device index")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                       help="Confidence threshold")
    parser.add_argument("--detection_interval", type=float, default=1.0,
                       help="Detection interval in seconds")
    parser.add_argument("--frames_per_clip", type=int, default=16,
                       help="Number of frames per clip")
    parser.add_argument("--no_async", action="store_true",
                       help="Disable asynchronous processing")
    
    args = parser.parse_args()
    
    # Load model configuration
    if args.config_path:
        import json
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
    
    # Create detector
    detector = RealTimeViolenceDetector(
        model_path=args.model_path,
        model_type=args.model_type,
        model_config=model_config,
        frames_per_clip=args.frames_per_clip,
        detection_interval=args.detection_interval,
        confidence_threshold=args.confidence_threshold
    )
    
    # Create webcam detector
    webcam_detector = WebcamViolenceDetector(detector)
    
    # Run detection
    print("Starting real-time violence detection...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    try:
        webcam_detector.run(
            camera_index=args.camera,
            use_async=not args.no_async
        )
    except KeyboardInterrupt:
        print("\nDetection stopped by user")

if __name__ == "__main__":
    main()