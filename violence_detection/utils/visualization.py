"""
Utility functions for violence detection system.

This module provides various utility functions for visualization,
logging, metrics, and other supporting operations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import logging
import os
import json
from typing import List, Dict, Any, Optional, Tuple
import torch
from datetime import datetime

# Set up matplotlib style
plt.style.use('default')
sns.set_palette("husl")

class Logger:
    """Custom logger for the violence detection system."""
    
    def __init__(self, name: str, log_file: Optional[str] = None, level: str = "INFO"):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_file: Optional log file path
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        
        # Set level
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        self.logger.setLevel(level_map.get(level.upper(), logging.INFO))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)

def visualize_training_progress(history: List[Dict[str, float]], 
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize training progress from history.
    
    Args:
        history: List of dictionaries containing training metrics
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    epochs = list(range(1, len(history) + 1))
    
    # Extract metrics
    train_loss = [h.get('train_loss', 0) for h in history]
    val_loss = [h.get('val_loss', 0) for h in history]
    train_acc = [h.get('train_accuracy', 0) for h in history]
    val_acc = [h.get('val_accuracy', 0) for h in history]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plots
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    if any(val_loss):
        ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plots
    ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    if any(val_acc):
        ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if 'learning_rate' in history[0]:
        learning_rates = [h.get('learning_rate', 0) for h in history]
        ax3.plot(epochs, learning_rates, 'g-', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    else:
        ax3.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=12)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
    
    # F1 score (if available)
    if 'val_f1' in history[0]:
        val_f1 = [h.get('val_f1', 0) for h in history]
        ax4.plot(epochs, val_f1, 'm-', linewidth=2)
        ax4.set_title('Validation F1 Score', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('F1 Score')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'F1 Score\nNot Available', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_video_frames(frames: np.ndarray, 
                          title: str = "Video Frames",
                          max_frames: int = 16,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize video frames in a grid.
    
    Args:
        frames: Video frames array with shape (frames, height, width, channels)
        title: Plot title
        max_frames: Maximum number of frames to display
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    num_frames = min(len(frames), max_frames)
    
    # Calculate grid size
    cols = 4
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(rows * cols):
        if i < num_frames:
            frame = frames[i]
            if frame.shape[-1] == 3:  # RGB
                axes[i].imshow(frame)
            else:  # Grayscale
                axes[i].imshow(frame, cmap='gray')
            axes[i].set_title(f'Frame {i+1}')
            axes[i].axis('off')
        else:
            axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_video_from_frames(frames: List[np.ndarray], 
                           output_path: str,
                           fps: int = 10,
                           codec: str = 'mp4v') -> None:
    """
    Create video from frames.
    
    Args:
        frames: List of frames (BGR format)
        output_path: Output video path
        fps: Frames per second
        codec: Video codec
    """
    if not frames:
        raise ValueError("No frames provided")
    
    height, width = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        writer.write(frame)
    
    writer.release()

def calculate_dataset_statistics(data_loader) -> Dict[str, Any]:
    """
    Calculate comprehensive dataset statistics.
    
    Args:
        data_loader: PyTorch DataLoader
        
    Returns:
        Dictionary containing statistics
    """
    total_samples = 0
    class_counts = {}
    frame_shapes = []
    
    for videos, labels in data_loader:
        batch_size = videos.shape[0]
        total_samples += batch_size
        
        # Count classes
        for label in labels:
            label_item = label.item()
            class_counts[label_item] = class_counts.get(label_item, 0) + 1
        
        # Collect frame shapes
        frame_shapes.append(videos.shape[1:])  # (frames, channels, height, width)
    
    # Calculate statistics
    stats = {
        'total_samples': total_samples,
        'num_classes': len(class_counts),
        'class_distribution': class_counts,
        'class_balance': {k: v/total_samples for k, v in class_counts.items()},
        'frame_shapes': frame_shapes[0] if frame_shapes else None
    }
    
    return stats

def save_experiment_config(config: Dict[str, Any], 
                         save_dir: str,
                         experiment_name: str) -> str:
    """
    Save experiment configuration with timestamp.
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save configuration
        experiment_name: Name of the experiment
        
    Returns:
        Path to saved configuration file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_filename = f"{experiment_name}_{timestamp}_config.json"
    config_path = os.path.join(save_dir, config_filename)
    
    os.makedirs(save_dir, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    return config_path

def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def create_model_comparison_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Create a formatted table comparing model results.
    
    Args:
        results: Dictionary with model names as keys and metrics as values
        
    Returns:
        Formatted table string
    """
    if not results:
        return "No results to display"
    
    # Get all metrics
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    
    # Sort metrics
    metric_order = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr']
    sorted_metrics = [m for m in metric_order if m in all_metrics]
    sorted_metrics.extend([m for m in all_metrics if m not in metric_order])
    
    # Create table
    table_lines = []
    
    # Header
    header = "| Model | " + " | ".join([m.capitalize() for m in sorted_metrics]) + " |"
    separator = "|" + "|".join(["---"] * (len(sorted_metrics) + 1)) + "|"
    
    table_lines.append(header)
    table_lines.append(separator)
    
    # Data rows
    for model_name, model_results in results.items():
        row_data = [model_name]
        for metric in sorted_metrics:
            value = model_results.get(metric, 0.0)
            if isinstance(value, float):
                row_data.append(f"{value:.4f}")
            else:
                row_data.append(str(value))
        
        row = "| " + " | ".join(row_data) + " |"
        table_lines.append(row)
    
    return "\n".join(table_lines)

def setup_colab_environment():
    """Set up Google Colab environment for violence detection."""
    try:
        # Mount Google Drive
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted")
        
        # Install additional packages
        import subprocess
        import sys
        
        packages = [
            'decord',
            'matplotlib',
            'seaborn',
            'scikit-learn',
            'tqdm',
            'wandb'
        ]
        
        for package in packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✓ {package} installed")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}")
        
        # Set up directories
        os.makedirs('/content/violence_detection_workspace', exist_ok=True)
        os.makedirs('/content/drive/MyDrive/violence_detection_data', exist_ok=True)
        os.makedirs('/content/drive/MyDrive/violence_detection_models', exist_ok=True)
        
        print("✓ Directories created")
        
        # Set environment variables
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        print("✓ Colab environment setup complete")
        
        return True
        
    except Exception as e:
        print(f"✗ Error setting up Colab environment: {e}")
        return False

def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and specifications.
    
    Returns:
        Dictionary containing GPU information
    """
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'device_name': None,
        'memory_total': None,
        'memory_used': None,
        'memory_free': None
    }
    
    if torch.cuda.is_available():
        gpu_info['current_device'] = torch.cuda.current_device()
        gpu_info['device_name'] = torch.cuda.get_device_name()
        
        # Memory information
        memory_total = torch.cuda.get_device_properties(0).total_memory
        memory_used = torch.cuda.memory_allocated()
        memory_free = memory_total - memory_used
        
        gpu_info['memory_total'] = memory_total / 1e9  # GB
        gpu_info['memory_used'] = memory_used / 1e9    # GB
        gpu_info['memory_free'] = memory_free / 1e9    # GB
    
    return gpu_info

def print_system_info():
    """Print comprehensive system information."""
    import platform
    import psutil
    
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    # Basic system info
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    
    # CPU info
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"RAM Total: {memory.total / 1e9:.1f} GB")
    print(f"RAM Available: {memory.available / 1e9:.1f} GB")
    print(f"RAM Usage: {memory.percent}%")
    
    # GPU info
    gpu_info = check_gpu_availability()
    print(f"CUDA Available: {gpu_info['cuda_available']}")
    if gpu_info['cuda_available']:
        print(f"GPU Count: {gpu_info['device_count']}")
        print(f"GPU Name: {gpu_info['device_name']}")
        print(f"GPU Memory Total: {gpu_info['memory_total']:.1f} GB")
        print(f"GPU Memory Free: {gpu_info['memory_free']:.1f} GB")
    
    print("=" * 60)

if __name__ == "__main__":
    # Example usage
    print_system_info()
    
    # Test logger
    logger = Logger("test_logger", log_file="test.log")
    logger.info("This is a test message")
    logger.warning("This is a warning")
    
    # Test visualization
    dummy_history = [
        {'train_loss': 1.0, 'val_loss': 0.9, 'train_accuracy': 60, 'val_accuracy': 65},
        {'train_loss': 0.8, 'val_loss': 0.7, 'train_accuracy': 70, 'val_accuracy': 75},
        {'train_loss': 0.6, 'val_loss': 0.6, 'train_accuracy': 80, 'val_accuracy': 82},
    ]
    
    fig = visualize_training_progress(dummy_history)
    plt.show()
    
    # Test model comparison table
    dummy_results = {
        'CNN+LSTM': {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.87, 'f1': 0.85},
        'C3D': {'accuracy': 0.82, 'precision': 0.80, 'recall': 0.84, 'f1': 0.82},
        'TimeSformer': {'accuracy': 0.88, 'precision': 0.86, 'recall': 0.90, 'f1': 0.88}
    }
    
    table = create_model_comparison_table(dummy_results)
    print(table)