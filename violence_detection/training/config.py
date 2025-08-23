"""
Training configuration for violence detection models.

This module contains configuration settings for training different
model architectures on the violence detection task.
"""

import torch
from typing import Dict, Any, List, Tuple
import os

class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(self):
        """Initialize training configuration with default values."""
        
        # Data configuration
        self.data_dir = "/content/godseye_dataset"  # Path to preprocessed data
        self.batch_size = 16
        self.num_workers = 4
        self.frames_per_clip = 16
        self.target_size = (224, 224)
        self.train_split = 0.7
        self.val_split = 0.15
        self.cache_data = False
        
        # Model configuration
        self.model_type = "cnn_lstm"
        self.model_config = {
            "num_classes": 2,
            "cnn_backbone": "resnet18",
            "cnn_feature_dim": 512,
            "lstm_hidden_dim": 256,
            "lstm_num_layers": 2,
            "use_attention": True,
            "dropout_rate": 0.3
        }
        
        # Training configuration
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.optimizer_type = "adam"
        self.scheduler_type = "cosine"
        self.warmup_epochs = 5
        self.gradient_clip_value = 1.0
        
        # Loss configuration
        self.loss_type = "cross_entropy"
        self.class_weights = None  # Auto-compute from data
        self.label_smoothing = 0.1
        
        # Regularization
        self.dropout_rate = 0.3
        self.mixup_alpha = 0.2
        self.cutmix_alpha = 1.0
        self.use_mixup = False
        self.use_cutmix = False
        
        # Early stopping
        self.early_stopping_patience = 10
        self.early_stopping_min_delta = 1e-4
        self.monitor_metric = "val_accuracy"
        self.monitor_mode = "max"
        
        # Checkpointing
        self.save_dir = "./checkpoints"
        self.save_best_only = True
        self.save_frequency = 5  # Save every N epochs
        
        # Logging
        self.log_frequency = 10  # Log every N batches
        self.use_tensorboard = True
        self.tensorboard_dir = "./runs"
        
        # Hardware configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = torch.cuda.is_available()
        self.num_gpus = torch.cuda.device_count()
        
        # Reproducibility
        self.seed = 42
        self.deterministic = True
        
        # Evaluation
        self.eval_frequency = 1  # Evaluate every N epochs
        self.test_at_end = True
        
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration updates
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration key: {key}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration dictionary."""
        return self.model_config.copy()
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """Get optimizer configuration dictionary."""
        return {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay
        }
    
    def get_scheduler_config(self) -> Dict[str, Any]:
        """Get scheduler configuration dictionary."""
        if self.scheduler_type == "cosine":
            return {
                "T_max": self.num_epochs,
                "eta_min": self.learning_rate * 0.01
            }
        elif self.scheduler_type == "step":
            return {
                "step_size": self.num_epochs // 3,
                "gamma": 0.1
            }
        elif self.scheduler_type == "plateau":
            return {
                "mode": self.monitor_mode,
                "factor": 0.5,
                "patience": 5,
                "threshold": self.early_stopping_min_delta
            }
        else:
            return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, torch.device):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value
        return config_dict
    
    def save(self, filepath: str) -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration
        """
        import json
        
        config_dict = self.to_dict()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            TrainingConfig instance
        """
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        config.update(config_dict)
        
        # Convert device string back to torch.device
        if isinstance(config.device, str):
            config.device = torch.device(config.device)
        
        return config

# Predefined configurations for different scenarios
CONFIG_PRESETS = {
    "fast_training": {
        "batch_size": 8,
        "num_epochs": 20,
        "learning_rate": 2e-4,
        "model_type": "cnn_lstm",
        "model_config": {
            "num_classes": 2,
            "cnn_backbone": "resnet18",
            "cnn_feature_dim": 256,
            "lstm_hidden_dim": 128,
            "lstm_num_layers": 1,
            "use_attention": False,
            "dropout_rate": 0.3
        }
    },
    
    "high_accuracy": {
        "batch_size": 12,
        "num_epochs": 100,
        "learning_rate": 1e-4,
        "model_type": "cnn_lstm",
        "model_config": {
            "num_classes": 2,
            "cnn_backbone": "resnet50",
            "cnn_feature_dim": 512,
            "lstm_hidden_dim": 256,
            "lstm_num_layers": 2,
            "use_attention": True,
            "dropout_rate": 0.3
        },
        "use_mixup": True,
        "label_smoothing": 0.1
    },
    
    "3d_cnn_training": {
        "batch_size": 8,
        "num_epochs": 75,
        "learning_rate": 1e-4,
        "model_type": "c3d",
        "model_config": {
            "num_classes": 2,
            "dropout_rate": 0.5
        },
        "frames_per_clip": 16
    },
    
    "transformer_training": {
        "batch_size": 4,
        "num_epochs": 60,
        "learning_rate": 1e-4,
        "model_type": "timesformer",
        "model_config": {
            "num_classes": 2,
            "input_size": (16, 224, 224),
            "patch_size": (2, 16, 16),
            "embed_dim": 384,
            "depth": 6,
            "num_heads": 6,
            "dropout_rate": 0.1
        },
        "gradient_clip_value": 0.5,
        "warmup_epochs": 10
    },
    
    "colab_optimized": {
        "batch_size": 8,
        "num_epochs": 30,
        "learning_rate": 1e-4,
        "num_workers": 2,
        "mixed_precision": True,
        "cache_data": False,
        "model_type": "cnn_lstm",
        "model_config": {
            "num_classes": 2,
            "cnn_backbone": "mobilenet_v3_small",
            "cnn_feature_dim": 256,
            "lstm_hidden_dim": 128,
            "lstm_num_layers": 1,
            "use_attention": True,
            "dropout_rate": 0.3
        }
    }
}

def get_config_preset(preset_name: str) -> TrainingConfig:
    """
    Get a predefined configuration preset.
    
    Args:
        preset_name: Name of the configuration preset
        
    Returns:
        TrainingConfig instance with preset values
    """
    if preset_name not in CONFIG_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. "
                        f"Available presets: {list(CONFIG_PRESETS.keys())}")
    
    config = TrainingConfig()
    config.update(CONFIG_PRESETS[preset_name])
    
    return config

def create_config_for_colab() -> TrainingConfig:
    """
    Create an optimized configuration for Google Colab.
    
    Returns:
        TrainingConfig optimized for Colab environment
    """
    config = get_config_preset("colab_optimized")
    
    # Additional Colab-specific optimizations
    config.data_dir = "/content/drive/MyDrive/godseye_dataset"
    config.save_dir = "/content/drive/MyDrive/violence_detection_checkpoints"
    config.tensorboard_dir = "/content/drive/MyDrive/violence_detection_runs"
    
    # Enable mixed precision if available
    config.mixed_precision = torch.cuda.is_available()
    
    # Reduce workers for Colab
    config.num_workers = 2
    
    return config

if __name__ == "__main__":
    # Example usage
    
    # Create default configuration
    config = TrainingConfig()
    print("Default configuration created")
    
    # Create Colab-optimized configuration
    colab_config = create_config_for_colab()
    print("Colab configuration created")
    
    # Get high accuracy preset
    high_acc_config = get_config_preset("high_accuracy")
    print("High accuracy configuration created")
    
    # Save configuration
    config.save("./config.json")
    print("Configuration saved")
    
    # Load configuration
    loaded_config = TrainingConfig.load("./config.json")
    print("Configuration loaded")
    
    print(f"Device: {config.device}")
    print(f"Mixed precision: {config.mixed_precision}")
    print(f"Model type: {config.model_type}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")