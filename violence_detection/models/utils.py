"""
Model utilities and helper functions for violence detection.

This module provides utility functions for model creation, loading,
saving, and various helper operations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
import os
import json
import logging
from .cnn_lstm import create_cnn_lstm_model, CNNLSTMModel, CNNGRUModel
from .cnn_3d import create_3d_cnn_model, C3DModel, I3DModel, R2Plus1DModel
from .transformer import create_transformer_model, TimeSformerModel, VideoSwinTransformer

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory class for creating violence detection models."""
    
    SUPPORTED_MODELS = {
        # CNN + RNN models
        "cnn_lstm": CNNLSTMModel,
        "cnn_gru": CNNGRUModel,
        
        # 3D CNN models
        "c3d": C3DModel,
        "i3d": I3DModel,
        "r2plus1d": R2Plus1DModel,
        
        # Transformer models
        "timesformer": TimeSformerModel,
        "video_swin": VideoSwinTransformer,
    }
    
    @classmethod
    def create_model(cls, model_type: str, model_config: Dict[str, Any]) -> nn.Module:
        """
        Create a model based on type and configuration.
        
        Args:
            model_type: Type of model to create
            model_config: Configuration dictionary for the model
            
        Returns:
            Initialized model
        """
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {list(cls.SUPPORTED_MODELS.keys())}")
        
        model_class = cls.SUPPORTED_MODELS[model_type]
        
        try:
            model = model_class(**model_config)
            logger.info(f"Created {model_type} model successfully")
            return model
        except Exception as e:
            logger.error(f"Error creating {model_type} model: {str(e)}")
            raise
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """
        Get information about a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary containing model information
        """
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_class = cls.SUPPORTED_MODELS[model_type]
        
        info = {
            "name": model_type,
            "class": model_class.__name__,
            "module": model_class.__module__,
            "supported": True
        }
        
        return info

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Get model size information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary containing size information
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    # Estimate memory usage (assuming float32)
    param_size_mb = total_params * 4 / 1024 / 1024
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameter_size_mb": param_size_mb
    }

def save_model_checkpoint(model: nn.Module,
                         optimizer: Optional[torch.optim.Optimizer],
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                         epoch: int,
                         loss: float,
                         metrics: Dict[str, float],
                         save_path: str,
                         model_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Save model checkpoint with all necessary information.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch
        loss: Current loss value
        metrics: Dictionary of evaluation metrics
        save_path: Path to save the checkpoint
        model_config: Model configuration dictionary
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
        "metrics": metrics
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if model_config is not None:
        checkpoint["model_config"] = model_config
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")

def load_model_checkpoint(model: nn.Module,
                         checkpoint_path: str,
                         optimizer: Optional[torch.optim.Optimizer] = None,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to the checkpoint file
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load the model on
        
    Returns:
        Dictionary containing checkpoint information
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", 0.0),
        "metrics": checkpoint.get("metrics", {}),
        "model_config": checkpoint.get("model_config", {})
    }

def initialize_weights(model: nn.Module, init_type: str = "normal") -> None:
    """
    Initialize model weights.
    
    Args:
        model: PyTorch model
        init_type: Type of initialization ("normal", "xavier", "kaiming")
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if init_type == "normal":
                nn.init.normal_(module.weight, 0.0, 0.02)
            elif init_type == "xavier":
                nn.init.xavier_normal_(module.weight)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.Linear):
            if init_type == "normal":
                nn.init.normal_(module.weight, 0.0, 0.02)
            elif init_type == "xavier":
                nn.init.xavier_normal_(module.weight)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

def freeze_model_layers(model: nn.Module, 
                       freeze_backbone: bool = True,
                       freeze_classifier: bool = False) -> None:
    """
    Freeze specific layers in the model.
    
    Args:
        model: PyTorch model
        freeze_backbone: Whether to freeze backbone layers
        freeze_classifier: Whether to freeze classifier layers
    """
    for name, param in model.named_parameters():
        if freeze_backbone and any(backbone_name in name for backbone_name in 
                                 ["backbone", "cnn_extractor", "conv", "patch_embed"]):
            param.requires_grad = False
            logger.info(f"Froze parameter: {name}")
        
        if freeze_classifier and any(classifier_name in name for classifier_name in 
                                   ["classifier", "head", "fc"]):
            param.requires_grad = False
            logger.info(f"Froze parameter: {name}")

def unfreeze_model_layers(model: nn.Module) -> None:
    """
    Unfreeze all layers in the model.
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True
    
    logger.info("Unfroze all model parameters")

def get_model_summary(model: nn.Module, input_size: tuple) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size for the model
        
    Returns:
        String containing model summary
    """
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = -1
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = -1
            
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
        
        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))
    
    # Create model summary
    from collections import OrderedDict
    device = next(model.parameters()).device
    
    summary = OrderedDict()
    hooks = []
    
    model.apply(register_hook)
    
    # Make a forward pass
    model.eval()
    x = torch.randn(1, *input_size).to(device)
    with torch.no_grad():
        model(x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Format summary
    summary_str = "=" * 70 + "\n"
    summary_str += f"{'Layer (type)':>20} {'Output Shape':>25} {'Param #':>15}\n"
    summary_str += "=" * 70 + "\n"
    
    total_params = 0
    total_output = 0
    trainable_params = 0
    
    for layer in summary:
        output_shape = str(summary[layer]["output_shape"])
        num_params = summary[layer]["nb_params"]
        
        summary_str += f"{layer:>20} {output_shape:>25} {num_params:>15,}\n"
        
        total_params += num_params
        if summary[layer].get("trainable", True):
            trainable_params += num_params
        
        if isinstance(summary[layer]["output_shape"], list):
            for shape in summary[layer]["output_shape"]:
                total_output += abs(torch.prod(torch.LongTensor(shape)))
        else:
            total_output += abs(torch.prod(torch.LongTensor(summary[layer]["output_shape"])))
    
    summary_str += "=" * 70 + "\n"
    summary_str += f"Total params: {total_params:,}\n"
    summary_str += f"Trainable params: {trainable_params:,}\n"
    summary_str += f"Non-trainable params: {total_params - trainable_params:,}\n"
    summary_str += "=" * 70
    
    return summary_str

def convert_model_to_torchscript(model: nn.Module, 
                                input_size: tuple,
                                save_path: str) -> None:
    """
    Convert model to TorchScript for production deployment.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        save_path: Path to save the TorchScript model
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # Trace the model
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(save_path)
        logger.info(f"Saved TorchScript model to {save_path}")
    except Exception as e:
        logger.error(f"Error converting model to TorchScript: {str(e)}")
        raise

# Model configuration templates
DEFAULT_MODEL_CONFIGS = {
    "cnn_lstm_small": {
        "num_classes": 2,
        "cnn_backbone": "resnet18",
        "cnn_feature_dim": 256,
        "lstm_hidden_dim": 128,
        "lstm_num_layers": 1,
        "use_attention": False,
        "dropout_rate": 0.3
    },
    
    "cnn_lstm_medium": {
        "num_classes": 2,
        "cnn_backbone": "resnet50",
        "cnn_feature_dim": 512,
        "lstm_hidden_dim": 256,
        "lstm_num_layers": 2,
        "use_attention": True,
        "dropout_rate": 0.3
    },
    
    "c3d_small": {
        "num_classes": 2,
        "dropout_rate": 0.5
    },
    
    "timesformer_small": {
        "num_classes": 2,
        "input_size": (16, 224, 224),
        "patch_size": (2, 16, 16),
        "embed_dim": 384,
        "depth": 6,
        "num_heads": 6,
        "dropout_rate": 0.1
    }
}

def get_default_config(model_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a model type.
    
    Args:
        model_type: Type of model
        
    Returns:
        Default configuration dictionary
    """
    if model_type in DEFAULT_MODEL_CONFIGS:
        return DEFAULT_MODEL_CONFIGS[model_type].copy()
    else:
        logger.warning(f"No default config found for {model_type}")
        return {}

if __name__ == "__main__":
    # Example usage
    from .cnn_lstm import CNNLSTMModel
    
    # Create a model using factory
    model_config = get_default_config("cnn_lstm_medium")
    model = ModelFactory.create_model("cnn_lstm", model_config)
    
    # Get model information
    info = get_model_size(model)
    print(f"Model parameters: {info['trainable_parameters']:,}")
    print(f"Model size: {info['parameter_size_mb']:.2f} MB")
    
    # Initialize weights
    initialize_weights(model, "kaiming")
    
    print("Model created and initialized successfully!")