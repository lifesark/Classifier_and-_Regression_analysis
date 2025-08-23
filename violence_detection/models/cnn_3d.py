"""
3D CNN models for violence detection.

This module implements various 3D CNN architectures including C3D, I3D, 
and R(2+1)D for spatio-temporal feature learning in videos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

class Conv3dBlock(nn.Module):
    """Basic 3D convolution block with batch normalization and activation."""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 padding: Tuple[int, int, int] = (1, 1, 1),
                 use_bn: bool = True,
                 activation: str = "relu",
                 dropout_rate: float = 0.0):
        """
        Initialize 3D convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            use_bn: Whether to use batch normalization
            activation: Activation function ("relu", "leaky_relu", "gelu")
            dropout_rate: Dropout rate
        """
        super(Conv3dBlock, self).__init__()
        
        layers = []
        
        # 3D Convolution
        layers.append(nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn
        ))
        
        # Batch Normalization
        if use_bn:
            layers.append(nn.BatchNorm3d(out_channels))
        
        # Activation
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky_relu":
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "gelu":
            layers.append(nn.GELU())
        
        # Dropout
        if dropout_rate > 0:
            layers.append(nn.Dropout3d(dropout_rate))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class C3DModel(nn.Module):
    """
    C3D model for video understanding.
    
    Reference: Learning spatiotemporal features with 3d convolutional networks
    """
    
    def __init__(self,
                 num_classes: int = 2,
                 input_channels: int = 3,
                 dropout_rate: float = 0.5):
        """
        Initialize C3D model.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            dropout_rate: Dropout rate for regularization
        """
        super(C3DModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.conv1 = Conv3dBlock(input_channels, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool1 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = Conv3dBlock(64, 128, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        
        self.conv3a = Conv3dBlock(128, 256, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.conv3b = Conv3dBlock(256, 256, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        
        self.conv4a = Conv3dBlock(256, 512, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.conv4b = Conv3dBlock(512, 512, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool4 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        
        self.conv5a = Conv3dBlock(512, 512, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.conv5b = Conv3dBlock(512, 512, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool5 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        
        # Classifier
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        
        logger.info(f"Initialized C3D model with {self._count_parameters()} parameters")
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor with shape (batch_size, channels, frames, height, width)
            
        Returns:
            Classification logits
        """
        # Feature extraction
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3b(self.conv3a(x)))
        x = self.pool4(self.conv4b(self.conv4a(x)))
        x = self.pool5(self.conv5b(self.conv5a(x)))
        
        # Global pooling and classification
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class Inflate3DConv(nn.Module):
    """Inflated 3D convolution for I3D model."""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 padding: Tuple[int, int, int] = (1, 1, 1)):
        """Initialize inflated 3D convolution."""
        super(Inflate3DConv, self).__init__()
        
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class I3DBlock(nn.Module):
    """Basic building block for I3D model."""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Tuple[int, int, int] = (1, 1, 1)):
        """
        Initialize I3D block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Convolution stride
        """
        super(I3DBlock, self).__init__()
        
        # Branch 1: 1x1x1 conv
        self.branch1 = Inflate3DConv(in_channels, out_channels // 4, (1, 1, 1))
        
        # Branch 2: 1x1x1 -> 3x3x3 conv
        self.branch2 = nn.Sequential(
            Inflate3DConv(in_channels, out_channels // 4, (1, 1, 1)),
            Inflate3DConv(out_channels // 4, out_channels // 4, (3, 3, 3), stride)
        )
        
        # Branch 3: 1x1x1 -> 3x3x3 -> 3x3x3 conv
        self.branch3 = nn.Sequential(
            Inflate3DConv(in_channels, out_channels // 8, (1, 1, 1)),
            Inflate3DConv(out_channels // 8, out_channels // 4, (3, 3, 3)),
            Inflate3DConv(out_channels // 4, out_channels // 4, (3, 3, 3), stride)
        )
        
        # Branch 4: MaxPool -> 1x1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool3d((3, 3, 3), stride, padding=(1, 1, 1)),
            Inflate3DConv(in_channels, out_channels // 4, (1, 1, 1))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)
        
        return torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)

class I3DModel(nn.Module):
    """
    I3D model for video understanding.
    
    Reference: Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
    """
    
    def __init__(self,
                 num_classes: int = 2,
                 input_channels: int = 3,
                 dropout_rate: float = 0.5):
        """
        Initialize I3D model.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            dropout_rate: Dropout rate
        """
        super(I3DModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv1 = Inflate3DConv(input_channels, 64, (7, 7, 7), (2, 2, 2), (3, 3, 3))
        self.maxpool1 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # I3D blocks
        self.block1 = I3DBlock(64, 256)
        self.block2 = I3DBlock(256, 512, stride=(2, 2, 2))
        self.block3 = I3DBlock(512, 1024, stride=(2, 2, 2))
        self.block4 = I3DBlock(1024, 2048, stride=(2, 2, 2))
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(2048, num_classes)
        
        logger.info(f"Initialized I3D model with {self._count_parameters()} parameters")
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor with shape (batch_size, channels, frames, height, width)
            
        Returns:
            Classification logits
        """
        x = self.conv1(x)
        x = self.maxpool1(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.global_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class R2Plus1DBlock(nn.Module):
    """R(2+1)D block that factorizes 3D convolution."""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Tuple[int, int, int] = (1, 1, 1)):
        """
        Initialize R(2+1)D block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Convolution stride
        """
        super(R2Plus1DBlock, self).__init__()
        
        # Calculate intermediate channels for factorization
        intermediate_channels = int(out_channels * 0.75)
        
        # Spatial convolution (2D)
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_channels, intermediate_channels, (1, 3, 3), 
                     (1, stride[1], stride[2]), (0, 1, 1), bias=False),
            nn.BatchNorm3d(intermediate_channels),
            nn.ReLU(inplace=True)
        )
        
        # Temporal convolution (1D)
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(intermediate_channels, out_channels, (3, 1, 1), 
                     (stride[0], 1, 1), (1, 0, 0), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Skip connection
        if in_channels != out_channels or stride != (1, 1, 1):
            self.skip_connection = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, (1, 1, 1), stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.skip_connection = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Apply factorized convolutions
        out = self.spatial_conv(x)
        out = self.temporal_conv(out)
        
        # Skip connection
        if self.skip_connection is not None:
            identity = self.skip_connection(identity)
        
        out += identity
        return F.relu(out)

class R2Plus1DModel(nn.Module):
    """
    R(2+1)D model for video understanding.
    
    Reference: A Closer Look at Spatiotemporal Convolutions for Action Recognition
    """
    
    def __init__(self,
                 num_classes: int = 2,
                 input_channels: int = 3,
                 layers: List[int] = [2, 2, 2, 2],
                 dropout_rate: float = 0.5):
        """
        Initialize R(2+1)D model.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            layers: Number of blocks in each layer
            dropout_rate: Dropout rate
        """
        super(R2Plus1DModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 64, (3, 7, 7), (1, 2, 2), (1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # R(2+1)D layers
        self.layer1 = self._make_layer(64, 64, layers[0], stride=(1, 1, 1))
        self.layer2 = self._make_layer(64, 128, layers[1], stride=(2, 2, 2))
        self.layer3 = self._make_layer(128, 256, layers[2], stride=(2, 2, 2))
        self.layer4 = self._make_layer(256, 512, layers[3], stride=(2, 2, 2))
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(512, num_classes)
        
        logger.info(f"Initialized R(2+1)D model with {self._count_parameters()} parameters")
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, 
                   stride: Tuple[int, int, int]) -> nn.Sequential:
        """Create a layer with multiple R(2+1)D blocks."""
        layers = []
        
        # First block with potential stride
        layers.append(R2Plus1DBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(R2Plus1DBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor with shape (batch_size, channels, frames, height, width)
            
        Returns:
            Classification logits
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.global_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def create_3d_cnn_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create 3D CNN models.
    
    Args:
        model_type: Type of 3D CNN model ("c3d", "i3d", "r2plus1d")
        **kwargs: Model configuration parameters
        
    Returns:
        Initialized 3D CNN model
    """
    if model_type.lower() == "c3d":
        return C3DModel(**kwargs)
    elif model_type.lower() == "i3d":
        return I3DModel(**kwargs)
    elif model_type.lower() == "r2plus1d":
        return R2Plus1DModel(**kwargs)
    else:
        raise ValueError(f"Unsupported 3D CNN model type: {model_type}")

if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test different models
    models = {
        "C3D": C3DModel(num_classes=2),
        "I3D": I3DModel(num_classes=2),
        "R(2+1)D": R2Plus1DModel(num_classes=2)
    }
    
    # Create dummy input (batch_size=2, channels=3, frames=16, height=224, width=224)
    dummy_input = torch.randn(2, 3, 16, 224, 224).to(device)
    
    for name, model in models.items():
        model = model.to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"{name}:")
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print()