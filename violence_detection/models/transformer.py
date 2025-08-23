"""
Transformer-based models for violence detection.

This module implements video transformer architectures including
TimeSformer and Video Swin Transformer for video understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class PatchEmbedding3D(nn.Module):
    """3D patch embedding for video transformers."""
    
    def __init__(self,
                 patch_size: Tuple[int, int, int] = (2, 16, 16),
                 in_channels: int = 3,
                 embed_dim: int = 768):
        """
        Initialize 3D patch embedding.
        
        Args:
            patch_size: Size of patches (frames, height, width)
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super(PatchEmbedding3D, self).__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 3D convolution for patch embedding
        self.projection = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for patch embedding.
        
        Args:
            x: Input tensor (batch_size, channels, frames, height, width)
            
        Returns:
            Patch embeddings (batch_size, num_patches, embed_dim)
        """
        # Apply 3D convolution
        x = self.projection(x)  # (B, embed_dim, T', H', W')
        
        # Flatten spatial and temporal dimensions
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, T'*H'*W', embed_dim)
        
        # Apply layer normalization
        x = self.norm(x)
        
        return x

class MultiHeadAttention3D(nn.Module):
    """Multi-head attention for 3D video data."""
    
    def __init__(self,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 dropout_rate: float = 0.1,
                 attention_type: str = "space_time"):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
            attention_type: Type of attention ("space_time", "space_only", "time_only")
        """
        super(MultiHeadAttention3D, self).__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_type = attention_type
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Scale factor
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        Args:
            x: Input tensor (batch_size, num_patches, embed_dim)
            mask: Attention mask (optional)
            
        Returns:
            Output tensor after attention
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        x = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
        
        # Final projection
        x = self.projection(x)
        
        return x

class TransformerBlock(nn.Module):
    """Transformer block for video understanding."""
    
    def __init__(self,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout_rate: float = 0.1,
                 attention_type: str = "space_time"):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout_rate: Dropout rate
            attention_type: Type of attention
        """
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention3D(
            embed_dim, num_heads, dropout_rate, attention_type
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for transformer block."""
        # Attention with residual connection
        x = x + self.attention(self.norm1(x), mask)
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x

class TimeSformerModel(nn.Module):
    """
    TimeSformer model for video understanding.
    
    Reference: Is Space-Time Attention All You Need for Video Understanding?
    """
    
    def __init__(self,
                 num_classes: int = 2,
                 input_size: Tuple[int, int, int] = (16, 224, 224),
                 patch_size: Tuple[int, int, int] = (2, 16, 16),
                 in_channels: int = 3,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout_rate: float = 0.1,
                 attention_type: str = "space_time"):
        """
        Initialize TimeSformer model.
        
        Args:
            num_classes: Number of output classes
            input_size: Input video size (frames, height, width)
            patch_size: Patch size (frames, height, width)
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout_rate: Dropout rate
            attention_type: Type of attention mechanism
        """
        super(TimeSformerModel, self).__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding3D(patch_size, in_channels, embed_dim)
        
        # Calculate number of patches
        num_patches = (input_size[0] // patch_size[0]) * \
                     (input_size[1] // patch_size[1]) * \
                     (input_size[2] // patch_size[2])
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_rate, attention_type)
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized TimeSformer model with {self._count_parameters()} parameters")
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for TimeSformer.
        
        Args:
            x: Input tensor (batch_size, channels, frames, height, width)
            
        Returns:
            Classification logits
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Classification using class token
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits

class WindowAttention3D(nn.Module):
    """Window-based multi-head attention for Video Swin Transformer."""
    
    def __init__(self,
                 dim: int,
                 window_size: Tuple[int, int, int],
                 num_heads: int,
                 qkv_bias: bool = True,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        """
        Initialize window attention.
        
        Args:
            dim: Input dimension
            window_size: Window size for attention
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
        """
        super(WindowAttention3D, self).__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for window attention."""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
        
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class VideoSwinTransformerBlock(nn.Module):
    """Video Swin Transformer block."""
    
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: Tuple[int, int, int] = (8, 7, 7),
                 shift_size: Tuple[int, int, int] = (0, 0, 0),
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0):
        """
        Initialize Video Swin Transformer block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            window_size: Window size for attention
            shift_size: Shift size for shifted window attention
            mlp_ratio: MLP expansion ratio
            qkv_bias: Whether to use bias in QKV projection
            drop: Dropout rate
            attn_drop: Attention dropout rate
        """
        super(VideoSwinTransformerBlock, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        
        # Window attention
        self.attn = WindowAttention3D(
            dim, window_size, num_heads, qkv_bias, attn_drop, drop
        )
        
        # Layer normalization
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Video Swin Transformer block."""
        shortcut = x
        x = self.norm1(x)
        
        # Window attention
        x = self.attn(x)
        
        # Residual connection
        x = shortcut + x
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x

class VideoSwinTransformer(nn.Module):
    """
    Video Swin Transformer for video understanding.
    
    Reference: Video Swin Transformer
    """
    
    def __init__(self,
                 num_classes: int = 2,
                 input_size: Tuple[int, int, int] = (32, 224, 224),
                 patch_size: Tuple[int, int, int] = (2, 4, 4),
                 in_channels: int = 3,
                 embed_dim: int = 96,
                 depths: List[int] = [2, 2, 6, 2],
                 num_heads: List[int] = [3, 6, 12, 24],
                 window_size: Tuple[int, int, int] = (8, 7, 7),
                 mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0):
        """
        Initialize Video Swin Transformer.
        
        Args:
            num_classes: Number of output classes
            input_size: Input video size
            patch_size: Patch size
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            depths: Number of blocks in each stage
            num_heads: Number of heads in each stage
            window_size: Window size for attention
            mlp_ratio: MLP expansion ratio
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
        """
        super(VideoSwinTransformer, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        
        # Patch embedding
        self.patch_embed = PatchEmbedding3D(patch_size, in_channels, embed_dim)
        
        # Stages
        self.stages = nn.ModuleList()
        dim = embed_dim
        
        for i_layer in range(self.num_layers):
            stage = nn.ModuleList([
                VideoSwinTransformerBlock(
                    dim=dim * (2 ** i_layer),
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate
                )
                for _ in range(depths[i_layer])
            ])
            self.stages.append(stage)
        
        # Final layer norm and classifier
        self.norm = nn.LayerNorm(dim * (2 ** (self.num_layers - 1)))
        self.head = nn.Linear(dim * (2 ** (self.num_layers - 1)), num_classes)
        
        logger.info(f"Initialized Video Swin Transformer with {self._count_parameters()} parameters")
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Video Swin Transformer.
        
        Args:
            x: Input tensor (batch_size, channels, frames, height, width)
            
        Returns:
            Classification logits
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Apply stages
        for stage in self.stages:
            for block in stage:
                x = block(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Layer normalization and classification
        x = self.norm(x)
        x = self.head(x)
        
        return x

def create_transformer_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create transformer models.
    
    Args:
        model_type: Type of transformer model ("timesformer", "video_swin")
        **kwargs: Model configuration parameters
        
    Returns:
        Initialized transformer model
    """
    if model_type.lower() == "timesformer":
        return TimeSformerModel(**kwargs)
    elif model_type.lower() == "video_swin":
        return VideoSwinTransformer(**kwargs)
    else:
        raise ValueError(f"Unsupported transformer model type: {model_type}")

if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test TimeSformer
    timesformer = TimeSformerModel(
        num_classes=2,
        input_size=(16, 224, 224),
        patch_size=(2, 16, 16),
        embed_dim=384,
        depth=6,
        num_heads=6
    ).to(device)
    
    # Test Video Swin Transformer
    video_swin = VideoSwinTransformer(
        num_classes=2,
        input_size=(32, 224, 224),
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24]
    ).to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 16, 224, 224).to(device)
    
    print("TimeSformer:")
    with torch.no_grad():
        output = timesformer(dummy_input)
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in timesformer.parameters()):,}")
    
    print("\nVideo Swin Transformer:")
    dummy_input_swin = torch.randn(1, 3, 32, 224, 224).to(device)
    with torch.no_grad():
        output = video_swin(dummy_input_swin)
        print(f"  Input shape: {dummy_input_swin.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in video_swin.parameters()):,}")