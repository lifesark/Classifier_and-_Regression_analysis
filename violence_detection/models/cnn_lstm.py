"""
CNN + LSTM baseline model for violence detection.

This module implements a baseline architecture that combines CNN for spatial
feature extraction with LSTM for temporal sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class CNNFeatureExtractor(nn.Module):
    """CNN backbone for extracting spatial features from video frames."""
    
    def __init__(self, 
                 backbone: str = "resnet18",
                 pretrained: bool = True,
                 feature_dim: int = 512,
                 dropout_rate: float = 0.3):
        """
        Initialize CNN feature extractor.
        
        Args:
            backbone: CNN backbone architecture
            pretrained: Whether to use pretrained weights
            feature_dim: Dimension of extracted features
            dropout_rate: Dropout rate for regularization
        """
        super(CNNFeatureExtractor, self).__init__()
        
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        
        # Load backbone model
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_output_dim = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_output_dim = 2048
        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            backbone_output_dim = 2048
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            backbone_output_dim = 1280
        elif backbone == "mobilenet_v3_small":
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            backbone_output_dim = 576
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final classification layer
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_output_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim)
        )
        
        logger.info(f"Initialized CNN feature extractor with {backbone} backbone")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input frames.
        
        Args:
            x: Input frames with shape (batch_size, frames, channels, height, width)
            
        Returns:
            Extracted features with shape (batch_size, frames, feature_dim)
        """
        batch_size, num_frames, channels, height, width = x.shape
        
        # Reshape to process all frames at once
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Extract features using backbone
        features = self.backbone(x)
        
        # Project features
        features = self.feature_projection(features)
        
        # Reshape back to sequence format
        features = features.view(batch_size, num_frames, self.feature_dim)
        
        return features

class LSTMTemporalEncoder(nn.Module):
    """LSTM encoder for temporal sequence modeling."""
    
    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 dropout_rate: float = 0.3):
        """
        Initialize LSTM temporal encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout_rate: Dropout rate for LSTM
        """
        super(LSTMTemporalEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output dimension
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        
        logger.info(f"Initialized LSTM temporal encoder with {num_layers} layers")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode temporal sequence.
        
        Args:
            x: Input features with shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Tuple of (sequence_output, final_hidden_state)
        """
        # LSTM forward pass
        sequence_output, (hidden, cell) = self.lstm(x)
        
        # Get final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final states
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final_hidden = hidden[-1]
        
        return sequence_output, final_hidden

class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence aggregation."""
    
    def __init__(self, feature_dim: int):
        """
        Initialize attention pooling.
        
        Args:
            feature_dim: Input feature dimension
        """
        super(AttentionPooling, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling.
        
        Args:
            x: Input features with shape (batch_size, sequence_length, feature_dim)
            
        Returns:
            Pooled features with shape (batch_size, feature_dim)
        """
        # Compute attention weights
        attention_weights = self.attention(x)  # (batch_size, sequence_length, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        pooled_features = torch.sum(x * attention_weights, dim=1)
        
        return pooled_features

class CNNLSTMModel(nn.Module):
    """Complete CNN + LSTM model for violence detection."""
    
    def __init__(self,
                 num_classes: int = 2,
                 cnn_backbone: str = "resnet18",
                 cnn_pretrained: bool = True,
                 cnn_feature_dim: int = 512,
                 lstm_hidden_dim: int = 256,
                 lstm_num_layers: int = 2,
                 lstm_bidirectional: bool = True,
                 use_attention: bool = True,
                 dropout_rate: float = 0.3):
        """
        Initialize CNN + LSTM model.
        
        Args:
            num_classes: Number of output classes
            cnn_backbone: CNN backbone architecture
            cnn_pretrained: Whether to use pretrained CNN weights
            cnn_feature_dim: CNN feature dimension
            lstm_hidden_dim: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            lstm_bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention pooling
            dropout_rate: Dropout rate
        """
        super(CNNLSTMModel, self).__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # CNN feature extractor
        self.cnn_extractor = CNNFeatureExtractor(
            backbone=cnn_backbone,
            pretrained=cnn_pretrained,
            feature_dim=cnn_feature_dim,
            dropout_rate=dropout_rate
        )
        
        # LSTM temporal encoder
        self.lstm_encoder = LSTMTemporalEncoder(
            input_dim=cnn_feature_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional,
            dropout_rate=dropout_rate
        )
        
        # Attention pooling (optional)
        if use_attention:
            self.attention_pooling = AttentionPooling(self.lstm_encoder.output_dim)
            classifier_input_dim = self.lstm_encoder.output_dim
        else:
            self.attention_pooling = None
            classifier_input_dim = self.lstm_encoder.output_dim
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_input_dim // 2, num_classes)
        )
        
        logger.info(f"Initialized CNN+LSTM model with {self._count_parameters()} parameters")
    
    def _count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input video frames with shape (batch_size, frames, channels, height, width)
            
        Returns:
            Classification logits with shape (batch_size, num_classes)
        """
        # Extract spatial features
        spatial_features = self.cnn_extractor(x)
        
        # Encode temporal sequence
        sequence_output, final_hidden = self.lstm_encoder(spatial_features)
        
        # Aggregate sequence features
        if self.use_attention:
            aggregated_features = self.attention_pooling(sequence_output)
        else:
            aggregated_features = final_hidden
        
        # Final classification
        logits = self.classifier(aggregated_features)
        
        return logits
    
    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """
        Get intermediate feature maps for visualization.
        
        Args:
            x: Input video frames
            
        Returns:
            Dictionary containing feature maps at different stages
        """
        features = {}
        
        # Spatial features
        spatial_features = self.cnn_extractor(x)
        features['spatial_features'] = spatial_features
        
        # Temporal features
        sequence_output, final_hidden = self.lstm_encoder(spatial_features)
        features['sequence_output'] = sequence_output
        features['final_hidden'] = final_hidden
        
        # Attention weights (if using attention)
        if self.use_attention:
            attention_weights = F.softmax(
                self.attention_pooling.attention(sequence_output), dim=1
            )
            features['attention_weights'] = attention_weights
        
        return features

class CNNGRUModel(nn.Module):
    """CNN + GRU variant of the baseline model."""
    
    def __init__(self,
                 num_classes: int = 2,
                 cnn_backbone: str = "resnet18",
                 cnn_pretrained: bool = True,
                 cnn_feature_dim: int = 512,
                 gru_hidden_dim: int = 256,
                 gru_num_layers: int = 2,
                 gru_bidirectional: bool = True,
                 use_attention: bool = True,
                 dropout_rate: float = 0.3):
        """
        Initialize CNN + GRU model.
        
        Args:
            num_classes: Number of output classes
            cnn_backbone: CNN backbone architecture
            cnn_pretrained: Whether to use pretrained CNN weights
            cnn_feature_dim: CNN feature dimension
            gru_hidden_dim: GRU hidden dimension
            gru_num_layers: Number of GRU layers
            gru_bidirectional: Whether to use bidirectional GRU
            use_attention: Whether to use attention pooling
            dropout_rate: Dropout rate
        """
        super(CNNGRUModel, self).__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # CNN feature extractor
        self.cnn_extractor = CNNFeatureExtractor(
            backbone=cnn_backbone,
            pretrained=cnn_pretrained,
            feature_dim=cnn_feature_dim,
            dropout_rate=dropout_rate
        )
        
        # GRU temporal encoder
        self.gru = nn.GRU(
            input_size=cnn_feature_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            bidirectional=gru_bidirectional,
            dropout=dropout_rate if gru_num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output dimension
        gru_output_dim = gru_hidden_dim * (2 if gru_bidirectional else 1)
        
        # Attention pooling (optional)
        if use_attention:
            self.attention_pooling = AttentionPooling(gru_output_dim)
            classifier_input_dim = gru_output_dim
        else:
            self.attention_pooling = None
            classifier_input_dim = gru_output_dim
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_input_dim // 2, num_classes)
        )
        
        logger.info(f"Initialized CNN+GRU model with {self._count_parameters()} parameters")
    
    def _count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # Extract spatial features
        spatial_features = self.cnn_extractor(x)
        
        # Encode temporal sequence with GRU
        sequence_output, final_hidden = self.gru(spatial_features)
        
        # Aggregate sequence features
        if self.use_attention:
            aggregated_features = self.attention_pooling(sequence_output)
        else:
            # For bidirectional GRU, concatenate forward and backward final states
            if final_hidden.shape[0] > 1:  # Bidirectional
                aggregated_features = torch.cat([final_hidden[-2], final_hidden[-1]], dim=1)
            else:
                aggregated_features = final_hidden[-1]
        
        # Final classification
        logits = self.classifier(aggregated_features)
        
        return logits

def create_cnn_lstm_model(model_config: dict) -> nn.Module:
    """
    Factory function to create CNN+LSTM models.
    
    Args:
        model_config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model_type = model_config.get("type", "lstm")
    
    if model_type == "lstm":
        return CNNLSTMModel(**model_config)
    elif model_type == "gru":
        return CNNGRUModel(**model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = CNNLSTMModel(
        num_classes=2,
        cnn_backbone="resnet18",
        cnn_feature_dim=512,
        lstm_hidden_dim=256,
        use_attention=True
    ).to(device)
    
    # Test with dummy input
    batch_size = 4
    num_frames = 16
    channels = 3
    height = width = 224
    
    dummy_input = torch.randn(batch_size, num_frames, channels, height, width).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        
        # Get feature maps
        features = model.get_feature_maps(dummy_input)
        for name, feature in features.items():
            print(f"{name} shape: {feature.shape}")