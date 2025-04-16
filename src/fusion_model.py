"""
Multimodal Fusion Model for Emotion Recognition

This module implements the multimodal fusion approach based on the MemoCMT paper.
It combines features from text and voice modalities for emotion prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFusionModel(nn.Module):
    """
    Multimodal fusion model for combining text and voice features.
    
    This model implements the cross-modal transformer fusion approach from the MemoCMT paper,
    which uses attention mechanisms to combine features from different modalities.
    """
    
    def __init__(self, text_dim=256, voice_dim=256, hidden_dim=256, num_classes=8, 
                 dropout_rate=0.1, fusion_method='cross_attention'):
        """
        Initialize the multimodal fusion model.
        
        Args:
            text_dim (int): Dimension of text features
            voice_dim (int): Dimension of voice features
            hidden_dim (int): Dimension of hidden layers
            num_classes (int): Number of emotion classes to predict
            dropout_rate (float): Dropout rate for regularization
            fusion_method (str): Method for fusion ('cross_attention', 'concat', 'min', 'max', 'avg')
        """
        super(MultimodalFusionModel, self).__init__()
        
        self.text_dim = text_dim
        self.voice_dim = voice_dim
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method
        
        # Projection layers to align feature dimensions
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.voice_projection = nn.Linear(voice_dim, hidden_dim)
        
        # Cross-modal attention layers
        if fusion_method == 'cross_attention':
            # Text-to-voice attention
            self.text_to_voice_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            
            # Voice-to-text attention
            self.voice_to_text_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Classification heads
        self.text_classifier = nn.Linear(hidden_dim, num_classes)
        self.voice_classifier = nn.Linear(hidden_dim, num_classes)
        
        # Fusion classifier
        if fusion_method == 'concat':
            self.fusion_classifier = nn.Linear(hidden_dim * 2, num_classes)
        else:
            self.fusion_classifier = nn.Linear(hidden_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, text_features, voice_features):
        """
        Forward pass through the model.
        
        Args:
            text_features (torch.Tensor): Features from text modality [batch_size, text_dim]
            voice_features (torch.Tensor): Features from voice modality [batch_size, voice_dim]
            
        Returns:
            tuple: Tuple containing:
                - text_logits (torch.Tensor): Logits from text modality
                - voice_logits (torch.Tensor): Logits from voice modality
                - fusion_logits (torch.Tensor): Logits from fused features
        """
        # Project features to common dimension
        text_proj = self.text_projection(text_features)  # [batch_size, hidden_dim]
        voice_proj = self.voice_projection(voice_features)  # [batch_size, hidden_dim]
        
        # Apply dropout
        text_proj = self.dropout(text_proj)
        voice_proj = self.dropout(voice_proj)
        
        # Get modality-specific predictions
        text_logits = self.text_classifier(text_proj)
        voice_logits = self.voice_classifier(voice_proj)
        
        # Fuse features based on specified method
        if self.fusion_method == 'cross_attention':
            # Text-to-voice attention
            text_attn = self.text_to_voice_attention(text_proj)  # [batch_size, 1]
            
            # Voice-to-text attention
            voice_attn = self.voice_to_text_attention(voice_proj)  # [batch_size, 1]
            
            # Normalize attention weights
            attn_weights = F.softmax(torch.cat([text_attn, voice_attn], dim=1), dim=1)  # [batch_size, 2]
            
            # Apply attention weights
            fused_features = (text_proj * attn_weights[:, 0].unsqueeze(1) + 
                             voice_proj * attn_weights[:, 1].unsqueeze(1))
        
        elif self.fusion_method == 'min':
            # Min aggregation (as per MemoCMT paper)
            fused_features = torch.min(text_proj, voice_proj)
        
        elif self.fusion_method == 'max':
            # Max aggregation
            fused_features = torch.max(text_proj, voice_proj)
        
        elif self.fusion_method == 'avg':
            # Average aggregation
            fused_features = (text_proj + voice_proj) / 2
        
        elif self.fusion_method == 'concat':
            # Concatenation
            fused_features = torch.cat([text_proj, voice_proj], dim=1)
        
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        
        # Get fusion predictions
        fusion_logits = self.fusion_classifier(fused_features)
        
        return text_logits, voice_logits, fusion_logits


class MemoCMTFusion(nn.Module):
    """
    Implementation of the MemoCMT fusion approach from the paper.
    
    This model uses a cross-modal transformer with min aggregation for fusion.
    """
    
    def __init__(self, feature_dim=256, num_heads=8, num_layers=2, num_classes=8, 
                 dropout_rate=0.1, fusion_method='min'):
        """
        Initialize the MemoCMT fusion model.
        
        Args:
            feature_dim (int): Dimension of input features
            num_heads (int): Number of attention heads in transformer
            num_layers (int): Number of transformer layers
            num_classes (int): Number of emotion classes to predict
            dropout_rate (float): Dropout rate for regularization
            fusion_method (str): Method for fusion ('min', 'max', 'avg', 'concat')
        """
        super(MemoCMTFusion, self).__init__()
        
        self.feature_dim = feature_dim
        self.fusion_method = fusion_method
        
        # Transformer encoder for cross-modal attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification heads
        self.text_classifier = nn.Linear(feature_dim, num_classes)
        self.voice_classifier = nn.Linear(feature_dim, num_classes)
        
        # Fusion classifier
        if fusion_method == 'concat':
            self.fusion_classifier = nn.Linear(feature_dim * 2, num_classes)
        else:
            self.fusion_classifier = nn.Linear(feature_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, text_features, voice_features):
        """
        Forward pass through the model.
        
        Args:
            text_features (torch.Tensor): Features from text modality [batch_size, feature_dim]
            voice_features (torch.Tensor): Features from voice modality [batch_size, feature_dim]
            
        Returns:
            tuple: Tuple containing:
                - text_logits (torch.Tensor): Logits from text modality
                - voice_logits (torch.Tensor): Logits from voice modality
                - fusion_logits (torch.Tensor): Logits from fused features
        """
        batch_size = text_features.size(0)
        
        # Stack features for transformer input [batch_size, 2, feature_dim]
        stacked_features = torch.stack([text_features, voice_features], dim=1)
        
        # Apply transformer
        transformed_features = self.transformer(stacked_features)
        
        # Extract modality-specific features
        text_transformed = transformed_features[:, 0]  # [batch_size, feature_dim]
        voice_transformed = transformed_features[:, 1]  # [batch_size, feature_dim]
        
        # Apply dropout
        text_transformed = self.dropout(text_transformed)
        voice_transformed = self.dropout(voice_transformed)
        
        # Get modality-specific predictions
        text_logits = self.text_classifier(text_transformed)
        voice_logits = self.voice_classifier(voice_transformed)
        
        # Fuse features based on specified method
        if self.fusion_method == 'min':
            # Min aggregation (as per MemoCMT paper)
            fused_features = torch.min(text_transformed, voice_transformed)
        
        elif self.fusion_method == 'max':
            # Max aggregation
            fused_features = torch.max(text_transformed, voice_transformed)
        
        elif self.fusion_method == 'avg':
            # Average aggregation
            fused_features = (text_transformed + voice_transformed) / 2
        
        elif self.fusion_method == 'concat':
            # Concatenation
            fused_features = torch.cat([text_transformed, voice_transformed], dim=1)
        
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        
        # Get fusion predictions
        fusion_logits = self.fusion_classifier(fused_features)
        
        return text_logits, voice_logits, fusion_logits


def load_pretrained_fusion_model(model_path, model_type='memocmt', feature_dim=256, num_classes=8):
    """
    Load a pretrained fusion model.
    
    Args:
        model_path (str): Path to the pretrained model
        model_type (str): Type of fusion model ('memocmt' or 'multimodal')
        feature_dim (int): Dimension of input features
        num_classes (int): Number of emotion classes
        
    Returns:
        nn.Module: Loaded fusion model
    """
    if model_type.lower() == 'memocmt':
        model = MemoCMTFusion(feature_dim=feature_dim, num_classes=num_classes)
    elif model_type.lower() == 'multimodal':
        model = MultimodalFusionModel(text_dim=feature_dim, voice_dim=feature_dim, 
                                     hidden_dim=feature_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path))
    return model


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    feature_dim = 256
    num_classes = 8
    
    # Create random features
    text_features = torch.randn(batch_size, feature_dim)
    voice_features = torch.randn(batch_size, feature_dim)
    
    # Initialize models
    multimodal_fusion = MultimodalFusionModel(
        text_dim=feature_dim, 
        voice_dim=feature_dim,
        hidden_dim=feature_dim,
        num_classes=num_classes
    )
    
    memocmt_fusion = MemoCMTFusion(
        feature_dim=feature_dim,
        num_classes=num_classes
    )
    
    # Forward pass through models
    mm_text_logits, mm_voice_logits, mm_fusion_logits = multimodal_fusion(text_features, voice_features)
    cmt_text_logits, cmt_voice_logits, cmt_fusion_logits = memocmt_fusion(text_features, voice_features)
    
    # Print shapes
    print("MultimodalFusion output shapes:")
    print(f"Text logits: {mm_text_logits.shape}")
    print(f"Voice logits: {mm_voice_logits.shape}")
    print(f"Fusion logits: {mm_fusion_logits.shape}")
    
    print("\nMemoCMTFusion output shapes:")
    print(f"Text logits: {cmt_text_logits.shape}")
    print(f"Voice logits: {cmt_voice_logits.shape}")
    print(f"Fusion logits: {cmt_fusion_logits.shape}")
