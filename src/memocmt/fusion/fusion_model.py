"""
Cross-Modal Transformer Fusion for MemoCMT Implementation

This module implements the cross-modal transformer fusion component of the MemoCMT model
as described in the paper "MemoCMT: multimodal emotion recognition using 
cross-modal transformer-based feature fusion" (2025).

The fusion model uses a cross-modal transformer to combine text and voice features.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism as specified in MemoCMT paper.
    
    This module implements the attention mechanism that allows each modality
    to attend to the other modality's features.
    """
    
    def __init__(self, feature_dim=256, num_heads=8, dropout_rate=0.1):
        """
        Initialize the cross-modal attention module.
        
        Args:
            feature_dim (int): Dimension of input features
            num_heads (int): Number of attention heads
            dropout_rate (float): Dropout rate for regularization
        """
        super(CrossModalAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass through the cross-modal attention.
        
        Args:
            query (torch.Tensor): Query tensor from one modality
            key (torch.Tensor): Key tensor from another modality
            value (torch.Tensor): Value tensor from another modality
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Attended features
        """
        batch_size = query.size(0)
        
        # Project query, key, value
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)
        output = self.out_proj(context)
        
        return output


class CrossModalTransformer(nn.Module):
    """
    Cross-modal transformer for multimodal fusion as specified in MemoCMT paper.
    
    This module implements the transformer-based fusion mechanism that combines
    features from text and voice modalities.
    """
    
    def __init__(self, feature_dim=256, num_heads=8, num_layers=2, dropout_rate=0.1):
        """
        Initialize the cross-modal transformer.
        
        Args:
            feature_dim (int): Dimension of input features
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dropout_rate (float): Dropout rate for regularization
        """
        super(CrossModalTransformer, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # Cross-modal attention layers
        self.text_to_voice_attns = nn.ModuleList([
            CrossModalAttention(feature_dim, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.voice_to_text_attns = nn.ModuleList([
            CrossModalAttention(feature_dim, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.text_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim * 4, feature_dim)
            )
            for _ in range(num_layers)
        ])
        
        self.voice_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim * 4, feature_dim)
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.text_norms1 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_layers)])
        self.text_norms2 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_layers)])
        self.voice_norms1 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_layers)])
        self.voice_norms2 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_layers)])
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, text_features, voice_features):
        """
        Forward pass through the cross-modal transformer.
        
        Args:
            text_features (torch.Tensor): Features from text modality
            voice_features (torch.Tensor): Features from voice modality
            
        Returns:
            tuple: Updated text and voice features after cross-modal attention
        """
        # Ensure features have the same batch size
        batch_size = text_features.size(0)
        
        # Reshape features to have sequence length of 1 if needed
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1)
        if len(voice_features.shape) == 2:
            voice_features = voice_features.unsqueeze(1)
        
        # Process through transformer layers
        for i in range(self.num_layers):
            # Text attends to voice
            text_attn_input = self.text_norms1[i](text_features)
            voice_attn_input = self.voice_norms1[i](voice_features)
            
            text_attn_output = self.text_to_voice_attns[i](
                text_attn_input, voice_attn_input, voice_attn_input
            )
            text_features = text_features + self.dropout(text_attn_output)
            
            # Feed-forward for text
            text_ff_input = self.text_norms2[i](text_features)
            text_ff_output = self.text_ffns[i](text_ff_input)
            text_features = text_features + self.dropout(text_ff_output)
            
            # Voice attends to text
            text_attn_input = self.text_norms1[i](text_features)
            voice_attn_input = self.voice_norms1[i](voice_features)
            
            voice_attn_output = self.voice_to_text_attns[i](
                voice_attn_input, text_attn_input, text_attn_input
            )
            voice_features = voice_features + self.dropout(voice_attn_output)
            
            # Feed-forward for voice
            voice_ff_input = self.voice_norms2[i](voice_features)
            voice_ff_output = self.voice_ffns[i](voice_ff_input)
            voice_features = voice_features + self.dropout(voice_ff_output)
        
        return text_features, voice_features


class MemoCMTFusion(nn.Module):
    """
    MemoCMT fusion model as specified in the paper.
    
    This model combines text and voice features using a cross-modal transformer
    and produces final emotion predictions.
    """
    
    def __init__(self, feature_dim=256, num_classes=4, fusion_method='min', dropout_rate=0.1):
        """
        Initialize the MemoCMT fusion model.
        
        Args:
            feature_dim (int): Dimension of input features
            num_classes (int): Number of emotion classes
            fusion_method (str): Method for fusion ('min', 'max', 'avg', 'concat')
            dropout_rate (float): Dropout rate for regularization
        """
        super(MemoCMTFusion, self).__init__()
        
        self.feature_dim = feature_dim
        self.fusion_method = fusion_method
        
        # Cross-modal transformer
        self.cross_modal_transformer = CrossModalTransformer(
            feature_dim=feature_dim,
            num_heads=8,
            num_layers=2,
            dropout_rate=dropout_rate
        )
        
        # Classification heads
        self.text_classifier = nn.Linear(feature_dim, num_classes)
        self.voice_classifier = nn.Linear(feature_dim, num_classes)
        
        # Fusion classifier (for concat fusion)
        if fusion_method == 'concat':
            self.fusion_classifier = nn.Linear(feature_dim * 2, num_classes)
        else:
            self.fusion_classifier = nn.Linear(feature_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, text_features, voice_features):
        """
        Forward pass through the MemoCMT fusion model.
        
        Args:
            text_features (torch.Tensor): Features from text modality
            voice_features (torch.Tensor): Features from voice modality
            
        Returns:
            tuple: Emotion logits from text, voice, and fusion
        """
        # Apply cross-modal transformer
        text_features, voice_features = self.cross_modal_transformer(text_features, voice_features)
        
        # Squeeze sequence dimension if present
        if len(text_features.shape) > 2:
            text_features = text_features.squeeze(1)
        if len(voice_features.shape) > 2:
            voice_features = voice_features.squeeze(1)
        
        # Apply dropout
        text_features = self.dropout(text_features)
        voice_features = self.dropout(voice_features)
        
        # Get modality-specific predictions
        text_logits = self.text_classifier(text_features)
        voice_logits = self.voice_classifier(voice_features)
        
        # Fuse features based on specified method
        if self.fusion_method == 'min':
            # Min aggregation (as per MemoCMT paper)
            fused_features = torch.min(text_features, voice_features)
        elif self.fusion_method == 'max':
            # Max aggregation
            fused_features = torch.max(text_features, voice_features)
        elif self.fusion_method == 'avg':
            # Average aggregation
            fused_features = (text_features + voice_features) / 2
        elif self.fusion_method == 'concat':
            # Concatenation
            fused_features = torch.cat([text_features, voice_features], dim=1)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        
        # Get fusion predictions
        fusion_logits = self.fusion_classifier(fused_features)
        
        return text_logits, voice_logits, fusion_logits
    
    def predict(self, text_features, voice_features):
        """
        Predict emotions using the fusion model.
        
        Args:
            text_features (torch.Tensor): Features from text modality
            voice_features (torch.Tensor): Features from voice modality
            
        Returns:
            torch.Tensor: Predicted emotion logits from fusion
        """
        _, _, fusion_logits = self.forward(text_features, voice_features)
        return fusion_logits


def load_fusion_model(model_path=None, feature_dim=256, num_classes=4, fusion_method='min', device='cuda'):
    """
    Load a pretrained fusion model or create a new one.
    
    Args:
        model_path (str, optional): Path to pretrained model weights
        feature_dim (int): Dimension of input features
        num_classes (int): Number of emotion classes
        fusion_method (str): Method for fusion
        device (str): Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        MemoCMTFusion: The loaded or newly created model
    """
    # Create model
    model = MemoCMTFusion(
        feature_dim=feature_dim,
        num_classes=num_classes,
        fusion_method=fusion_method
    )
    
    # Load pretrained weights if provided
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded fusion model from {model_path}")
    else:
        print("Initialized new fusion model")
    
    # Move model to device
    model = model.to(device)
    
    return model


def predict_emotion(fusion_model, text_model, voice_model, text_processor, voice_processor, 
                   texts, audio_paths, device='cuda'):
    """
    Predict emotions from multimodal inputs.
    
    Args:
        fusion_model (MemoCMTFusion): The fusion model
        text_model (TextEmotionModel): The text emotion model
        voice_model (VoiceEmotionModel): The voice emotion model
        text_processor (TextProcessor): The text processor
        voice_processor (VoiceProcessor): The voice processor
        texts (list): List of text strings
        audio_paths (list): List of paths to audio files
        device (str): Device to run inference on
        
    Returns:
        dict: Predicted emotion probabilities from text, voice, and fusion
    """
    # Set models to evaluation mode
    fusion_model.eval()
    text_model.eval()
    voice_model.eval()
    
    # Process text inputs
    text_inputs = text_processor.preprocess(texts)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    # Process audio inputs
    voice_features_list = []
    for audio_path in audio_paths:
        voice_features = voice_processor.preprocess(audio_path, device)
        voice_features_list.append(voice_features)
    
    # Stack voice features
    if voice_features_list:
        max_length = max(f.shape[1] for f in voice_features_list)
        padded_features = []
        
        for feat in voice_features_list:
            if feat.shape[1] < max_length:
                padding = torch.zeros(
                    (feat.shape[0], max_length - feat.shape[1], feat.shape[2]),
                    device=feat.device
                )
                feat = torch.cat([feat, padding], dim=1)
            padded_features.append(feat)
        
        voice_features = torch.cat(padded_features, dim=0)
    else:
        # Handle empty list case
        return None
    
    # Extract features from modality-specific models
    with torch.no_grad():
        # Get text features
        _, text_features = text_model(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            token_type_ids=text_inputs.get('token_type_ids', None),
            return_features=True
        )
        
        # Get voice features
        _, voice_features = voice_model(voice_features, return_features=True)
        
        # Get predictions from fusion model
        text_logits, voice_logits, fusion_logits = fusion_model(text_features, voice_features)
    
    # Convert to probabilities
    text_probs = F.softmax(text_logits, dim=1).cpu().numpy()
    voice_probs = F.softmax(voice_logits, dim=1).cpu().numpy()
    fusion_probs = F.softmax(fusion_logits, dim=1).cpu().numpy()
    
    return {
        'text': text_probs,
        'voice': voice_probs,
        'fusion': fusion_probs
    }
