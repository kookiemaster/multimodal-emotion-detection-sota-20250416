"""
Text Emotion Detection Model for MemoCMT Implementation

This module implements the text emotion detection component of the MemoCMT model
as described in the paper "MemoCMT: multimodal emotion recognition using 
cross-modal transformer-based feature fusion" (2025).

The text model uses BERT for text feature extraction as specified in the paper.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import numpy as np

class TextEmotionModel(nn.Module):
    """
    Text emotion detection model based on BERT as specified in MemoCMT paper.
    
    This model extracts features from text using BERT and processes them
    for emotion classification.
    """
    
    def __init__(self, num_classes=4, dropout_rate=0.1, bert_model="bert-base-uncased"):
        """
        Initialize the text emotion detection model.
        
        Args:
            num_classes (int): Number of emotion classes (default: 4 for IEMOCAP)
            dropout_rate (float): Dropout rate for regularization
            bert_model (str): BERT model to use for feature extraction
        """
        super(TextEmotionModel, self).__init__()
        
        # Load BERT model for feature extraction
        self.bert = BertModel.from_pretrained(bert_model)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Feature processing layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.bert_hidden_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(256)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, return_features=False):
        """
        Forward pass through the text emotion model.
        
        Args:
            input_ids (torch.Tensor): Token IDs from BERT tokenizer
            attention_mask (torch.Tensor): Attention mask for BERT
            token_type_ids (torch.Tensor, optional): Token type IDs for BERT
            return_features (bool): Whether to return intermediate features for fusion
            
        Returns:
            torch.Tensor: Emotion class logits
            torch.Tensor (optional): Intermediate features for fusion if return_features=True
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use the [CLS] token representation as the text embedding
        pooled_output = outputs.pooler_output
        
        # Apply dropout for regularization
        x = self.dropout(pooled_output)
        
        # Process through fully connected layers
        features = self.fc1(x)
        features = F.relu(features)
        features = self.layer_norm(features)
        features = self.dropout(features)
        
        # Final classification layer
        logits = self.fc2(features)
        
        if return_features:
            return logits, features
        else:
            return logits
    
    def extract_features(self, input_ids, attention_mask, token_type_ids=None):
        """
        Extract features for fusion with other modalities.
        
        Args:
            input_ids (torch.Tensor): Token IDs from BERT tokenizer
            attention_mask (torch.Tensor): Attention mask for BERT
            token_type_ids (torch.Tensor, optional): Token type IDs for BERT
            
        Returns:
            torch.Tensor: Features for fusion
        """
        _, features = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_features=True
        )
        return features


class TextProcessor:
    """
    Text processing class for emotion detection.
    
    This class handles tokenization and preprocessing of text data
    for the text emotion model.
    """
    
    def __init__(self, bert_model="bert-base-uncased", max_length=128):
        """
        Initialize the text processor.
        
        Args:
            bert_model (str): BERT model to use for tokenization
            max_length (int): Maximum sequence length for tokenization
        """
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.max_length = max_length
        
    def preprocess(self, texts, return_tensors=True):
        """
        Preprocess text data for the model.
        
        Args:
            texts (list): List of text strings to process
            return_tensors (bool): Whether to return PyTorch tensors
            
        Returns:
            dict: Tokenized inputs for the model
        """
        # Tokenize the texts
        encoded_inputs = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt' if return_tensors else None
        )
        
        return encoded_inputs
    
    def batch_preprocess(self, texts, batch_size=32):
        """
        Preprocess text data in batches.
        
        Args:
            texts (list): List of text strings to process
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of tokenized inputs for each batch
        """
        batches = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_inputs = self.preprocess(batch_texts)
            batches.append(batch_inputs)
        
        return batches


def load_text_model(model_path=None, num_classes=4, device='cuda'):
    """
    Load a pretrained text emotion model or create a new one.
    
    Args:
        model_path (str, optional): Path to pretrained model weights
        num_classes (int): Number of emotion classes
        device (str): Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        TextEmotionModel: The loaded or newly created model
    """
    # Create model
    model = TextEmotionModel(num_classes=num_classes)
    
    # Load pretrained weights if provided
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded text model from {model_path}")
    else:
        print("Initialized new text model")
    
    # Move model to device
    model = model.to(device)
    
    return model


def predict_emotion(model, processor, texts, device='cuda'):
    """
    Predict emotions from text inputs.
    
    Args:
        model (TextEmotionModel): The text emotion model
        processor (TextProcessor): The text processor
        texts (list): List of text strings
        device (str): Device to run inference on
        
    Returns:
        numpy.ndarray: Predicted emotion probabilities
    """
    # Preprocess the texts
    inputs = processor.preprocess(texts)
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Set model to evaluation mode
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        logits = model(**inputs)
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=1).cpu().numpy()
    
    return probs
