"""
Voice Emotion Detection Model for MemoCMT Implementation

This module implements the voice emotion detection component of the MemoCMT model
as described in the paper "MemoCMT: multimodal emotion recognition using 
cross-modal transformer-based feature fusion" (2025).

The voice model uses HuBERT for audio feature extraction as specified in the paper.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

class VoiceEmotionModel(nn.Module):
    """
    Voice emotion detection model based on HuBERT as specified in MemoCMT paper.
    
    This model extracts features from audio using HuBERT and processes them
    for emotion classification.
    """
    
    def __init__(self, num_classes=4, dropout_rate=0.1, hubert_hidden_size=768):
        """
        Initialize the voice emotion detection model.
        
        Args:
            num_classes (int): Number of emotion classes (default: 4 for IEMOCAP)
            dropout_rate (float): Dropout rate for regularization
            hubert_hidden_size (int): Hidden size of HuBERT features
        """
        super(VoiceEmotionModel, self).__init__()
        
        # Feature processing layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hubert_hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(256)
        
        # Attention mechanism for temporal aggregation
        self.attention = nn.Sequential(
            nn.Linear(hubert_hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x, return_features=False):
        """
        Forward pass through the voice emotion model.
        
        Args:
            x (torch.Tensor): Audio features from HuBERT (batch_size, seq_len, hidden_size)
            return_features (bool): Whether to return intermediate features for fusion
            
        Returns:
            torch.Tensor: Emotion class logits
            torch.Tensor (optional): Intermediate features for fusion if return_features=True
        """
        # Apply attention to aggregate temporal features
        attention_weights = self.attention(x).squeeze(-1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights to get context vector
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),
            x
        ).squeeze(1)
        
        # Apply dropout for regularization
        x = self.dropout(context_vector)
        
        # Process through fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.layer_norm1(x)
        x = self.dropout(x)
        
        features = self.fc2(x)
        features = F.relu(features)
        features = self.layer_norm2(features)
        features = self.dropout(features)
        
        # Final classification layer
        logits = self.fc3(features)
        
        if return_features:
            return logits, features
        else:
            return logits
    
    def extract_features(self, x):
        """
        Extract features for fusion with other modalities.
        
        Args:
            x (torch.Tensor): Audio features from HuBERT
            
        Returns:
            torch.Tensor: Features for fusion
        """
        _, features = self.forward(x, return_features=True)
        return features


class VoiceProcessor:
    """
    Voice processing class for emotion detection.
    
    This class handles feature extraction and preprocessing of audio data
    for the voice emotion model.
    """
    
    def __init__(self, sample_rate=16000, max_duration=10, use_hubert=True):
        """
        Initialize the voice processor.
        
        Args:
            sample_rate (int): Sample rate for audio processing
            max_duration (float): Maximum duration of audio in seconds
            use_hubert (bool): Whether to use HuBERT for feature extraction
        """
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.use_hubert = use_hubert
        
        # Load HuBERT model if specified
        if use_hubert:
            try:
                from transformers import HubertModel, Wav2Vec2FeatureExtractor
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
                self.hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
            except ImportError:
                print("Warning: transformers library not available. Using fallback features.")
                self.use_hubert = False
        
    def load_audio(self, audio_path):
        """
        Load audio file and resample if necessary.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            numpy.ndarray: Audio waveform
            int: Sample rate
        """
        # Load audio file
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Trim silence
        waveform, _ = librosa.effects.trim(waveform, top_db=20)
        
        # Ensure consistent length
        max_length = int(self.max_duration * self.sample_rate)
        if len(waveform) > max_length:
            waveform = waveform[:max_length]
        else:
            # Pad with zeros if shorter
            waveform = np.pad(waveform, (0, max(0, max_length - len(waveform))), 'constant')
        
        return waveform, sr
    
    def extract_hubert_features(self, waveform, device='cuda'):
        """
        Extract HuBERT features from audio waveform.
        
        Args:
            waveform (numpy.ndarray): Audio waveform
            device (str): Device to run feature extraction on
            
        Returns:
            torch.Tensor: HuBERT features
        """
        if not self.use_hubert:
            # Fallback to MFCC features if HuBERT not available
            return self.extract_fallback_features(waveform)
        
        # Prepare inputs for HuBERT
        inputs = self.feature_extractor(
            waveform, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.hubert_model(**inputs)
            
        # Get the hidden states
        hidden_states = outputs.last_hidden_state
        
        return hidden_states
    
    def extract_fallback_features(self, waveform):
        """
        Extract fallback features (MFCCs) when HuBERT is not available.
        
        Args:
            waveform (numpy.ndarray): Audio waveform
            
        Returns:
            torch.Tensor: MFCC features
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=waveform, 
            sr=self.sample_rate, 
            n_mfcc=40,
            hop_length=512,
            n_fft=2048
        )
        
        # Add delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Combine features
        features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs], axis=0)
        
        # Transpose to (time, features)
        features = features.T
        
        # Convert to tensor
        features = torch.from_numpy(features).float()
        
        # Add batch dimension
        features = features.unsqueeze(0)
        
        return features
    
    def preprocess(self, audio_path, device='cuda'):
        """
        Preprocess audio file for the model.
        
        Args:
            audio_path (str): Path to audio file
            device (str): Device to run preprocessing on
            
        Returns:
            torch.Tensor: Processed audio features
        """
        # Load audio
        waveform, _ = self.load_audio(audio_path)
        
        # Extract features
        if self.use_hubert:
            features = self.extract_hubert_features(waveform, device)
        else:
            features = self.extract_fallback_features(waveform)
        
        return features
    
    def batch_preprocess(self, audio_paths, batch_size=8, device='cuda'):
        """
        Preprocess audio files in batches.
        
        Args:
            audio_paths (list): List of paths to audio files
            batch_size (int): Batch size for processing
            device (str): Device to run preprocessing on
            
        Returns:
            list: List of processed audio features for each batch
        """
        batches = []
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i+batch_size]
            batch_features = []
            
            for path in batch_paths:
                features = self.preprocess(path, device)
                batch_features.append(features)
            
            # Stack features into a batch
            if self.use_hubert:
                # For HuBERT features, pad to the same length
                max_length = max(f.shape[1] for f in batch_features)
                padded_features = []
                
                for feat in batch_features:
                    if feat.shape[1] < max_length:
                        padding = torch.zeros(
                            (feat.shape[0], max_length - feat.shape[1], feat.shape[2]),
                            device=feat.device
                        )
                        feat = torch.cat([feat, padding], dim=1)
                    padded_features.append(feat)
                
                batch_tensor = torch.cat(padded_features, dim=0)
            else:
                # For fallback features, they should already be the same length
                batch_tensor = torch.cat(batch_features, dim=0)
            
            batches.append(batch_tensor)
        
        return batches


def load_voice_model(model_path=None, num_classes=4, device='cuda'):
    """
    Load a pretrained voice emotion model or create a new one.
    
    Args:
        model_path (str, optional): Path to pretrained model weights
        num_classes (int): Number of emotion classes
        device (str): Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        VoiceEmotionModel: The loaded or newly created model
    """
    # Create model
    model = VoiceEmotionModel(num_classes=num_classes)
    
    # Load pretrained weights if provided
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded voice model from {model_path}")
    else:
        print("Initialized new voice model")
    
    # Move model to device
    model = model.to(device)
    
    return model


def predict_emotion(model, processor, audio_paths, device='cuda'):
    """
    Predict emotions from audio inputs.
    
    Args:
        model (VoiceEmotionModel): The voice emotion model
        processor (VoiceProcessor): The voice processor
        audio_paths (list): List of paths to audio files
        device (str): Device to run inference on
        
    Returns:
        numpy.ndarray: Predicted emotion probabilities
    """
    # Set model to evaluation mode
    model.eval()
    
    # Process audio in batches
    all_probs = []
    batches = processor.batch_preprocess(audio_paths, device=device)
    
    for batch_features in batches:
        # Get predictions
        with torch.no_grad():
            logits = model(batch_features)
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    
    # Combine batch results
    if all_probs:
        return np.vstack(all_probs)
    else:
        return np.array([])
