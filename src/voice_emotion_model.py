"""
Voice Emotion Detection Model for Multimodal Emotion Recognition

This module implements the voice emotion detection component based on the MemoCMT paper.
It uses a CNN-based architecture to extract features from audio inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

class VoiceEmotionModel(nn.Module):
    """
    Voice emotion detection model based on CNN architecture.
    
    This model uses a CNN-based architecture for audio feature extraction,
    followed by a classification head for emotion prediction.
    """
    
    def __init__(self, num_classes=8, dropout_rate=0.1, input_dim=40, hidden_dim=256):
        """
        Initialize the voice emotion detection model.
        
        Args:
            num_classes (int): Number of emotion classes to predict
            dropout_rate (float): Dropout rate for regularization
            input_dim (int): Dimension of input features (e.g., MFCCs)
            hidden_dim (int): Dimension of the hidden layer
        """
        super(VoiceEmotionModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Attention mechanism for temporal aggregation
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, return_features=False):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input audio features [batch_size, time_steps, features]
            return_features (bool): Whether to return features along with logits
            
        Returns:
            torch.Tensor: Logits for emotion classification
            torch.Tensor (optional): Audio features if return_features is True
        """
        # Transpose to [batch_size, features, time_steps] for CNN
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Transpose back to [batch_size, time_steps, features]
        x = x.transpose(1, 2)
        
        # Apply attention mechanism
        attention_weights = self.attention(x).squeeze(-1)  # [batch_size, time_steps]
        attention_weights = F.softmax(attention_weights, dim=1)  # [batch_size, time_steps]
        
        # Apply attention weights to get context vector
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, time_steps]
            x  # [batch_size, time_steps, features]
        ).squeeze(1)  # [batch_size, features]
        
        # Get logits from classifier
        logits = self.classifier(context_vector)
        
        if return_features:
            return logits, context_vector
        else:
            return logits


class VoiceProcessor:
    """
    Voice processor for preparing audio inputs for the model.
    
    This class handles feature extraction from audio files.
    """
    
    def __init__(self, sample_rate=16000, n_mfcc=40, max_duration=10):
        """
        Initialize the voice processor.
        
        Args:
            sample_rate (int): Sample rate for audio processing
            n_mfcc (int): Number of MFCCs to extract
            max_duration (float): Maximum duration of audio in seconds
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_duration = max_duration
        self.max_frames = int(max_duration * sample_rate)
    
    def extract_features(self, audio_path):
        """
        Extract features from an audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            np.ndarray: Extracted features
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Pad or truncate to max_frames
            if len(y) > self.max_frames:
                y = y[:self.max_frames]
            else:
                y = np.pad(y, (0, max(0, self.max_frames - len(y))), mode='constant')
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            
            # Transpose to [time_steps, features]
            mfccs = mfccs.T
            
            return mfccs
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            # Return zeros if extraction fails
            return np.zeros((int(self.max_duration * 100), self.n_mfcc))
    
    def process_batch(self, audio_paths):
        """
        Process a batch of audio files.
        
        Args:
            audio_paths (list): List of paths to audio files
            
        Returns:
            torch.Tensor: Batch of processed features
        """
        features = []
        for path in audio_paths:
            feat = self.extract_features(path)
            features.append(feat)
        
        # Convert to tensor
        features_tensor = torch.tensor(np.array(features), dtype=torch.float32)
        
        return features_tensor
    
    def process_single(self, audio_path):
        """
        Process a single audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            torch.Tensor: Processed features
        """
        feat = self.extract_features(audio_path)
        return torch.tensor(feat, dtype=torch.float32).unsqueeze(0)


def load_pretrained_voice_model(model_path, num_classes=8):
    """
    Load a pretrained voice emotion model.
    
    Args:
        model_path (str): Path to the pretrained model
        num_classes (int): Number of emotion classes
        
    Returns:
        VoiceEmotionModel: Loaded model
    """
    model = VoiceEmotionModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    return model


if __name__ == "__main__":
    # Example usage
    model = VoiceEmotionModel(num_classes=8)
    processor = VoiceProcessor()
    
    # Example audio file (replace with actual path)
    audio_path = "example.wav"
    
    try:
        # Process the audio
        features = processor.process_single(audio_path)
        
        # Move features to the same device as the model
        features = features.to(next(model.parameters()).device)
        
        # Get predictions
        with torch.no_grad():
            logits = model(features)
        
        # Print the predicted emotion
        from src.text_emotion_model import create_emotion_mapping
        emotion_mapping = create_emotion_mapping()
        predicted_idx = torch.argmax(logits, dim=1).item()
        predicted_emotion = [k for k, v in emotion_mapping.items() if v == predicted_idx][0]
        
        print(f"Audio: {audio_path}")
        print(f"Predicted emotion: {predicted_emotion}")
    except Exception as e:
        print(f"Error processing audio: {e}")
