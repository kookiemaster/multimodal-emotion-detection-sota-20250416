"""
Multimodal Fusion Component

This module implements the fusion of text and voice modalities for emotion detection.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultimodalFusion:
    """
    A class for fusing text and voice emotion predictions.
    """
    
    def __init__(self, text_processor=None, voice_processor=None, fusion_method='weighted_average'):
        """
        Initialize the multimodal fusion component.
        
        Args:
            text_processor: Text emotion processor
            voice_processor: Voice emotion processor
            fusion_method (str): Fusion method ('weighted_average', 'attention', or 'max')
        """
        self.text_processor = text_processor
        self.voice_processor = voice_processor
        self.fusion_method = fusion_method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default weights for weighted average fusion
        self.text_weight = 0.6
        self.voice_weight = 0.4
        
        # Emotion labels (ensure consistency between modalities)
        self.emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    
    def set_processors(self, text_processor, voice_processor):
        """
        Set the text and voice processors.
        
        Args:
            text_processor: Text emotion processor
            voice_processor: Voice emotion processor
        """
        self.text_processor = text_processor
        self.voice_processor = voice_processor
    
    def set_fusion_weights(self, text_weight, voice_weight):
        """
        Set weights for weighted average fusion.
        
        Args:
            text_weight (float): Weight for text modality
            voice_weight (float): Weight for voice modality
        """
        # Normalize weights
        total = text_weight + voice_weight
        self.text_weight = text_weight / total
        self.voice_weight = voice_weight / total
    
    def predict_emotion(self, text, audio_path):
        """
        Predict emotion from text and audio.
        
        Args:
            text (str): Input text
            audio_path (str): Path to audio file
            
        Returns:
            dict: Dictionary containing fused emotion predictions and probabilities
        """
        if self.text_processor is None or self.voice_processor is None:
            raise ValueError("Text and voice processors must be set before prediction.")
        
        # Get text emotion prediction
        text_result = self.text_processor.predict_emotion(text)
        
        # Get voice emotion prediction
        voice_result = self.voice_processor.predict_emotion(audio_path)
        
        # Fuse predictions
        fused_result = self.fuse_predictions(text_result, voice_result)
        
        return fused_result
    
    def fuse_predictions(self, text_result, voice_result):
        """
        Fuse text and voice emotion predictions.
        
        Args:
            text_result (dict): Text emotion prediction result
            voice_result (dict): Voice emotion prediction result
            
        Returns:
            dict: Fused emotion prediction result
        """
        # Extract probabilities
        text_probs = np.array([text_result['all_probabilities'][emotion] for emotion in self.emotion_labels])
        voice_probs = np.array([voice_result['all_probabilities'][emotion] for emotion in self.emotion_labels])
        
        # Apply fusion method
        if self.fusion_method == 'weighted_average':
            fused_probs = self.text_weight * text_probs + self.voice_weight * voice_probs
        elif self.fusion_method == 'attention':
            fused_probs = self.attention_fusion(text_probs, voice_probs)
        elif self.fusion_method == 'max':
            fused_probs = np.maximum(text_probs, voice_probs)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Get the predicted class
        predicted_class = np.argmax(fused_probs)
        predicted_emotion = self.emotion_labels[predicted_class]
        
        # Create result dictionary
        result = {
            "predicted_emotion": predicted_emotion,
            "confidence": float(fused_probs[predicted_class]),
            "all_probabilities": {emotion: float(prob) for emotion, prob in zip(self.emotion_labels, fused_probs)},
            "text_prediction": text_result['predicted_emotion'],
            "text_confidence": text_result['confidence'],
            "voice_prediction": voice_result['predicted_emotion'],
            "voice_confidence": voice_result['confidence']
        }
        
        return result
    
    def attention_fusion(self, text_probs, voice_probs):
        """
        Fuse predictions using attention mechanism.
        
        Args:
            text_probs (numpy.ndarray): Text emotion probabilities
            voice_probs (numpy.ndarray): Voice emotion probabilities
            
        Returns:
            numpy.ndarray: Fused probabilities
        """
        # Calculate attention weights based on confidence
        text_conf = np.max(text_probs)
        voice_conf = np.max(voice_probs)
        
        # Apply softmax to get normalized attention weights
        attention_weights = np.array([text_conf, voice_conf])
        attention_weights = np.exp(attention_weights) / np.sum(np.exp(attention_weights))
        
        # Apply attention weights
        fused_probs = attention_weights[0] * text_probs + attention_weights[1] * voice_probs
        
        return fused_probs


class MultimodalFusionModel(nn.Module):
    """
    Neural network model for multimodal fusion of text and voice features.
    """
    
    def __init__(self, text_feature_dim=768, voice_feature_dim=256, hidden_dim=128, num_classes=7):
        """
        Initialize the multimodal fusion model.
        
        Args:
            text_feature_dim (int): Dimension of text features
            voice_feature_dim (int): Dimension of voice features
            hidden_dim (int): Dimension of hidden layer
            num_classes (int): Number of emotion classes
        """
        super(MultimodalFusionModel, self).__init__()
        
        # Feature transformation layers
        self.text_transform = nn.Linear(text_feature_dim, hidden_dim)
        self.voice_transform = nn.Linear(voice_feature_dim, hidden_dim)
        
        # Attention mechanism
        self.text_attention = nn.Linear(hidden_dim, 1)
        self.voice_attention = nn.Linear(hidden_dim, 1)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Classification layer
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, text_features, voice_features):
        """
        Forward pass of the model.
        
        Args:
            text_features (torch.Tensor): Text features
            voice_features (torch.Tensor): Voice features
            
        Returns:
            torch.Tensor: Logits for each emotion class
        """
        # Transform features
        text_hidden = F.relu(self.text_transform(text_features))
        voice_hidden = F.relu(self.voice_transform(voice_features))
        
        # Calculate attention weights
        text_attention = torch.sigmoid(self.text_attention(text_hidden))
        voice_attention = torch.sigmoid(self.voice_attention(voice_hidden))
        
        # Apply attention
        text_attended = text_hidden * text_attention
        voice_attended = voice_hidden * voice_attention
        
        # Concatenate features
        combined = torch.cat([text_attended, voice_attended], dim=1)
        
        # Fusion
        fused = F.relu(self.fusion(combined))
        fused = self.dropout(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits


# Example usage
if __name__ == "__main__":
    # This is a demonstration of how the fusion component would be used
    # In a real scenario, you would use actual text and voice processors
    
    from text.text_processor import TextEmotionProcessor
    from voice.voice_processor import VoiceEmotionProcessor
    
    # Initialize processors
    text_processor = TextEmotionProcessor()
    voice_processor = VoiceEmotionProcessor()
    
    # Load models
    text_processor.load_model()
    voice_processor.load_model()
    
    # Initialize fusion component
    fusion = MultimodalFusion(text_processor, voice_processor, fusion_method='attention')
    
    # Example inputs
    text = "I'm feeling really happy today!"
    audio_path = "test_audio.wav"
    
    # For demonstration, create a dummy audio file if it doesn't exist
    if not os.path.exists(audio_path):
        print(f"Creating dummy audio file: {audio_path}")
        import numpy as np
        import soundfile as sf
        
        # Create a simple sine wave
        sample_rate = 16000
        duration = 3  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Add some noise
        signal += 0.1 * np.random.randn(*signal.shape)
        
        # Save as WAV file
        sf.write(audio_path, signal, sample_rate)
    
    # Predict emotion
    result = fusion.predict_emotion(text, audio_path)
    
    # Print results
    print(f"Text input: \"{text}\"")
    print(f"Audio input: {audio_path}")
    print(f"Text prediction: {result['text_prediction']} (Confidence: {result['text_confidence']:.4f})")
    print(f"Voice prediction: {result['voice_prediction']} (Confidence: {result['voice_confidence']:.4f})")
    print(f"Fused prediction: {result['predicted_emotion']} (Confidence: {result['confidence']:.4f})")
    print("All probabilities:", {k: f"{v:.4f}" for k, v in result['all_probabilities'].items()})
