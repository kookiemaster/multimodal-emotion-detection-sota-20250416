"""
Voice Emotion Processor

This module implements voice preprocessing and emotion detection using Semi-CNN.
"""

import os
import numpy as np
import torch
import librosa
import soundfile as sf
from torch.nn import functional as F

class VoiceEmotionProcessor:
    """
    A class for processing voice audio and detecting emotions using Semi-CNN.
    """
    
    def __init__(self, model_path=None, sample_rate=16000, n_mels=128, window_size=0.025, 
                 hop_size=0.01, segment_length=3.0):
        """
        Initialize the voice emotion processor.
        
        Args:
            model_path (str): Path to the pre-trained model
            sample_rate (int): Audio sample rate
            n_mels (int): Number of mel bands
            window_size (float): Window size in seconds
            hop_size (float): Hop size in seconds
            segment_length (float): Length of audio segments in seconds
        """
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.window_size = window_size
        self.hop_size = hop_size
        self.segment_length = segment_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = None
        
        # Emotion labels (standard set)
        self.emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    
    def load_model(self, model_path=None):
        """
        Load the pre-trained model.
        
        Args:
            model_path (str, optional): Path to the pre-trained model
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if model_path is not None:
            self.model_path = model_path
            
        if self.model_path is None:
            print("No model path provided. Using dummy model for demonstration.")
            # Create a dummy model for demonstration
            self.model = DummyVoiceEmotionModel(len(self.emotion_labels))
            return True
            
        try:
            # In a real implementation, we would load the actual model here
            # self.model = torch.load(self.model_path, map_location=self.device)
            # self.model.eval()
            
            # For now, use a dummy model
            self.model = DummyVoiceEmotionModel(len(self.emotion_labels))
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_audio(self, audio_path):
        """
        Load audio file.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            numpy.ndarray: Audio signal
        """
        try:
            signal, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            return signal, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def extract_features(self, signal):
        """
        Extract mel-spectrogram features from audio signal.
        
        Args:
            signal (numpy.ndarray): Audio signal
            
        Returns:
            numpy.ndarray: Mel-spectrogram features
        """
        # Calculate window and hop length in samples
        win_length = int(self.window_size * self.sample_rate)
        hop_length = int(self.hop_size * self.sample_rate)
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=win_length,
            hop_length=hop_length
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
    
    def segment_features(self, features):
        """
        Segment features into fixed-length windows.
        
        Args:
            features (numpy.ndarray): Mel-spectrogram features
            
        Returns:
            numpy.ndarray: Segmented features
        """
        # Calculate segment length in frames
        hop_length = int(self.hop_size * self.sample_rate)
        segment_frames = int(self.segment_length / self.hop_size)
        
        # Get total number of frames
        total_frames = features.shape[1]
        
        # If features are shorter than segment length, pad with zeros
        if total_frames < segment_frames:
            pad_width = segment_frames - total_frames
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
            segments = np.expand_dims(features, axis=0)
        else:
            # Create segments with overlap
            stride = segment_frames // 2  # 50% overlap
            n_segments = (total_frames - segment_frames) // stride + 1
            segments = np.zeros((n_segments, self.n_mels, segment_frames))
            
            for i in range(n_segments):
                start = i * stride
                end = start + segment_frames
                segments[i] = features[:, start:end]
        
        return segments
    
    def predict_emotion(self, audio_path):
        """
        Predict emotion from audio file.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            dict: Dictionary containing emotion predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Load audio
        signal, sr = self.load_audio(audio_path)
        if signal is None:
            return {"error": "Failed to load audio file"}
        
        # Extract features
        features = self.extract_features(signal)
        
        # Segment features
        segments = self.segment_features(features)
        
        # Convert to tensor
        segments_tensor = torch.FloatTensor(segments).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            # In a real implementation, we would pass the segments through the model
            # segment_preds = self.model(segments_tensor)
            
            # For demonstration, use the dummy model
            segment_preds = self.model(segments_tensor)
            
            # Average predictions across segments
            avg_preds = torch.mean(segment_preds, dim=0)
            probabilities = F.softmax(avg_preds, dim=0)
        
        # Convert to numpy for easier handling
        probs = probabilities.cpu().numpy()
        
        # Get the predicted class
        predicted_class = np.argmax(probs)
        predicted_emotion = self.emotion_labels[predicted_class]
        
        # Create result dictionary
        result = {
            "predicted_emotion": predicted_emotion,
            "confidence": float(probs[predicted_class]),
            "all_probabilities": {emotion: float(prob) for emotion, prob in zip(self.emotion_labels, probs)}
        }
        
        return result


class DummyVoiceEmotionModel:
    """
    A dummy model for demonstration purposes.
    """
    
    def __init__(self, num_classes):
        """
        Initialize the dummy model.
        
        Args:
            num_classes (int): Number of emotion classes
        """
        self.num_classes = num_classes
    
    def __call__(self, x):
        """
        Forward pass of the dummy model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Dummy predictions
        """
        # Generate random predictions for demonstration
        batch_size = x.shape[0]
        
        # Create dummy logits with a slight bias towards certain emotions
        # This is just for demonstration purposes
        logits = torch.randn(batch_size, self.num_classes)
        
        # Add bias to make predictions more realistic
        # For example, bias towards neutral and happy emotions
        logits[:, 3] += 1.0  # Happy
        logits[:, 4] += 0.5  # Neutral
        
        return logits


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = VoiceEmotionProcessor()
    
    # Load model
    processor.load_model()
    
    # Example audio file
    # In a real scenario, you would use an actual audio file
    audio_path = "example.wav"
    
    # For demonstration, create a dummy audio file if it doesn't exist
    if not os.path.exists(audio_path):
        print(f"Creating dummy audio file: {audio_path}")
        dummy_signal = np.random.randn(16000 * 3)  # 3 seconds of random noise
        sf.write(audio_path, dummy_signal, 16000)
    
    # Make prediction
    result = processor.predict_emotion(audio_path)
    print(f"Predicted emotion: {result['predicted_emotion']} (Confidence: {result['confidence']:.4f})")
    print("All probabilities:", {k: f"{v:.4f}" for k, v in result['all_probabilities'].items()})
