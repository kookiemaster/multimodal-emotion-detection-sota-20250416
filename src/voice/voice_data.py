"""
Voice Data Utilities

This module provides utilities for loading and processing voice data for emotion detection.
"""

import os
import numpy as np
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset, DataLoader

class VoiceEmotionDataset(Dataset):
    """
    Dataset class for voice emotion data.
    """
    
    def __init__(self, audio_paths, labels=None, sample_rate=16000, n_mels=128, 
                 window_size=0.025, hop_size=0.01, segment_length=3.0):
        """
        Initialize the dataset.
        
        Args:
            audio_paths (list): List of paths to audio files
            labels (list, optional): List of emotion labels
            sample_rate (int): Audio sample rate
            n_mels (int): Number of mel bands
            window_size (float): Window size in seconds
            hop_size (float): Hop size in seconds
            segment_length (float): Length of audio segments in seconds
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.window_size = window_size
        self.hop_size = hop_size
        self.segment_length = segment_length
        
        # Calculate window and hop length in samples
        self.win_length = int(window_size * sample_rate)
        self.hop_length = int(hop_size * sample_rate)
        
        # Calculate segment length in frames
        self.segment_frames = int(segment_length / hop_size)
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        
        # Load audio
        try:
            signal, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return a dummy feature if loading fails
            features = np.zeros((1, self.n_mels, self.segment_frames))
            if self.labels is not None:
                return {'features': torch.FloatTensor(features), 'labels': torch.tensor(0)}
            else:
                return {'features': torch.FloatTensor(features)}
        
        # Extract features
        features = self.extract_features(signal)
        
        # Create item dictionary
        item = {'features': torch.FloatTensor(features)}
        
        # Add label if available
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item
    
    def extract_features(self, signal):
        """
        Extract mel-spectrogram features from audio signal.
        
        Args:
            signal (numpy.ndarray): Audio signal
            
        Returns:
            numpy.ndarray: Mel-spectrogram features
        """
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.win_length,
            hop_length=self.hop_length
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Segment features
        segments = self.segment_features(log_mel_spec)
        
        return segments
    
    def segment_features(self, features):
        """
        Segment features into fixed-length windows.
        
        Args:
            features (numpy.ndarray): Mel-spectrogram features
            
        Returns:
            numpy.ndarray: Segmented features
        """
        # Get total number of frames
        total_frames = features.shape[1]
        
        # If features are shorter than segment length, pad with zeros
        if total_frames < self.segment_frames:
            pad_width = self.segment_frames - total_frames
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
            segments = np.expand_dims(features, axis=0)
        else:
            # Create segments with overlap
            stride = self.segment_frames // 2  # 50% overlap
            n_segments = (total_frames - self.segment_frames) // stride + 1
            segments = np.zeros((n_segments, self.n_mels, self.segment_frames))
            
            for i in range(n_segments):
                start = i * stride
                end = start + self.segment_frames
                segments[i] = features[:, start:end]
        
        return segments

def load_voice_data(data_path, split='train'):
    """
    Load voice data from file.
    
    Args:
        data_path (str): Path to data directory
        split (str): Data split to load ('train', 'val', or 'test')
        
    Returns:
        tuple: (audio_paths, labels)
    """
    file_path = os.path.join(data_path, f"{split}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load data from CSV
    df = pd.read_csv(file_path)
    
    # Extract audio paths and labels
    audio_paths = df['path'].tolist()
    labels = df['label'].tolist() if 'label' in df.columns else None
    
    return audio_paths, labels

def create_data_loaders(audio_paths, labels, batch_size=16, train_ratio=0.8, **dataset_kwargs):
    """
    Create data loaders for training and validation.
    
    Args:
        audio_paths (list): List of paths to audio files
        labels (list): List of emotion labels
        batch_size (int): Batch size
        train_ratio (float): Ratio of training data
        **dataset_kwargs: Additional arguments for VoiceEmotionDataset
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Split data into train and validation sets
    train_size = int(len(audio_paths) * train_ratio)
    
    train_paths = audio_paths[:train_size]
    train_labels = labels[:train_size] if labels is not None else None
    
    val_paths = audio_paths[train_size:]
    val_labels = labels[train_size:] if labels is not None else None
    
    # Create datasets
    train_dataset = VoiceEmotionDataset(train_paths, train_labels, **dataset_kwargs)
    val_dataset = VoiceEmotionDataset(val_paths, val_labels, **dataset_kwargs)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def prepare_example_data():
    """
    Prepare example data for testing.
    
    Returns:
        tuple: (audio_paths, labels)
    """
    # Create dummy audio files for demonstration
    os.makedirs("dummy_audio", exist_ok=True)
    
    audio_paths = []
    for i in range(10):
        path = f"dummy_audio/example_{i}.wav"
        audio_paths.append(path)
        
        # Create dummy audio file if it doesn't exist
        if not os.path.exists(path):
            # Generate random noise
            duration = 3.0  # seconds
            sr = 16000
            signal = np.random.randn(int(duration * sr))
            
            # Save as WAV file
            librosa.output.write_wav(path, signal, sr)
    
    # Map to emotion labels: 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise
    labels = [0, 1, 2, 3, 4, 5, 6, 3, 4, 5]
    
    return audio_paths, labels
