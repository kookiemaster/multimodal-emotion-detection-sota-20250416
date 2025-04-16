"""
Data Preprocessing Pipeline

This module implements the data preprocessing pipeline for multimodal emotion detection.
"""

import os
import pandas as pd
import numpy as np
import torch
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

class MultimodalEmotionDataset(Dataset):
    """
    Dataset class for multimodal emotion data.
    """
    
    def __init__(self, data_df, text_tokenizer=None, sample_rate=16000, n_mels=128, 
                 window_size=0.025, hop_size=0.01, segment_length=3.0, max_text_length=128):
        """
        Initialize the dataset.
        
        Args:
            data_df (pandas.DataFrame): DataFrame containing text, audio paths, and labels
            text_tokenizer: Tokenizer for processing text
            sample_rate (int): Audio sample rate
            n_mels (int): Number of mel bands
            window_size (float): Window size in seconds
            hop_size (float): Hop size in seconds
            segment_length (float): Length of audio segments in seconds
            max_text_length (int): Maximum text sequence length
        """
        self.data_df = data_df
        self.text_tokenizer = text_tokenizer
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.window_size = window_size
        self.hop_size = hop_size
        self.segment_length = segment_length
        self.max_text_length = max_text_length
        
        # Calculate window and hop length in samples
        self.win_length = int(window_size * sample_rate)
        self.hop_length = int(hop_size * sample_rate)
        
        # Calculate segment length in frames
        self.segment_frames = int(segment_length / hop_size)
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        
        # Process text
        text = row['text']
        text_features = self.process_text(text)
        
        # Process audio
        audio_path = row['audio_path']
        audio_features = self.process_audio(audio_path)
        
        # Create item dictionary
        item = {
            'text_features': text_features,
            'audio_features': audio_features
        }
        
        # Add label if available
        if 'label' in row:
            item['label'] = torch.tensor(row['label'], dtype=torch.long)
            
        return item
    
    def process_text(self, text):
        """
        Process text data.
        
        Args:
            text (str): Input text
            
        Returns:
            torch.Tensor: Processed text features
        """
        if self.text_tokenizer is None:
            # If no tokenizer is provided, return dummy features
            return torch.zeros(768)  # Typical BERT/DeBERTa embedding size
        
        # Tokenize text
        encoding = self.text_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        
        # Remove batch dimension added by tokenizer
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        
        return encoding
    
    def process_audio(self, audio_path):
        """
        Process audio data.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            torch.Tensor: Processed audio features
        """
        try:
            # Load audio
            signal, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Extract features
            features = self.extract_audio_features(signal)
            
            return torch.FloatTensor(features)
        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            # Return dummy features
            return torch.zeros((1, self.n_mels, self.segment_frames))
    
    def extract_audio_features(self, signal):
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
        segments = self.segment_audio_features(log_mel_spec)
        
        return segments
    
    def segment_audio_features(self, features):
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


class DataPreprocessor:
    """
    Class for preprocessing multimodal emotion data.
    """
    
    def __init__(self, text_tokenizer=None, sample_rate=16000, n_mels=128, 
                 window_size=0.025, hop_size=0.01, segment_length=3.0, max_text_length=128):
        """
        Initialize the data preprocessor.
        
        Args:
            text_tokenizer: Tokenizer for processing text
            sample_rate (int): Audio sample rate
            n_mels (int): Number of mel bands
            window_size (float): Window size in seconds
            hop_size (float): Hop size in seconds
            segment_length (float): Length of audio segments in seconds
            max_text_length (int): Maximum text sequence length
        """
        self.text_tokenizer = text_tokenizer
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.window_size = window_size
        self.hop_size = hop_size
        self.segment_length = segment_length
        self.max_text_length = max_text_length
    
    def load_data(self, data_path, split='train'):
        """
        Load data from file.
        
        Args:
            data_path (str): Path to data directory
            split (str): Data split to load ('train', 'val', or 'test')
            
        Returns:
            pandas.DataFrame: DataFrame containing text, audio paths, and labels
        """
        file_path = os.path.join(data_path, f"{split}.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data from CSV
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist
        required_columns = ['text', 'audio_path']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data file")
        
        return df
    
    def create_dataset(self, data_df):
        """
        Create a dataset from a DataFrame.
        
        Args:
            data_df (pandas.DataFrame): DataFrame containing text, audio paths, and labels
            
        Returns:
            MultimodalEmotionDataset: Dataset for multimodal emotion data
        """
        return MultimodalEmotionDataset(
            data_df,
            text_tokenizer=self.text_tokenizer,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            window_size=self.window_size,
            hop_size=self.hop_size,
            segment_length=self.segment_length,
            max_text_length=self.max_text_length
        )
    
    def create_data_loaders(self, data_df, batch_size=16, train_ratio=0.8, val_ratio=0.1):
        """
        Create data loaders for training, validation, and testing.
        
        Args:
            data_df (pandas.DataFrame): DataFrame containing text, audio paths, and labels
            batch_size (int): Batch size
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        # Shuffle data
        data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split data
        n_samples = len(data_df)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        train_df = data_df.iloc[:train_size]
        val_df = data_df.iloc[train_size:train_size+val_size]
        test_df = data_df.iloc[train_size+val_size:]
        
        # Create datasets
        train_dataset = self.create_dataset(train_df)
        val_dataset = self.create_dataset(val_df)
        test_dataset = self.create_dataset(test_df)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    
    def prepare_example_data(self, n_samples=10):
        """
        Prepare example data for testing.
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pandas.DataFrame: DataFrame containing example data
        """
        # Create directory for dummy audio files
        os.makedirs("dummy_audio", exist_ok=True)
        
        data = []
        for i in range(n_samples):
            # Create text
            emotions = ["happy", "sad", "angry", "fearful", "neutral", "surprised", "disgusted"]
            emotion = emotions[i % len(emotions)]
            text = f"I am feeling {emotion} today because of what happened."
            
            # Create audio path
            audio_path = f"dummy_audio/example_{i}.wav"
            
            # Create dummy audio file if it doesn't exist
            if not os.path.exists(audio_path):
                # Generate random noise
                duration = 3.0  # seconds
                sr = self.sample_rate
                signal = np.random.randn(int(duration * sr))
                
                # Save as WAV file
                sf.write(audio_path, signal, sr)
            
            # Create label (0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise)
            label_map = {"angry": 0, "disgusted": 1, "fearful": 2, "happy": 3, "neutral": 4, "sad": 5, "surprised": 6}
            label = label_map.get(emotion, 4)  # Default to neutral if not found
            
            data.append({
                'text': text,
                'audio_path': audio_path,
                'label': label
            })
        
        return pd.DataFrame(data)


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Prepare example data
    data_df = preprocessor.prepare_example_data()
    
    # Create dataset
    dataset = preprocessor.create_dataset(data_df)
    
    # Create data loaders
    train_loader, val_loader, test_loader = preprocessor.create_data_loaders(data_df)
    
    # Print dataset information
    print(f"Dataset size: {len(dataset)}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print("\nSample data:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: Tensor of shape {value.shape}")
        else:
            print(f"{key}: {value}")
