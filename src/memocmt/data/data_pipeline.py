"""
Data Preprocessing Pipeline for MemoCMT Implementation

This module implements the data preprocessing pipeline for the MemoCMT model
as described in the paper "MemoCMT: multimodal emotion recognition using 
cross-modal transformer-based feature fusion" (2025).

The pipeline processes IEMOCAP and ESD datasets according to the paper specifications.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import librosa
import json
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultimodalEmotionDataset(Dataset):
    """
    Dataset class for multimodal emotion recognition.
    
    This class handles loading and preprocessing of text and audio data
    from IEMOCAP and ESD datasets.
    """
    
    def __init__(self, metadata_path, audio_dir, text_processor, voice_processor=None, 
                 max_text_length=128, dataset_type='iemocap', emotion_map=None, device='cpu'):
        """
        Initialize the multimodal emotion dataset.
        
        Args:
            metadata_path (str): Path to metadata CSV file
            audio_dir (str): Directory containing audio files
            text_processor: Processor for text data
            voice_processor: Processor for voice data (optional)
            max_text_length (int): Maximum text sequence length
            dataset_type (str): Type of dataset ('iemocap' or 'esd')
            emotion_map (dict): Mapping from emotion labels to indices
            device (str): Device to process data on
        """
        self.metadata = pd.read_csv(metadata_path)
        self.audio_dir = audio_dir
        self.text_processor = text_processor
        self.voice_processor = voice_processor
        self.max_text_length = max_text_length
        self.dataset_type = dataset_type
        self.device = device
        
        # Define emotion mapping if not provided
        if emotion_map is None:
            if dataset_type == 'iemocap':
                self.emotion_map = {
                    'angry': 0,
                    'happy': 1,
                    'sad': 2,
                    'neutral': 3
                }
            elif dataset_type == 'esd':
                self.emotion_map = {
                    'neutral': 0,
                    'happy': 1,
                    'angry': 2,
                    'sad': 3,
                    'surprise': 4
                }
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
        else:
            self.emotion_map = emotion_map
        
        # Filter metadata to include only supported emotions
        self.metadata = self.metadata[self.metadata['emotion'].isin(self.emotion_map.keys())]
        
        # Reset index
        self.metadata = self.metadata.reset_index(drop=True)
        
        logger.info(f"Loaded {len(self.metadata)} samples from {dataset_type} dataset")
        logger.info(f"Emotion distribution: {self.metadata['emotion'].value_counts().to_dict()}")
    
    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Sample containing text, audio, and label
        """
        # Get metadata for the sample
        sample = self.metadata.iloc[idx]
        
        # Get emotion label
        emotion = sample['emotion']
        label = self.emotion_map[emotion]
        
        # Process text
        if self.dataset_type == 'iemocap':
            text = sample['text']
        elif self.dataset_type == 'esd':
            # ESD doesn't have text, use a placeholder
            text = ""
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        
        # Tokenize text
        text_inputs = self.text_processor.preprocess([text], return_tensors=True)
        
        # Process audio if voice processor is available
        if self.voice_processor is not None:
            audio_path = sample['audio_path']
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(self.audio_dir, os.path.basename(audio_path))
            
            try:
                audio_features = self.voice_processor.preprocess(audio_path, device=self.device)
            except Exception as e:
                logger.warning(f"Error processing audio file {audio_path}: {e}")
                # Return None for audio features to handle in collate_fn
                audio_features = None
        else:
            audio_features = None
        
        return {
            'text_inputs': text_inputs,
            'audio_features': audio_features,
            'label': label,
            'emotion': emotion,
            'id': sample.get('id', str(idx))
        }
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function for DataLoader.
        
        Args:
            batch (list): List of samples
            
        Returns:
            dict: Batched samples
        """
        # Filter out samples with None audio features
        valid_samples = [sample for sample in batch if sample['audio_features'] is not None]
        
        if not valid_samples:
            return None
        
        # Collate text inputs
        text_inputs = {
            'input_ids': torch.cat([sample['text_inputs']['input_ids'] for sample in valid_samples], dim=0),
            'attention_mask': torch.cat([sample['text_inputs']['attention_mask'] for sample in valid_samples], dim=0)
        }
        
        if 'token_type_ids' in valid_samples[0]['text_inputs']:
            text_inputs['token_type_ids'] = torch.cat(
                [sample['text_inputs']['token_type_ids'] for sample in valid_samples], dim=0
            )
        
        # Collate audio features
        # For audio features, we need to pad to the same length
        max_length = max(sample['audio_features'].shape[1] for sample in valid_samples)
        padded_audio_features = []
        
        for sample in valid_samples:
            features = sample['audio_features']
            if features.shape[1] < max_length:
                padding = torch.zeros(
                    (features.shape[0], max_length - features.shape[1], features.shape[2]),
                    device=features.device
                )
                features = torch.cat([features, padding], dim=1)
            padded_audio_features.append(features)
        
        audio_features = torch.cat(padded_audio_features, dim=0)
        
        # Collate labels
        labels = torch.tensor([sample['label'] for sample in valid_samples], dtype=torch.long)
        
        # Collate metadata
        emotions = [sample['emotion'] for sample in valid_samples]
        ids = [sample['id'] for sample in valid_samples]
        
        return {
            'text_inputs': text_inputs,
            'audio_features': audio_features,
            'labels': labels,
            'emotions': emotions,
            'ids': ids
        }


def create_dataloaders(metadata_path, audio_dir, text_processor, voice_processor=None,
                      batch_size=32, dataset_type='iemocap', emotion_map=None, 
                      num_workers=4, device='cpu'):
    """
    Create DataLoader for the dataset.
    
    Args:
        metadata_path (str): Path to metadata CSV file
        audio_dir (str): Directory containing audio files
        text_processor: Processor for text data
        voice_processor: Processor for voice data (optional)
        batch_size (int): Batch size
        dataset_type (str): Type of dataset ('iemocap' or 'esd')
        emotion_map (dict): Mapping from emotion labels to indices
        num_workers (int): Number of workers for DataLoader
        device (str): Device to process data on
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the dataset
    """
    # Create dataset
    dataset = MultimodalEmotionDataset(
        metadata_path=metadata_path,
        audio_dir=audio_dir,
        text_processor=text_processor,
        voice_processor=voice_processor,
        dataset_type=dataset_type,
        emotion_map=emotion_map,
        device=device
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=MultimodalEmotionDataset.collate_fn,
        pin_memory=True if device == 'cuda' else False
    )
    
    return dataloader


def prepare_iemocap_data(processed_dir, text_processor, voice_processor=None,
                        batch_size=32, num_workers=4, device='cpu'):
    """
    Prepare IEMOCAP dataset for training and testing.
    
    Args:
        processed_dir (str): Directory containing processed IEMOCAP data
        text_processor: Processor for text data
        voice_processor: Processor for voice data (optional)
        batch_size (int): Batch size
        num_workers (int): Number of workers for DataLoader
        device (str): Device to process data on
        
    Returns:
        tuple: Train and test DataLoaders
    """
    # Define paths
    train_metadata_path = os.path.join(processed_dir, 'iemocap_train.csv')
    test_metadata_path = os.path.join(processed_dir, 'iemocap_test.csv')
    audio_dir = os.path.join(processed_dir, 'audio')
    
    # Check if files exist
    for path in [train_metadata_path, test_metadata_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Directory not found: {audio_dir}")
    
    # Create DataLoaders
    train_loader = create_dataloaders(
        metadata_path=train_metadata_path,
        audio_dir=audio_dir,
        text_processor=text_processor,
        voice_processor=voice_processor,
        batch_size=batch_size,
        dataset_type='iemocap',
        num_workers=num_workers,
        device=device
    )
    
    test_loader = create_dataloaders(
        metadata_path=test_metadata_path,
        audio_dir=audio_dir,
        text_processor=text_processor,
        voice_processor=voice_processor,
        batch_size=batch_size,
        dataset_type='iemocap',
        num_workers=num_workers,
        device=device
    )
    
    return train_loader, test_loader


def prepare_esd_data(processed_dir, text_processor, voice_processor=None,
                    batch_size=32, num_workers=4, device='cpu'):
    """
    Prepare ESD dataset for training, validation, and testing.
    
    Args:
        processed_dir (str): Directory containing processed ESD data
        text_processor: Processor for text data
        voice_processor: Processor for voice data (optional)
        batch_size (int): Batch size
        num_workers (int): Number of workers for DataLoader
        device (str): Device to process data on
        
    Returns:
        tuple: Train, validation, and test DataLoaders
    """
    # Define paths
    train_metadata_path = os.path.join(processed_dir, 'esd_train.csv')
    val_metadata_path = os.path.join(processed_dir, 'esd_val.csv')
    test_metadata_path = os.path.join(processed_dir, 'esd_test.csv')
    audio_dir = os.path.join(processed_dir, 'audio')
    
    # Check if files exist
    for path in [train_metadata_path, val_metadata_path, test_metadata_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Directory not found: {audio_dir}")
    
    # Create DataLoaders
    train_loader = create_dataloaders(
        metadata_path=train_metadata_path,
        audio_dir=audio_dir,
        text_processor=text_processor,
        voice_processor=voice_processor,
        batch_size=batch_size,
        dataset_type='esd',
        num_workers=num_workers,
        device=device
    )
    
    val_loader = create_dataloaders(
        metadata_path=val_metadata_path,
        audio_dir=audio_dir,
        text_processor=text_processor,
        voice_processor=voice_processor,
        batch_size=batch_size,
        dataset_type='esd',
        num_workers=num_workers,
        device=device
    )
    
    test_loader = create_dataloaders(
        metadata_path=test_metadata_path,
        audio_dir=audio_dir,
        text_processor=text_processor,
        voice_processor=voice_processor,
        batch_size=batch_size,
        dataset_type='esd',
        num_workers=num_workers,
        device=device
    )
    
    return train_loader, val_loader, test_loader
