"""
Text Data Utilities

This module provides utilities for loading and processing text data for emotion detection.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class TextEmotionDataset(Dataset):
    """
    Dataset class for text emotion data.
    """
    
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            texts (list): List of text samples
            labels (list, optional): List of emotion labels
            tokenizer: Tokenizer for processing text
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove batch dimension added by tokenizer
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # Add label if available
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item

def load_text_data(data_path, split='train'):
    """
    Load text data from file.
    
    Args:
        data_path (str): Path to data directory
        split (str): Data split to load ('train', 'val', or 'test')
        
    Returns:
        tuple: (texts, labels)
    """
    file_path = os.path.join(data_path, f"{split}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load data from CSV
    df = pd.read_csv(file_path)
    
    # Extract texts and labels
    texts = df['text'].tolist()
    labels = df['label'].tolist() if 'label' in df.columns else None
    
    return texts, labels

def create_data_loaders(texts, labels, tokenizer, batch_size=16, max_length=128, train_ratio=0.8):
    """
    Create data loaders for training and validation.
    
    Args:
        texts (list): List of text samples
        labels (list): List of emotion labels
        tokenizer: Tokenizer for processing text
        batch_size (int): Batch size
        max_length (int): Maximum sequence length
        train_ratio (float): Ratio of training data
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Split data into train and validation sets
    train_size = int(len(texts) * train_ratio)
    
    train_texts = texts[:train_size]
    train_labels = labels[:train_size] if labels is not None else None
    
    val_texts = texts[train_size:]
    val_labels = labels[train_size:] if labels is not None else None
    
    # Create datasets
    train_dataset = TextEmotionDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextEmotionDataset(val_texts, val_labels, tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def prepare_example_data():
    """
    Prepare example data for testing.
    
    Returns:
        tuple: (texts, labels)
    """
    texts = [
        "I'm so happy today! Everything is going well.",
        "I'm feeling really sad and disappointed about the news.",
        "That makes me so angry! How could they do that?",
        "I'm a bit nervous about the upcoming presentation.",
        "I'm disgusted by what I just saw in that restaurant.",
        "I was surprised to see her at the party yesterday.",
        "Just another normal day at work, nothing special.",
        "I'm so excited about the upcoming vacation!",
        "I'm terrified of heights, I can't look down.",
        "I'm bored with this routine, need something new."
    ]
    
    # Map to emotion labels: 0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise
    labels = [3, 5, 0, 2, 1, 6, 4, 3, 2, 4]
    
    return texts, labels
