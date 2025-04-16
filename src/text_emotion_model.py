"""
Text Emotion Detection Model for Multimodal Emotion Recognition

This module implements the text emotion detection component based on the MemoCMT paper.
It uses a transformer-based architecture to extract features from text inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

class TextEmotionModel(nn.Module):
    """
    Text emotion detection model based on transformer architecture.
    
    This model uses BERT or RoBERTa as the backbone for text feature extraction,
    followed by a classification head for emotion prediction.
    """
    
    def __init__(self, num_classes=8, dropout_rate=0.1, model_name="bert-base-uncased", hidden_dim=256):
        """
        Initialize the text emotion detection model.
        
        Args:
            num_classes (int): Number of emotion classes to predict
            dropout_rate (float): Dropout rate for regularization
            model_name (str): Name of the pretrained model to use
            hidden_dim (int): Dimension of the hidden layer
        """
        super(TextEmotionModel, self).__init__()
        
        self.model_name = model_name
        
        # Initialize the appropriate transformer model based on model_name
        if 'bert' in model_name.lower():
            self.transformer = BertModel.from_pretrained(model_name)
            self.feature_dim = self.transformer.config.hidden_size
        elif 'roberta' in model_name.lower():
            self.transformer = RobertaModel.from_pretrained(model_name)
            self.feature_dim = self.transformer.config.hidden_size
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Attention mechanism for sentence-level representation
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_features=False):
        """
        Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Token IDs of input text
            attention_mask (torch.Tensor): Attention mask for input text
            token_type_ids (torch.Tensor): Token type IDs for BERT models
            return_features (bool): Whether to return features along with logits
            
        Returns:
            torch.Tensor: Logits for emotion classification
            torch.Tensor (optional): Text features if return_features is True
        """
        # Prepare inputs based on model type
        if 'bert' in self.model_name.lower():
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:  # RoBERTa doesn't use token_type_ids
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get the hidden states from the transformer
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply attention mechanism to get sentence representation
        attention_weights = self.attention(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask to attention weights if provided
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax to get normalized weights
        attention_weights = F.softmax(attention_weights, dim=1)  # [batch_size, seq_len]
        
        # Apply attention weights to get context vector
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, seq_len]
            hidden_states  # [batch_size, seq_len, hidden_size]
        ).squeeze(1)  # [batch_size, hidden_size]
        
        # Get logits from classifier
        logits = self.classifier(context_vector)
        
        if return_features:
            return logits, context_vector
        else:
            return logits


class TextProcessor:
    """
    Text processor for preparing text inputs for the model.
    
    This class handles tokenization and preprocessing of text inputs.
    """
    
    def __init__(self, model_name="bert-base-uncased", max_length=128):
        """
        Initialize the text processor.
        
        Args:
            model_name (str): Name of the pretrained model to use for tokenization
            max_length (int): Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Initialize the appropriate tokenizer based on model_name
        if 'bert' in model_name.lower():
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        elif 'roberta' in model_name.lower():
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def process(self, texts):
        """
        Process a batch of texts.
        
        Args:
            texts (list): List of text strings to process
            
        Returns:
            dict: Dictionary of tokenized inputs
        """
        # Tokenize the texts
        tokenized = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return tokenized
    
    def process_single(self, text):
        """
        Process a single text.
        
        Args:
            text (str): Text string to process
            
        Returns:
            dict: Dictionary of tokenized inputs
        """
        return self.process([text])


def create_emotion_mapping(dataset_name="iemocap"):
    """
    Create a mapping from emotion labels to indices based on the dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        dict: Mapping from emotion labels to indices
    """
    if dataset_name.lower() == "iemocap":
        # Based on the analysis of IEMOCAP_Final.csv
        return {
            'Frustration': 0,
            'Excited': 1,
            'Anger': 2,
            'Sadness': 3,
            'Neutral state': 4,
            'Happiness': 5,
            'Fear': 6,
            'Surprise': 7
        }
    elif dataset_name.lower() == "meld":
        return {
            'neutral': 0,
            'joy': 1,
            'sadness': 2,
            'anger': 3,
            'surprise': 4,
            'fear': 5,
            'disgust': 6
        }
    elif dataset_name.lower() == "esd":
        return {
            'neutral': 0,
            'happy': 1,
            'angry': 2,
            'sad': 3,
            'surprise': 4
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_pretrained_text_model(model_path, num_classes=8, model_name="bert-base-uncased"):
    """
    Load a pretrained text emotion model.
    
    Args:
        model_path (str): Path to the pretrained model
        num_classes (int): Number of emotion classes
        model_name (str): Name of the pretrained transformer model
        
    Returns:
        TextEmotionModel: Loaded model
    """
    model = TextEmotionModel(num_classes=num_classes, model_name=model_name)
    model.load_state_dict(torch.load(model_path))
    return model


if __name__ == "__main__":
    # Example usage
    model = TextEmotionModel(num_classes=8)
    processor = TextProcessor()
    
    # Example text
    text = "I am feeling really happy today!"
    
    # Process the text
    inputs = processor.process_single(text)
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(model.transformer.device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        logits = model(**inputs)
    
    # Print the predicted emotion
    emotion_mapping = create_emotion_mapping()
    predicted_idx = torch.argmax(logits, dim=1).item()
    predicted_emotion = [k for k, v in emotion_mapping.items() if v == predicted_idx][0]
    
    print(f"Text: {text}")
    print(f"Predicted emotion: {predicted_emotion}")
