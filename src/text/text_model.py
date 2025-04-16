"""
Text Emotion Detection Model

This module implements the DeBERTa model for text emotion detection.
"""

import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config

class TextEmotionModel(nn.Module):
    """
    A DeBERTa-based model for text emotion recognition.
    """
    
    def __init__(self, model_name="microsoft/deberta-v3-small", num_labels=7, dropout_rate=0.1):
        """
        Initialize the text emotion model.
        
        Args:
            model_name (str): The name of the pre-trained DeBERTa model
            num_labels (int): Number of emotion classes to predict
            dropout_rate (float): Dropout rate for regularization
        """
        super(TextEmotionModel, self).__init__()
        
        # Load pre-trained DeBERTa model
        self.deberta = DebertaV2Model.from_pretrained(model_name)
        self.config = self.deberta.config
        
        # Emotion classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights(self.classifier)
        
    def _init_weights(self, module):
        """
        Initialize the weights of the classifier.
        
        Args:
            module: The module to initialize
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Token IDs
            attention_mask (torch.Tensor, optional): Attention mask
            token_type_ids (torch.Tensor, optional): Token type IDs
            
        Returns:
            torch.Tensor: Logits for each emotion class
        """
        # Get DeBERTa outputs
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get the [CLS] token representation
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]  # [CLS] token
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

class TextEmotionTrainer:
    """
    Trainer class for the text emotion model.
    """
    
    def __init__(self, model, tokenizer, device, learning_rate=2e-5):
        """
        Initialize the trainer.
        
        Args:
            model (TextEmotionModel): The model to train
            tokenizer: The tokenizer for processing text
            device (torch.device): The device to use for training
            learning_rate (float): Learning rate for optimization
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        
        # Set up optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, dataloader):
        """
        Train the model for one epoch.
        
        Args:
            dataloader: DataLoader containing training data
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch.get('token_type_ids', None)
            )
            
            # Compute loss
            loss = self.criterion(outputs, batch['labels'])
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """
        Evaluate the model.
        
        Args:
            dataloader: DataLoader containing evaluation data
            
        Returns:
            tuple: (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch.get('token_type_ids', None)
                )
                
                # Compute loss
                loss = self.criterion(outputs, batch['labels'])
                total_loss += loss.item()
                
                # Compute accuracy
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        
        return total_loss / len(dataloader), correct / total
    
    def save_model(self, path):
        """
        Save the model.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
    def load_model(self, path):
        """
        Load the model.
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
