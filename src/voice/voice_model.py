"""
Voice Emotion Model

This module implements the Semi-CNN model for voice emotion detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SemiCNN(nn.Module):
    """
    Semi-CNN architecture for speech emotion recognition.
    
    This model uses a combination of convolutional layers for feature extraction
    and LSTM layers for temporal modeling.
    """
    
    def __init__(self, n_mels=128, num_classes=7):
        """
        Initialize the Semi-CNN model.
        
        Args:
            n_mels (int): Number of mel bands in the input
            num_classes (int): Number of emotion classes to predict
        """
        super(SemiCNN, self).__init__()
        
        # Convolutional feature extraction blocks
        # Block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(0.3)
        
        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout(0.3)
        
        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout3 = nn.Dropout(0.3)
        
        # Calculate the size of features after convolution and pooling
        # Assuming input is (batch_size, 1, n_mels, time_frames)
        # After 3 pooling layers with kernel size 2, the dimensions are reduced by 2^3 = 8
        self.n_mels_reduced = n_mels // 8
        
        # LSTM layers for temporal modeling
        self.lstm1 = nn.LSTM(256 * self.n_mels_reduced, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(128, 64)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_frames, n_mels)
                              or (batch_size, 1, n_mels, time_frames)
            
        Returns:
            torch.Tensor: Logits for each emotion class
        """
        batch_size = x.size(0)
        
        # Ensure input is in the right shape (batch_size, channels, height, width)
        if x.dim() == 3:  # (batch_size, time_frames, n_mels)
            x = x.unsqueeze(1)  # Add channel dimension
            x = x.transpose(2, 3)  # Swap time_frames and n_mels
        
        # Convolutional feature extraction
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Reshape for LSTM: (batch_size, time_frames, channels * n_mels_reduced)
        time_frames = x.size(3)
        x = x.permute(0, 3, 1, 2)  # (batch_size, time_frames, channels, n_mels_reduced)
        x = x.reshape(batch_size, time_frames, -1)  # (batch_size, time_frames, channels * n_mels_reduced)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Global average pooling over time dimension
        x = torch.mean(x, dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x

class VoiceEmotionTrainer:
    """
    Trainer class for the voice emotion model.
    """
    
    def __init__(self, model, device, learning_rate=0.001):
        """
        Initialize the trainer.
        
        Args:
            model (SemiCNN): The model to train
            device (torch.device): The device to use for training
            learning_rate (float): Learning rate for optimization
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        
        # Set up optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
            inputs = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Compute loss
            loss = self.criterion(outputs, labels)
            
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
                inputs = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Compute accuracy
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
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
