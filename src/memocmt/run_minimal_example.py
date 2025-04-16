"""
Run script for training MemoCMT models on minimal example datasets.

This script provides a lightweight training example that can run with limited resources.
It uses a small subset of data to demonstrate the model's functionality.
"""

import os
import torch
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define evaluation metrics function here to avoid import issues
def calculate_metrics(predictions, labels, emotion_map=None):
    """
    Calculate evaluation metrics for emotion recognition.
    
    Args:
        predictions (numpy.ndarray): Predicted emotion indices
        labels (numpy.ndarray): Ground truth emotion indices
        emotion_map (dict, optional): Mapping from emotion indices to names
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Calculate weighted accuracy (standard accuracy)
    weighted_acc = accuracy_score(labels, predictions)
    
    # Calculate unweighted accuracy (average recall across classes)
    cm = confusion_matrix(labels, predictions)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    unweighted_acc = np.mean(np.diag(cm_norm))
    
    # Calculate F1 scores
    f1_weighted = f1_score(labels, predictions, average='weighted')
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_per_class = f1_score(labels, predictions, average=None)
    
    # Create metrics dictionary
    metrics = {
        'weighted_accuracy': weighted_acc * 100,  # Convert to percentage
        'unweighted_accuracy': unweighted_acc * 100,  # Convert to percentage
        'f1_weighted': f1_weighted * 100,  # Convert to percentage
        'f1_macro': f1_macro * 100,  # Convert to percentage
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_norm.tolist()
    }
    
    # Add per-class F1 scores
    if emotion_map is not None:
        # Create reverse mapping from indices to emotion names
        idx_to_emotion = {v: k for k, v in emotion_map.items()}
        for i, f1 in enumerate(f1_per_class):
            emotion = idx_to_emotion.get(i, f"class_{i}")
            metrics[f'f1_{emotion}'] = f1 * 100  # Convert to percentage
    else:
        for i, f1 in enumerate(f1_per_class):
            metrics[f'f1_class_{i}'] = f1 * 100  # Convert to percentage
    
    return metrics

def plot_confusion_matrix(cm, class_names=None, normalize=True, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        class_names (list, optional): List of class names
        normalize (bool): Whether to normalize the confusion matrix
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The confusion matrix figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return plt.gcf()

class SimpleTextModel(torch.nn.Module):
    """Simple text model for demonstration purposes."""
    
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=4, dropout_rate=0.1):
        """Initialize simple text model."""
        super(SimpleTextModel, self).__init__()
        
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        
    def forward(self, x, return_features=False):
        """Forward pass through the model."""
        features = self.fc1(x)
        features = torch.nn.functional.relu(features)
        features = self.layer_norm(features)
        features = self.dropout(features)
        
        logits = self.fc2(features)
        
        if return_features:
            return logits, features
        else:
            return logits

class SimpleVoiceModel(torch.nn.Module):
    """Simple voice model for demonstration purposes."""
    
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=4, dropout_rate=0.1):
        """Initialize simple voice model."""
        super(SimpleVoiceModel, self).__init__()
        
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        
        # Attention mechanism for temporal aggregation
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1)
        )
        
    def forward(self, x, return_features=False):
        """Forward pass through the model."""
        # Apply attention to aggregate temporal features
        attention_weights = self.attention(x).squeeze(-1)
        attention_weights = torch.nn.functional.softmax(attention_weights, dim=1)
        
        # Apply attention weights to get context vector
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),
            x
        ).squeeze(1)
        
        # Process through fully connected layers
        features = self.fc1(context_vector)
        features = torch.nn.functional.relu(features)
        features = self.layer_norm(features)
        features = self.dropout(features)
        
        logits = self.fc2(features)
        
        if return_features:
            return logits, features
        else:
            return logits

class SimpleFusionModel(torch.nn.Module):
    """Simple fusion model for demonstration purposes."""
    
    def __init__(self, feature_dim=256, num_classes=4, fusion_method='min', dropout_rate=0.1):
        """Initialize simple fusion model."""
        super(SimpleFusionModel, self).__init__()
        
        self.fusion_method = fusion_method
        
        # Classification heads
        self.text_classifier = torch.nn.Linear(feature_dim, num_classes)
        self.voice_classifier = torch.nn.Linear(feature_dim, num_classes)
        
        # Fusion classifier
        if fusion_method == 'concat':
            self.fusion_classifier = torch.nn.Linear(feature_dim * 2, num_classes)
        else:
            self.fusion_classifier = torch.nn.Linear(feature_dim, num_classes)
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout_rate)
        
    def forward(self, text_features, voice_features):
        """Forward pass through the model."""
        # Apply dropout
        text_features = self.dropout(text_features)
        voice_features = self.dropout(voice_features)
        
        # Get modality-specific predictions
        text_logits = self.text_classifier(text_features)
        voice_logits = self.voice_classifier(voice_features)
        
        # Fuse features based on specified method
        if self.fusion_method == 'min':
            # Min aggregation (as per MemoCMT paper)
            fused_features = torch.min(text_features, voice_features)
        elif self.fusion_method == 'max':
            # Max aggregation
            fused_features = torch.max(text_features, voice_features)
        elif self.fusion_method == 'avg':
            # Average aggregation
            fused_features = (text_features + voice_features) / 2
        elif self.fusion_method == 'concat':
            # Concatenation
            fused_features = torch.cat([text_features, voice_features], dim=1)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        
        # Get fusion predictions
        fusion_logits = self.fusion_classifier(fused_features)
        
        return text_logits, voice_logits, fusion_logits

class MinimalDataset:
    """Minimal dataset for demonstration purposes."""
    
    def __init__(self, num_samples=100, num_classes=4, feature_dim=768, seq_length=10):
        """Initialize minimal dataset."""
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.seq_length = seq_length
        
        # Generate random data
        self.text_features = torch.randn(num_samples, feature_dim)
        self.voice_features = torch.randn(num_samples, seq_length, feature_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
        # Create emotion map
        self.emotion_map = {
            'angry': 0,
            'happy': 1,
            'sad': 2,
            'neutral': 3
        }
        
        # Create sample texts
        self.texts = [
            "I am so angry about this situation.",
            "I'm really happy today!",
            "I feel sad and disappointed.",
            "I am neutral about this topic."
        ]
        self.texts = [self.texts[label] for label in self.labels]
    
    def __len__(self):
        """Get dataset length."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        return {
            'text_features': self.text_features[idx],
            'voice_features': self.voice_features[idx],
            'label': self.labels[idx],
            'text': self.texts[idx]
        }
    
    def get_batch(self, batch_size=32):
        """Get a batch of samples."""
        indices = torch.randperm(len(self))[:batch_size]
        
        return {
            'text_features': self.text_features[indices],
            'voice_features': self.voice_features[indices],
            'labels': self.labels[indices],
            'texts': [self.texts[i] for i in indices]
        }

def train_minimal_model(output_dir="results/minimal_example"):
    """Train a minimal MemoCMT model for demonstration."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create minimal dataset
    dataset = MinimalDataset(num_samples=500, num_classes=4)
    logger.info(f"Created minimal dataset with {len(dataset)} samples")
    
    # Initialize models - using simple models instead of the full BERT/HuBERT models
    text_model = SimpleTextModel(input_dim=768, hidden_dim=256, num_classes=4)
    voice_model = SimpleVoiceModel(input_dim=768, hidden_dim=256, num_classes=4)
    fusion_model = SimpleFusionModel(feature_dim=256, num_classes=4, fusion_method='min')
    
    # Move models to device
    text_model.to(device)
    voice_model.to(device)
    fusion_model.to(device)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(
        list(text_model.parameters()) +
        list(voice_model.parameters()) +
        list(fusion_model.parameters()),
        lr=1e-3
    )
    
    # Set up loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 10
    batch_size = 32
    steps_per_epoch = len(dataset) // batch_size
    
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Training
        text_model.train()
        voice_model.train()
        fusion_model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for step in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Get batch
            batch = dataset.get_batch(batch_size)
            
            # Move batch to device
            text_features = batch['text_features'].to(device)
            voice_features = batch['voice_features'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass through text model
            text_logits, text_features = text_model(text_features, return_features=True)
            
            # Forward pass through voice model
            voice_logits, voice_features = voice_model(voice_features, return_features=True)
            
            # Forward pass through fusion model
            text_logits, voice_logits, fusion_logits = fusion_model(text_features, voice_features)
            
            # Calculate losses
            text_loss = criterion(text_logits, labels)
            voice_loss = criterion(voice_logits, labels)
            fusion_loss = criterion(fusion_logits, labels)
            
            # Combine losses
            loss = 0.25 * text_loss + 0.25 * voice_loss + 0.5 * fusion_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(fusion_logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        # Calculate epoch metrics
        avg_loss = total_loss / steps_per_epoch
        accuracy = 100 * correct / total
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    logger.info("Training complete")
    
    # Evaluation
    logger.info("Evaluating model...")
    
    text_model.eval()
    voice_model.eval()
    fusion_model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for _ in tqdm(range(10), desc="Evaluating"):
            # Get batch
            batch = dataset.get_batch(batch_size=50)
            
            # Move batch to device
            text_features = batch['text_features'].to(device)
            voice_features = batch['voice_features'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass through text model
            _, text_features = text_model(text_features, return_features=True)
            
            # Forward pass through voice model
            _, voice_features = voice_model(voice_features, return_features=True)
            
            # Forward pass through fusion model
            _, _, fusion_logits = fusion_model(text_features, voice_features)
            
            # Get predictions
            _, predicted = torch.max(fusion_logits, 1)
            
            # Save predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(
        predictions=np.array(all_predictions),
        labels=np.array(all_labels),
        emotion_map=dataset.emotion_map
    )
    
    # Log metrics
    logger.info(f"Evaluation metrics:")
    logger.info(f"  Weighted Accuracy: {metrics['weighted_accuracy']:.2f}%")
    logger.info(f"  Unweighted Accuracy: {metrics['unweighted_accuracy']:.2f}%")
    logger.info(f"  F1 Score (Weighted): {metrics['f1_weighted']:.2f}%")
    
    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    class_names = list(dataset.emotion_map.keys())
    
    # Save confusion matrix
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names=class_names, normalize=False, save_path=cm_path)
    
    # Save normalized confusion matrix
    cm_norm_path = os.path.join(output_dir, "confusion_matrix_normalized.png")
    plot_confusion_matrix(cm, class_names=class_names, normalize=True, save_path=cm_norm_path)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Results saved to {output_dir}")
    
    return metrics

if __name__ == "__main__":
    train_minimal_model()
