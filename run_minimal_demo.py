"""
Minimal demo of multimodal emotion detection with synthetic data.

This script demonstrates the core functionality of the implemented models
using synthetic data to avoid memory and resource constraints.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Set output directory
OUTPUT_DIR = "/home/ubuntu/multimodal-emotion-detection-sota-20250416/results/demo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create emotion mapping based on IEMOCAP dataset
emotion_mapping = {
    'Frustration': 0,
    'Excited': 1,
    'Anger': 2,
    'Sadness': 3,
    'Neutral state': 4,
    'Happiness': 5,
    'Fear': 6,
    'Surprise': 7
}
idx_to_emotion = {v: k for k, v in emotion_mapping.items()}
num_classes = len(emotion_mapping)

print(f"Emotion mapping: {emotion_mapping}")
print(f"Number of classes: {num_classes}")

# Generate synthetic data
def generate_synthetic_data(num_samples=100, feature_dim=768):
    """Generate synthetic data for demonstration."""
    # Generate random labels
    labels = np.random.randint(0, num_classes, size=num_samples)
    
    # Generate text features with some class separation
    text_features = np.random.randn(num_samples, feature_dim) * 0.1
    for i, label in enumerate(labels):
        # Add class-specific pattern
        text_features[i, label * (feature_dim // num_classes):(label + 1) * (feature_dim // num_classes)] += 1.0
    
    # Generate voice features with some class separation
    voice_features = np.random.randn(num_samples, feature_dim) * 0.1
    for i, label in enumerate(labels):
        # Add class-specific pattern
        voice_features[i, label * (feature_dim // num_classes):(label + 1) * (feature_dim // num_classes)] += 0.8
    
    return torch.tensor(text_features, dtype=torch.float32), torch.tensor(voice_features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# Generate data
print("Generating synthetic data...")
text_features, voice_features, labels = generate_synthetic_data(num_samples=200, feature_dim=256)

# Split into train and test sets
train_size = 150
train_text = text_features[:train_size]
train_voice = voice_features[:train_size]
train_labels = labels[:train_size]

test_text = text_features[train_size:]
test_voice = voice_features[train_size:]
test_labels = labels[train_size:]

print(f"Training samples: {train_size}")
print(f"Testing samples: {len(test_labels)}")

# Define a simplified text model that doesn't require pretrained weights
class SimpleTextModel(torch.nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_classes=8):
        super(SimpleTextModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, return_features=False):
        if return_features:
            features = x
            logits = self.layers(features)
            return logits, features
        else:
            return self.layers(x)

# Define a simplified voice model
class SimpleVoiceModel(torch.nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_classes=8):
        super(SimpleVoiceModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, return_features=False):
        if return_features:
            features = x
            logits = self.layers(features)
            return logits, features
        else:
            return self.layers(x)

# Define a simplified fusion model based on MemoCMT
class SimpleFusionModel(torch.nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=128, num_classes=8, fusion_method='min'):
        super(SimpleFusionModel, self).__init__()
        self.fusion_method = fusion_method
        
        # Text classifier
        self.text_classifier = torch.nn.Linear(feature_dim, num_classes)
        
        # Voice classifier
        self.voice_classifier = torch.nn.Linear(feature_dim, num_classes)
        
        # Fusion classifier
        if fusion_method == 'concat':
            self.fusion_classifier = torch.nn.Linear(feature_dim * 2, num_classes)
        else:
            self.fusion_classifier = torch.nn.Linear(feature_dim, num_classes)
    
    def forward(self, text_features, voice_features):
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

# Initialize models
print("Initializing models...")
text_model = SimpleTextModel(input_dim=256, num_classes=num_classes)
voice_model = SimpleVoiceModel(input_dim=256, num_classes=num_classes)
fusion_model = SimpleFusionModel(feature_dim=256, num_classes=num_classes, fusion_method='min')

# Set up training
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params': text_model.parameters()},
    {'params': voice_model.parameters()},
    {'params': fusion_model.parameters()}
], lr=0.001)

# Training loop
print("Training models...")
num_epochs = 10
history = {
    'text_loss': [],
    'voice_loss': [],
    'fusion_loss': [],
    'text_acc': [],
    'voice_acc': [],
    'fusion_acc': []
}

for epoch in range(num_epochs):
    # Training
    text_model.train()
    voice_model.train()
    fusion_model.train()
    
    # Forward pass
    text_logits, text_features = text_model(train_text, return_features=True)
    voice_logits, voice_features = voice_model(train_voice, return_features=True)
    text_logits, voice_logits, fusion_logits = fusion_model(text_features, voice_features)
    
    # Calculate losses
    text_loss = criterion(text_logits, train_labels)
    voice_loss = criterion(voice_logits, train_labels)
    fusion_loss = criterion(fusion_logits, train_labels)
    
    # Combined loss
    loss = 0.25 * text_loss + 0.25 * voice_loss + 0.5 * fusion_loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate training accuracy
    _, text_preds = torch.max(text_logits, 1)
    _, voice_preds = torch.max(voice_logits, 1)
    _, fusion_preds = torch.max(fusion_logits, 1)
    
    text_acc = (text_preds == train_labels).float().mean() * 100
    voice_acc = (voice_preds == train_labels).float().mean() * 100
    fusion_acc = (fusion_preds == train_labels).float().mean() * 100
    
    # Save history
    history['text_loss'].append(text_loss.item())
    history['voice_loss'].append(voice_loss.item())
    history['fusion_loss'].append(fusion_loss.item())
    history['text_acc'].append(text_acc.item())
    history['voice_acc'].append(voice_acc.item())
    history['fusion_acc'].append(fusion_acc.item())
    
    # Print progress
    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Text Loss: {text_loss.item():.4f}, Acc: {text_acc.item():.2f}%")
        print(f"  Voice Loss: {voice_loss.item():.4f}, Acc: {voice_acc.item():.2f}%")
        print(f"  Fusion Loss: {fusion_loss.item():.4f}, Acc: {fusion_acc.item():.2f}%")

# Evaluation
print("\nEvaluating models...")
text_model.eval()
voice_model.eval()
fusion_model.eval()

with torch.no_grad():
    # Forward pass
    text_logits, text_features = text_model(test_text, return_features=True)
    voice_logits, voice_features = voice_model(test_voice, return_features=True)
    text_logits, voice_logits, fusion_logits = fusion_model(text_features, voice_features)
    
    # Calculate predictions
    _, text_preds = torch.max(text_logits, 1)
    _, voice_preds = torch.max(voice_logits, 1)
    _, fusion_preds = torch.max(fusion_logits, 1)
    
    # Calculate accuracy
    text_acc = (text_preds == test_labels).float().mean() * 100
    voice_acc = (voice_preds == test_labels).float().mean() * 100
    fusion_acc = (fusion_preds == test_labels).float().mean() * 100
    
    print(f"Text Accuracy: {text_acc.item():.2f}%")
    print(f"Voice Accuracy: {voice_acc.item():.2f}%")
    print(f"Fusion Accuracy: {fusion_acc.item():.2f}%")
    
    # Calculate confusion matrix for fusion model
    cm = confusion_matrix(test_labels.numpy(), fusion_preds.numpy())
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot training history
plt.figure(figsize=(15, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history['text_loss'], label='Text Loss')
plt.plot(history['voice_loss'], label='Voice Loss')
plt.plot(history['fusion_loss'], label='Fusion Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history['text_acc'], label='Text Accuracy')
plt.plot(history['voice_acc'], label='Voice Accuracy')
plt.plot(history['fusion_acc'], label='Fusion Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"))
print(f"Training history plot saved to {os.path.join(OUTPUT_DIR, 'training_history.png')}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
class_names = [idx_to_emotion[i] for i in range(num_classes)]
plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
thresh = cm_norm.max() / 2.
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                 ha="center", va="center",
                 color="white" if cm_norm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
print(f"Confusion matrix plot saved to {os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}")

print("\nDemo completed successfully!")
