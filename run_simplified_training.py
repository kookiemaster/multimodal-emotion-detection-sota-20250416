"""
Script to train and evaluate the multimodal emotion detection model on IEMOCAP data.

This script runs the training pipeline with a small subset of the data
to demonstrate the functionality while working within resource constraints.
"""

import os
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.text_emotion_model import TextEmotionModel, TextProcessor, create_emotion_mapping

# Set paths
CSV_PATH = "/home/ubuntu/upload/IEMOCAP_Final.csv"
OUTPUT_DIR = "/home/ubuntu/multimodal-emotion-detection-sota-20250416/results/iemocap_run"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting simplified training on IEMOCAP data")

# Load and prepare data
print("Loading data from CSV...")
df = pd.read_csv(CSV_PATH)

# Filter to keep only the main emotions (based on analysis)
main_emotions = ['Frustration', 'Excited', 'Anger', 'Sadness', 
                'Neutral state', 'Happiness', 'Fear', 'Surprise']
df = df[df['Major_emotion'].str.strip().isin(main_emotions)]

# Limit samples for demonstration
MAX_SAMPLES = 500
if MAX_SAMPLES < len(df):
    df = df.sample(MAX_SAMPLES, random_state=42)

# Create emotion mapping
emotion_mapping = create_emotion_mapping('iemocap')
print(f"Emotion mapping: {emotion_mapping}")

# Extract texts and labels
texts = df['Transcript'].tolist()
emotions = df['Major_emotion'].str.strip().tolist()
labels = [emotion_mapping[emotion] for emotion in emotions]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Initialize text processor
text_processor = TextProcessor(model_name="bert-base-uncased")

# Initialize model
model = TextEmotionModel(
    num_classes=len(emotion_mapping),
    model_name="bert-base-uncased"
)

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 3

# Set up optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
print("Starting training...")
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    # Process data in batches
    for i in range(0, len(X_train), BATCH_SIZE):
        batch_texts = X_train[i:i+BATCH_SIZE]
        batch_labels = y_train[i:i+BATCH_SIZE]
        
        # Process texts
        inputs = text_processor.process(batch_texts)
        labels_tensor = torch.tensor(batch_labels)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        # Calculate loss
        loss = criterion(logits, labels_tensor)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update metrics
        train_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        train_correct += (predicted == labels_tensor).sum().item()
        train_total += len(batch_labels)
        
        # Print progress
        if (i // BATCH_SIZE) % 5 == 0:
            print(f"  Batch {i//BATCH_SIZE+1}/{len(X_train)//BATCH_SIZE+1}, Loss: {loss.item():.4f}")
    
    # Calculate epoch metrics
    train_loss = train_loss / (len(X_train) // BATCH_SIZE + 1)
    train_acc = 100 * train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), BATCH_SIZE):
            batch_texts = X_test[i:i+BATCH_SIZE]
            batch_labels = y_test[i:i+BATCH_SIZE]
            
            # Process texts
            inputs = text_processor.process(batch_texts)
            labels_tensor = torch.tensor(batch_labels)
            
            # Forward pass
            logits = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            # Calculate loss
            loss = criterion(logits, labels_tensor)
            
            # Update metrics
            val_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            val_preds.extend(predicted.numpy())
            val_labels.extend(batch_labels)
    
    # Calculate validation metrics
    val_loss = val_loss / (len(X_test) // BATCH_SIZE + 1)
    val_acc = 100 * accuracy_score(val_labels, val_preds)
    
    # Update history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Print metrics
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Save model
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "text_model.pt"))

# Calculate final metrics
final_metrics = {
    'accuracy': accuracy_score(val_labels, val_preds) * 100,
    'f1_weighted': f1_score(val_labels, val_preds, average='weighted') * 100,
    'f1_macro': f1_score(val_labels, val_preds, average='macro') * 100,
    'f1_per_class': f1_score(val_labels, val_preds, average=None) * 100
}

# Calculate confusion matrix
cm = confusion_matrix(val_labels, val_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
final_metrics['unweighted_accuracy'] = np.mean(np.diag(cm_norm)) * 100

# Save metrics
with open(os.path.join(OUTPUT_DIR, "final_metrics.json"), "w") as f:
    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for k, v in final_metrics.items():
        if hasattr(v, 'tolist'):
            serializable_metrics[k] = v.tolist()
        else:
            serializable_metrics[k] = v
    json.dump(serializable_metrics, f, indent=4)

# Save history
with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
    json.dump(history, f, indent=4)

# Plot training history
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
class_names = [k for k, v in sorted(emotion_mapping.items(), key=lambda item: item[1])]
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

print("\nTraining completed successfully!")
print(f"Results saved to {OUTPUT_DIR}")

# Print final metrics
print("\nFinal Metrics:")
for key, value in final_metrics.items():
    if isinstance(value, list) or (hasattr(value, 'tolist') and len(value) > 1):
        print(f"{key}: [...]")
    else:
        print(f"{key}: {value:.2f}")
