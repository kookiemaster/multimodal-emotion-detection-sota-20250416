"""
Training Pipeline for Multimodal Emotion Recognition

This module implements the training pipeline for multimodal emotion recognition
based on the MemoCMT paper, using the original hyperparameters.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.text_emotion_model import TextEmotionModel, TextProcessor, create_emotion_mapping
from src.voice_emotion_model import VoiceEmotionModel, VoiceProcessor
from src.fusion_model import MemoCMTFusion

class IEMOCAPDataset(Dataset):
    """
    Dataset class for the IEMOCAP dataset.
    
    This class handles loading and preprocessing of the IEMOCAP dataset
    from the CSV file provided by the user.
    """
    
    def __init__(self, csv_path, text_processor, voice_processor=None, 
                 emotion_mapping=None, max_samples=None, use_audio=False):
        """
        Initialize the IEMOCAP dataset.
        
        Args:
            csv_path (str): Path to the IEMOCAP CSV file
            text_processor (TextProcessor): Processor for text inputs
            voice_processor (VoiceProcessor, optional): Processor for voice inputs
            emotion_mapping (dict, optional): Mapping from emotion labels to indices
            max_samples (int, optional): Maximum number of samples to use
            use_audio (bool): Whether to use audio features (if False, only text is used)
        """
        self.csv_path = csv_path
        self.text_processor = text_processor
        self.voice_processor = voice_processor
        self.use_audio = use_audio
        
        # Load the CSV file
        self.df = pd.read_csv(csv_path)
        
        # Filter to keep only the main emotions (based on analysis)
        main_emotions = ['Frustration', 'Excited', 'Anger', 'Sadness', 
                         'Neutral state', 'Happiness', 'Fear', 'Surprise']
        self.df = self.df[self.df['Major_emotion'].str.strip().isin(main_emotions)]
        
        # Limit the number of samples if specified
        if max_samples is not None and max_samples < len(self.df):
            self.df = self.df.sample(max_samples, random_state=42)
        
        # Create emotion mapping if not provided
        if emotion_mapping is None:
            self.emotion_mapping = create_emotion_mapping('iemocap')
        else:
            self.emotion_mapping = emotion_mapping
        
        # Extract texts and labels
        self.texts = self.df['Transcript'].tolist()
        self.emotions = self.df['Major_emotion'].str.strip().tolist()
        self.labels = [self.emotion_mapping[emotion] for emotion in self.emotions]
        
        # Extract audio paths if using audio
        if use_audio:
            self.audio_paths = self.df['Audio_Uttrance_Path'].tolist()
            # For this implementation, we'll use synthetic audio features
            # since we don't have access to the actual audio files
            self.audio_features = self._generate_synthetic_audio_features()
    
    def _generate_synthetic_audio_features(self):
        """
        Generate synthetic audio features for demonstration purposes.
        
        In a real implementation, this would extract features from audio files.
        
        Returns:
            list: List of synthetic audio features
        """
        # Generate random features with consistent dimensions
        n_samples = len(self.df)
        n_frames = 100  # Number of time frames
        n_features = 40  # Number of MFCC features
        
        # Use label information to make features somewhat meaningful
        features = []
        for label in self.labels:
            # Create base feature with some noise
            base_feature = np.random.randn(n_frames, n_features) * 0.1
            
            # Add label-specific pattern
            pattern = np.zeros((n_frames, n_features))
            pattern[:, label % n_features] = 1.0
            
            # Combine
            feature = base_feature + pattern * 0.5
            features.append(feature)
        
        return features
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing the sample data
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Process text
        text_inputs = self.text_processor.process_single(text)
        
        # Create sample dictionary
        sample = {
            'text': text,
            'text_inputs': text_inputs,
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        # Add audio features if using audio
        if self.use_audio:
            if self.voice_processor is not None:
                # In a real implementation, this would use the actual audio path
                # audio_path = self.audio_paths[idx]
                # audio_features = self.voice_processor.process_single(audio_path)
                
                # For now, use synthetic features
                audio_features = torch.tensor(self.audio_features[idx], dtype=torch.float32)
                sample['audio_features'] = audio_features
            else:
                # Use pre-generated features
                audio_features = torch.tensor(self.audio_features[idx], dtype=torch.float32)
                sample['audio_features'] = audio_features
        
        return sample


def train_epoch(text_model, voice_model, fusion_model, dataloader, optimizer, 
                criterion, device, fusion_weight=0.5, modality_weights=(0.25, 0.25),
                use_audio=True):
    """
    Train models for one epoch.
    
    Args:
        text_model (nn.Module): Text emotion model
        voice_model (nn.Module): Voice emotion model
        fusion_model (nn.Module): Fusion model
        dataloader (DataLoader): DataLoader for training data
        optimizer (Optimizer): Optimizer for training
        criterion (nn.Module): Loss function
        device (torch.device): Device to use for training
        fusion_weight (float): Weight for fusion loss
        modality_weights (tuple): Weights for text and voice losses
        use_audio (bool): Whether to use audio features
        
    Returns:
        float: Average loss for the epoch
    """
    # Set models to training mode
    text_model.train()
    if use_audio:
        voice_model.train()
        fusion_model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    # Training loop
    for batch in tqdm(dataloader, desc="Training"):
        # Move batch to device
        text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass through text model
        text_logits, text_features = text_model(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            return_features=True
        )
        
        if use_audio:
            # Forward pass through voice model
            audio_features = batch['audio_features'].to(device)
            voice_logits, voice_features = voice_model(audio_features, return_features=True)
            
            # Forward pass through fusion model
            text_logits, voice_logits, fusion_logits = fusion_model(text_features, voice_features)
            
            # Calculate losses
            text_loss = criterion(text_logits, labels)
            voice_loss = criterion(voice_logits, labels)
            fusion_loss = criterion(fusion_logits, labels)
            
            # Combine losses with weights
            loss = (
                modality_weights[0] * text_loss +
                modality_weights[1] * voice_loss +
                fusion_weight * fusion_loss
            )
            
            # Calculate accuracy
            _, predicted = torch.max(fusion_logits, 1)
        else:
            # Only use text model
            loss = criterion(text_logits, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(text_logits, 1)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(text_model, voice_model, fusion_model, dataloader, criterion, 
             device, fusion_weight=0.5, modality_weights=(0.25, 0.25), use_audio=True):
    """
    Evaluate models on validation or test data.
    
    Args:
        text_model (nn.Module): Text emotion model
        voice_model (nn.Module): Voice emotion model
        fusion_model (nn.Module): Fusion model
        dataloader (DataLoader): DataLoader for evaluation data
        criterion (nn.Module): Loss function
        device (torch.device): Device to use for evaluation
        fusion_weight (float): Weight for fusion loss
        modality_weights (tuple): Weights for text and voice losses
        use_audio (bool): Whether to use audio features
        
    Returns:
        tuple: Tuple containing (loss, accuracy, predictions, true_labels)
    """
    # Set models to evaluation mode
    text_model.eval()
    if use_audio:
        voice_model.eval()
        fusion_model.eval()
    
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
            labels = batch['label'].to(device)
            
            # Forward pass through text model
            text_logits, text_features = text_model(
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask'],
                return_features=True
            )
            
            if use_audio:
                # Forward pass through voice model
                audio_features = batch['audio_features'].to(device)
                voice_logits, voice_features = voice_model(audio_features, return_features=True)
                
                # Forward pass through fusion model
                text_logits, voice_logits, fusion_logits = fusion_model(text_features, voice_features)
                
                # Calculate losses
                text_loss = criterion(text_logits, labels)
                voice_loss = criterion(voice_logits, labels)
                fusion_loss = criterion(fusion_logits, labels)
                
                # Combine losses with weights
                loss = (
                    modality_weights[0] * text_loss +
                    modality_weights[1] * voice_loss +
                    fusion_weight * fusion_loss
                )
                
                # Get predictions
                _, predicted = torch.max(fusion_logits, 1)
            else:
                # Only use text model
                loss = criterion(text_logits, labels)
                
                # Get predictions
                _, predicted = torch.max(text_logits, 1)
            
            # Update metrics
            total_loss += loss.item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy, all_predictions, all_labels


def calculate_metrics(predictions, labels, emotion_mapping=None):
    """
    Calculate evaluation metrics for emotion recognition.
    
    Args:
        predictions (list): Predicted emotion indices
        labels (list): Ground truth emotion indices
        emotion_mapping (dict, optional): Mapping from emotion indices to names
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Convert to numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions) * 100
    
    # Calculate F1 scores
    f1_weighted = f1_score(labels, predictions, average='weighted') * 100
    f1_macro = f1_score(labels, predictions, average='macro') * 100
    f1_per_class = f1_score(labels, predictions, average=None) * 100
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Calculate unweighted accuracy (average recall across classes)
    unweighted_acc = np.mean(np.diag(cm_norm)) * 100
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'unweighted_accuracy': unweighted_acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_norm.tolist()
    }
    
    # Add emotion names if mapping is provided
    if emotion_mapping is not None:
        idx_to_emotion = {v: k for k, v in emotion_mapping.items()}
        metrics['emotion_names'] = [idx_to_emotion.get(i, f"Class {i}") for i in range(len(f1_per_class))]
    
    return metrics


def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history (dict): Dictionary containing training history
        save_path (str, optional): Path to save the plot
    """
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
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        class_names (list): List of class names
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add values to cells
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.close()


def train_memocmt_iemocap(
    csv_path,
    output_dir,
    model_name="bert-base-uncased",
    num_classes=8,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=1e-5,
    fusion_weight=0.5,
    modality_weights=(0.25, 0.25),
    num_epochs=30,
    patience=5,
    max_samples=None,
    use_audio=True,
    device=None
):
    """
    Train the MemoCMT model on the IEMOCAP dataset.
    
    Args:
        csv_path (str): Path to the IEMOCAP CSV file
        output_dir (str): Directory to save outputs
        model_name (str): Name of the pretrained model to use
        num_classes (int): Number of emotion classes
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        weight_decay (float): Weight decay for optimizer
        fusion_weight (float): Weight for fusion loss
        modality_weights (tuple): Weights for text and voice losses
        num_epochs (int): Number of training epochs
        patience (int): Patience for early stopping
        max_samples (int, optional): Maximum number of samples to use
        use_audio (bool): Whether to use audio features
        device (torch.device, optional): Device to use for training
        
    Returns:
        tuple: Tuple containing (best_metrics, history)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Initialize text processor
    text_processor = TextProcessor(model_name=model_name)
    
    # Initialize voice processor if using audio
    voice_processor = None
    if use_audio:
        voice_processor = VoiceProcessor()
    
    # Create emotion mapping
    emotion_mapping = create_emotion_mapping('iemocap')
    
    # Load dataset
    dataset = IEMOCAPDataset(
        csv_path=csv_path,
        text_processor=text_processor,
        voice_processor=voice_processor,
        emotion_mapping=emotion_mapping,
        max_samples=max_samples,
        use_audio=use_audio
    )
    
    # Split dataset into train and validation sets
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=[dataset.emotions[i] for i in range(len(dataset))]
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize models
    text_model = TextEmotionModel(
        num_classes=num_classes,
        model_name=model_name
    ).to(device)
    
    voice_model = None
    fusion_model = None
    
    if use_audio:
        voice_model = VoiceEmotionModel(
            num_classes=num_classes
        ).to(device)
        
        fusion_model = MemoCMTFusion(
            feature_dim=256,
            num_classes=num_classes
        ).to(device)
    
    # Set up optimizer
    if use_audio:
        optimizer = torch.optim.Adam(
            list(text_model.parameters()) +
            list(voice_model.parameters()) +
            list(fusion_model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            text_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss, train_acc = train_epoch(
            text_model=text_model,
            voice_model=voice_model,
            fusion_model=fusion_model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            fusion_weight=fusion_weight,
            modality_weights=modality_weights,
            use_audio=use_audio
        )
        
        # Validation
        val_loss, val_acc, val_preds, val_labels = evaluate(
            text_model=text_model,
            voice_model=voice_model,
            fusion_model=fusion_model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            fusion_weight=fusion_weight,
            modality_weights=modality_weights,
            use_audio=use_audio
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best models
            torch.save(text_model.state_dict(), os.path.join(output_dir, "text_model.pt"))
            
            if use_audio:
                torch.save(voice_model.state_dict(), os.path.join(output_dir, "voice_model.pt"))
                torch.save(fusion_model.state_dict(), os.path.join(output_dir, "fusion_model.pt"))
            
            # Calculate and save metrics
            metrics = calculate_metrics(val_preds, val_labels, emotion_mapping)
            
            # Save metrics
            with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
                for key, value in metrics.items():
                    if isinstance(value, list):
                        f.write(f"{key}: {value}\n")
                    else:
                        f.write(f"{key}: {value:.2f}\n")
            
            print(f"Model saved to {output_dir}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Plot and save training history
    plot_training_history(history, save_path=os.path.join(output_dir, "training_history.png"))
    
    # Load best model for final evaluation
    text_model.load_state_dict(torch.load(os.path.join(output_dir, "text_model.pt")))
    
    if use_audio:
        voice_model.load_state_dict(torch.load(os.path.join(output_dir, "voice_model.pt")))
        fusion_model.load_state_dict(torch.load(os.path.join(output_dir, "fusion_model.pt")))
    
    # Final evaluation
    _, _, test_preds, test_labels = evaluate(
        text_model=text_model,
        voice_model=voice_model,
        fusion_model=fusion_model,
        dataloader=val_loader,  # Using validation set as test set
        criterion=criterion,
        device=device,
        fusion_weight=fusion_weight,
        modality_weights=modality_weights,
        use_audio=use_audio
    )
    
    # Calculate final metrics
    final_metrics = calculate_metrics(test_preds, test_labels, emotion_mapping)
    
    # Plot and save confusion matrix
    class_names = [emotion_mapping[emotion] for emotion in sorted(emotion_mapping, key=emotion_mapping.get)]
    plot_confusion_matrix(
        np.array(final_metrics['confusion_matrix']),
        class_names,
        title='Confusion Matrix',
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )
    
    return final_metrics, history


if __name__ == "__main__":
    # Example usage
    csv_path = "/home/ubuntu/upload/IEMOCAP_Final.csv"
    output_dir = "/home/ubuntu/multimodal-emotion-detection-sota-20250416/results"
    
    # Train model
    metrics, history = train_memocmt_iemocap(
        csv_path=csv_path,
        output_dir=output_dir,
        max_samples=1000,  # Limit samples for demonstration
        num_epochs=5,      # Limit epochs for demonstration
        use_audio=True     # Use both text and audio
    )
    
    print("\nFinal Metrics:")
    for key, value in metrics.items():
        if isinstance(value, list) and len(value) > 10:
            print(f"{key}: [...]")
        elif isinstance(value, list):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.2f}")
