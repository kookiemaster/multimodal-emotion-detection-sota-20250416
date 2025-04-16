"""
Training Pipeline

This module implements the training pipeline for multimodal emotion detection.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

class MultimodalTrainer:
    """
    Trainer class for multimodal emotion detection.
    """
    
    def __init__(self, model, device, learning_rate=0.001, weight_decay=1e-5):
        """
        Initialize the trainer.
        
        Args:
            model: The multimodal fusion model
            device (torch.device): The device to use for training
            learning_rate (float): Learning rate for optimization
            weight_decay (float): Weight decay for regularization
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Set up optimizer and loss function
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader):
        """
        Train the model for one epoch.
        
        Args:
            train_loader (DataLoader): DataLoader for training data
            
        Returns:
            tuple: (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            # Move batch to device
            text_features = self._prepare_text_features(batch['text_features'])
            audio_features = batch['audio_features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(text_features, audio_features)
            
            # Compute loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader):
        """
        Evaluate the model.
        
        Args:
            val_loader (DataLoader): DataLoader for validation data
            
        Returns:
            tuple: (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                text_features = self._prepare_text_features(batch['text_features'])
                audio_features = batch['audio_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(text_features, audio_features)
                
                # Compute loss
                loss = self.criterion(outputs, labels)
                
                # Update statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions and labels for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def _prepare_text_features(self, text_features):
        """
        Prepare text features for the model.
        
        Args:
            text_features: Text features from the dataset
            
        Returns:
            torch.Tensor: Prepared text features
        """
        # If text_features is a dictionary (tokenized input)
        if isinstance(text_features, dict):
            return {k: v.to(self.device) for k, v in text_features.items()}
        
        # If text_features is already a tensor
        return text_features.to(self.device)
    
    def train(self, train_loader, val_loader, epochs=10, patience=5, model_save_path=None):
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): DataLoader for training data
            val_loader (DataLoader): DataLoader for validation data
            epochs (int): Number of epochs to train
            patience (int): Early stopping patience
            model_save_path (str, optional): Path to save the best model
            
        Returns:
            dict: Training history
        """
        print(f"Starting training for {epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Train one epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Evaluate
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - "
                  f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                if model_save_path is not None:
                    self.save_model(model_save_path)
                    print(f"Model saved to {model_save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        
        return self.history
    
    def save_model(self, path):
        """
        Save the model.
        
        Args:
            path (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
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
        self.history = checkpoint.get('history', self.history)
    
    def plot_history(self, save_path=None):
        """
        Plot training history.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss')
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def evaluate_metrics(self, test_loader, class_names=None):
        """
        Evaluate the model and compute detailed metrics.
        
        Args:
            test_loader (DataLoader): DataLoader for test data
            class_names (list, optional): List of class names
            
        Returns:
            dict: Evaluation metrics
        """
        # Evaluate model
        test_loss, test_acc, all_preds, all_labels = self.evaluate(test_loader)
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Compute classification report
        if class_names is None:
            class_names = [str(i) for i in range(len(np.unique(all_labels)))]
        
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        
        # Print results
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Return metrics
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        return metrics


# Example usage
if __name__ == "__main__":
    # This is a demonstration of how the training pipeline would be used
    # In a real scenario, you would use actual models and data loaders
    
    from fusion.fusion import MultimodalFusionModel
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = MultimodalFusionModel(
        text_feature_dim=768,  # DeBERTa embedding size
        voice_feature_dim=256,  # Semi-CNN feature size
        hidden_dim=128,
        num_classes=7
    ).to(device)
    
    # Initialize trainer
    trainer = MultimodalTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        weight_decay=1e-5
    )
    
    # In a real scenario, you would load data and train the model
    # For demonstration, we'll just print the trainer's methods
    print("Training pipeline initialized.")
    print("Available methods:")
    print("- train_epoch: Train for one epoch")
    print("- evaluate: Evaluate the model")
    print("- train: Train the model for multiple epochs with early stopping")
    print("- save_model: Save the model")
    print("- load_model: Load a saved model")
    print("- plot_history: Plot training history")
    print("- evaluate_metrics: Compute detailed evaluation metrics")
