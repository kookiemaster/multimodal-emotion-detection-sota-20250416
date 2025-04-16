"""
Training Pipeline for MemoCMT Implementation

This module implements the training pipeline for the MemoCMT model
as described in the paper "MemoCMT: multimodal emotion recognition using 
cross-modal transformer-based feature fusion" (2025).

The training pipeline uses the hyperparameters specified in the original paper.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoCMTTrainer:
    """
    Trainer class for MemoCMT model.
    
    This class handles training, validation, and testing of the MemoCMT model
    using the hyperparameters specified in the original paper.
    """
    
    def __init__(self, text_model, voice_model, fusion_model, device='cuda',
                 learning_rate=1e-4, weight_decay=1e-5, fusion_weight=0.5,
                 modality_weights=(0.25, 0.25), patience=5, save_dir='checkpoints'):
        """
        Initialize the trainer.
        
        Args:
            text_model: Text emotion model
            voice_model: Voice emotion model
            fusion_model: Fusion model
            device (str): Device to train on
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay for regularization
            fusion_weight (float): Weight for fusion loss
            modality_weights (tuple): Weights for text and voice losses
            patience (int): Patience for early stopping
            save_dir (str): Directory to save checkpoints
        """
        self.text_model = text_model
        self.voice_model = voice_model
        self.fusion_model = fusion_model
        self.device = device
        
        # Move models to device
        self.text_model.to(device)
        self.voice_model.to(device)
        self.fusion_model.to(device)
        
        # Set up optimizer (as per MemoCMT paper)
        self.optimizer = optim.Adam(
            list(self.text_model.parameters()) +
            list(self.voice_model.parameters()) +
            list(self.fusion_model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Set up scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=patience // 2,
            verbose=True
        )
        
        # Set up loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Set up weights for different losses
        self.text_weight = modality_weights[0]
        self.voice_weight = modality_weights[1]
        self.fusion_weight = fusion_weight
        
        # Set up early stopping
        self.patience = patience
        self.best_val_acc = 0
        self.counter = 0
        
        # Set up checkpoint directory
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set up metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        logger.info("Initialized MemoCMT trainer")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Fusion weight: {fusion_weight}")
        logger.info(f"Modality weights: {modality_weights}")
        logger.info(f"Patience: {patience}")
        logger.info(f"Save directory: {save_dir}")
    
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch (int): Current epoch number
            
        Returns:
            tuple: Average loss and accuracy for the epoch
        """
        # Set models to training mode
        self.text_model.train()
        self.voice_model.train()
        self.fusion_model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        # Create progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch in pbar:
            # Skip None batches (from collate_fn)
            if batch is None:
                continue
            
            # Move batch to device
            text_inputs = {k: v.to(self.device) for k, v in batch['text_inputs'].items()}
            audio_features = batch['audio_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass through text model
            _, text_features = self.text_model(
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask'],
                token_type_ids=text_inputs.get('token_type_ids', None),
                return_features=True
            )
            
            # Forward pass through voice model
            _, voice_features = self.voice_model(audio_features, return_features=True)
            
            # Forward pass through fusion model
            text_logits, voice_logits, fusion_logits = self.fusion_model(text_features, voice_features)
            
            # Calculate losses
            text_loss = self.criterion(text_logits, labels)
            voice_loss = self.criterion(voice_logits, labels)
            fusion_loss = self.criterion(fusion_logits, labels)
            
            # Combine losses with weights
            loss = (
                self.text_weight * text_loss +
                self.voice_weight * voice_loss +
                self.fusion_weight * fusion_loss
            )
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(fusion_logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total
            })
        
        # Calculate average metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        # Log metrics
        logger.info(f"Epoch {epoch+1} [Train] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, epoch):
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            epoch (int): Current epoch number
            
        Returns:
            tuple: Average loss and accuracy for validation
        """
        # Set models to evaluation mode
        self.text_model.eval()
        self.voice_model.eval()
        self.fusion_model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        # Create progress bar
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                # Skip None batches (from collate_fn)
                if batch is None:
                    continue
                
                # Move batch to device
                text_inputs = {k: v.to(self.device) for k, v in batch['text_inputs'].items()}
                audio_features = batch['audio_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass through text model
                _, text_features = self.text_model(
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs['attention_mask'],
                    token_type_ids=text_inputs.get('token_type_ids', None),
                    return_features=True
                )
                
                # Forward pass through voice model
                _, voice_features = self.voice_model(audio_features, return_features=True)
                
                # Forward pass through fusion model
                text_logits, voice_logits, fusion_logits = self.fusion_model(text_features, voice_features)
                
                # Calculate losses
                text_loss = self.criterion(text_logits, labels)
                voice_loss = self.criterion(voice_logits, labels)
                fusion_loss = self.criterion(fusion_logits, labels)
                
                # Combine losses with weights
                loss = (
                    self.text_weight * text_loss +
                    self.voice_weight * voice_loss +
                    self.fusion_weight * fusion_loss
                )
                
                # Update metrics
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(fusion_logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100 * correct / total
                })
        
        # Calculate average metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # Log metrics
        logger.info(f"Epoch {epoch+1} [Val] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=30):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs (int): Number of epochs to train for
            
        Returns:
            dict: Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # Training loop
        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, epoch)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.counter = 0
                
                # Save best model
                self.save_checkpoint(epoch, val_acc, is_best=True)
                logger.info(f"Saved best model with validation accuracy: {val_acc:.2f}%")
            else:
                self.counter += 1
                
                # Save regular checkpoint
                self.save_checkpoint(epoch, val_acc)
                
                # Check for early stopping
                if self.counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save training history
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_acc': self.train_accs,
            'val_acc': self.val_accs,
            'best_val_acc': self.best_val_acc
        }
        
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f)
        
        logger.info(f"Training completed. Best validation accuracy: {self.best_val_acc:.2f}%")
        logger.info(f"Training history saved to {history_path}")
        
        return history
    
    def test(self, test_loader):
        """
        Test the model.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            tuple: Test loss and accuracy
        """
        # Load best model
        best_model_path = os.path.join(self.save_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            self.load_checkpoint(best_model_path)
            logger.info(f"Loaded best model from {best_model_path}")
        else:
            logger.warning("Best model checkpoint not found. Using current model state.")
        
        # Set models to evaluation mode
        self.text_model.eval()
        self.voice_model.eval()
        self.fusion_model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        # Create progress bar
        pbar = tqdm(test_loader, desc="Testing")
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in pbar:
                # Skip None batches (from collate_fn)
                if batch is None:
                    continue
                
                # Move batch to device
                text_inputs = {k: v.to(self.device) for k, v in batch['text_inputs'].items()}
                audio_features = batch['audio_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass through text model
                _, text_features = self.text_model(
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs['attention_mask'],
                    token_type_ids=text_inputs.get('token_type_ids', None),
                    return_features=True
                )
                
                # Forward pass through voice model
                _, voice_features = self.voice_model(audio_features, return_features=True)
                
                # Forward pass through fusion model
                text_logits, voice_logits, fusion_logits = self.fusion_model(text_features, voice_features)
                
                # Calculate losses
                text_loss = self.criterion(text_logits, labels)
                voice_loss = self.criterion(voice_logits, labels)
                fusion_loss = self.criterion(fusion_logits, labels)
                
                # Combine losses with weights
                loss = (
                    self.text_weight * text_loss +
                    self.voice_weight * voice_loss +
                    self.fusion_weight * fusion_loss
                )
                
                # Update metrics
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(fusion_logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Save predictions and labels for later analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100 * correct / total
                })
        
        # Calculate average metrics
        avg_loss = total_loss / len(test_loader)
        accuracy = 100 * correct / total
        
        # Log metrics
        logger.info(f"Test - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save predictions and labels
        predictions_path = os.path.join(self.save_dir, 'test_predictions.npz')
        np.savez(
            predictions_path,
            predictions=np.array(all_predictions),
            labels=np.array(all_labels)
        )
        
        logger.info(f"Test predictions saved to {predictions_path}")
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch number
            val_acc (float): Validation accuracy
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'val_acc': val_acc,
            'text_model': self.text_model.state_dict(),
            'voice_model': self.voice_model.state_dict(),
            'fusion_model': self.fusion_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if needed
        if is_best:
            best_model_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_model_path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.text_model.load_state_dict(checkpoint['text_model'])
        self.voice_model.load_state_dict(checkpoint['voice_model'])
        self.fusion_model.load_state_dict(checkpoint['fusion_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_val_acc = checkpoint['best_val_acc']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Epoch: {checkpoint['epoch']+1}, Validation Accuracy: {checkpoint['val_acc']:.2f}%")


def train_memocmt_iemocap(text_model, voice_model, fusion_model, train_loader, test_loader,
                         learning_rate=1e-4, weight_decay=1e-5, fusion_weight=0.5,
                         modality_weights=(0.25, 0.25), num_epochs=30, patience=5,
                         save_dir='checkpoints/memocmt_iemocap', device='cuda'):
    """
    Train MemoCMT model on IEMOCAP dataset.
    
    Args:
        text_model: Text emotion model
        voice_model: Voice emotion model
        fusion_model: Fusion model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay for regularization
        fusion_weight (float): Weight for fusion loss
        modality_weights (tuple): Weights for text and voice losses
        num_epochs (int): Number of epochs to train for
        patience (int): Patience for early stopping
        save_dir (str): Directory to save checkpoints
        device (str): Device to train on
        
    Returns:
        tuple: Training history and test accuracy
    """
    # Create timestamp for save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{save_dir}_{timestamp}"
    
    # Initialize trainer
    trainer = MemoCMTTrainer(
        text_model=text_model,
        voice_model=voice_model,
        fusion_model=fusion_model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fusion_weight=fusion_weight,
        modality_weights=modality_weights,
        patience=patience,
        save_dir=save_dir
    )
    
    # Train model
    history = trainer.train(train_loader, test_loader, num_epochs=num_epochs)
    
    # Test model
    test_loss, test_acc = trainer.test(test_loader)
    
    # Save test results
    results = {
        'test_loss': test_loss,
        'test_acc': test_acc
    }
    
    results_path = os.path.join(save_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f)
    
    logger.info(f"Test results saved to {results_path}")
    
    return history, test_acc


def train_memocmt_esd(text_model, voice_model, fusion_model, train_loader, val_loader, test_loader,
                     learning_rate=1e-4, weight_decay=1e-5, fusion_weight=0.5,
                     modality_weights=(0.25, 0.25), num_epochs=30, patience=5,
                     save_dir='checkpoints/memocmt_esd', device='cuda'):
    """
    Train MemoCMT model on ESD dataset.
    
    Args:
        text_model: Text emotion model
        voice_model: Voice emotion model
        fusion_model: Fusion model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay for regularization
        fusion_weight (float): Weight for fusion loss
        modality_weights (tuple): Weights for text and voice losses
        num_epochs (int): Number of epochs to train for
        patience (int): Patience for early stopping
        save_dir (str): Directory to save checkpoints
        device (str): Device to train on
        
    Returns:
        tuple: Training history and test accuracy
    """
    # Create timestamp for save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{save_dir}_{timestamp}"
    
    # Initialize trainer
    trainer = MemoCMTTrainer(
        text_model=text_model,
        voice_model=voice_model,
        fusion_model=fusion_model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fusion_weight=fusion_weight,
        modality_weights=modality_weights,
        patience=patience,
        save_dir=save_dir
    )
    
    # Train model
    history = trainer.train(train_loader, val_loader, num_epochs=num_epochs)
    
    # Test model
    test_loss, test_acc = trainer.test(test_loader)
    
    # Save test results
    results = {
        'test_loss': test_loss,
        'test_acc': test_acc
    }
    
    results_path = os.path.join(save_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f)
    
    logger.info(f"Test results saved to {results_path}")
    
    return history, test_acc
