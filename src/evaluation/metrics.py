"""
Evaluation Metrics

This module implements evaluation metrics for multimodal emotion detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

class EmotionEvaluator:
    """
    Class for evaluating emotion detection models.
    """
    
    def __init__(self, class_names=None):
        """
        Initialize the evaluator.
        
        Args:
            class_names (list, optional): List of class names
        """
        self.class_names = class_names or [
            "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
        ]
    
    def compute_basic_metrics(self, y_true, y_pred):
        """
        Compute basic evaluation metrics.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_score(y_true, y_pred, average=None)[i]
            metrics[f'recall_{class_name}'] = recall_score(y_true, y_pred, average=None)[i]
            metrics[f'f1_{class_name}'] = f1_score(y_true, y_pred, average=None)[i]
        
        return metrics
    
    def compute_confusion_matrix(self, y_true, y_pred):
        """
        Compute confusion matrix.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            
        Returns:
            numpy.ndarray: Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False, figsize=(10, 8), save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            normalize (bool): Whether to normalize the confusion matrix
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def compute_classification_report(self, y_true, y_pred, output_dict=True):
        """
        Compute classification report.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            output_dict (bool): Whether to return a dictionary
            
        Returns:
            dict or str: Classification report
        """
        return classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=output_dict
        )
    
    def compute_roc_auc(self, y_true, y_score):
        """
        Compute ROC AUC score for multi-class classification.
        
        Args:
            y_true (array-like): True labels
            y_score (array-like): Predicted probabilities
            
        Returns:
            dict: Dictionary of ROC AUC scores
        """
        # Convert y_true to one-hot encoding
        y_true_onehot = np.zeros((len(y_true), len(self.class_names)))
        for i, label in enumerate(y_true):
            y_true_onehot[i, label] = 1
        
        # Compute ROC AUC for each class
        roc_auc = {}
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
            roc_auc[class_name] = auc(fpr, tpr)
        
        # Compute micro-average ROC curve and ROC AUC
        fpr, tpr, _ = roc_curve(y_true_onehot.ravel(), y_score.ravel())
        roc_auc['micro'] = auc(fpr, tpr)
        
        # Compute macro-average ROC AUC
        roc_auc['macro'] = np.mean(list(roc_auc.values())[:-1])  # Exclude 'micro'
        
        return roc_auc
    
    def plot_roc_curves(self, y_true, y_score, figsize=(10, 8), save_path=None):
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            y_true (array-like): True labels
            y_score (array-like): Predicted probabilities
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the plot
        """
        # Convert y_true to one-hot encoding
        y_true_onehot = np.zeros((len(y_true), len(self.class_names)))
        for i, label in enumerate(y_true):
            y_true_onehot[i, label] = 1
        
        # Plot ROC curves
        plt.figure(figsize=figsize)
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        # Plot micro-average ROC curve
        fpr, tpr, _ = roc_curve(y_true_onehot.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Micro-average (AUC = {roc_auc:.2f})')
        
        # Plot random guess line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def compute_precision_recall(self, y_true, y_score):
        """
        Compute precision-recall curves for multi-class classification.
        
        Args:
            y_true (array-like): True labels
            y_score (array-like): Predicted probabilities
            
        Returns:
            dict: Dictionary of average precision scores
        """
        # Convert y_true to one-hot encoding
        y_true_onehot = np.zeros((len(y_true), len(self.class_names)))
        for i, label in enumerate(y_true):
            y_true_onehot[i, label] = 1
        
        # Compute precision-recall for each class
        average_precision = {}
        for i, class_name in enumerate(self.class_names):
            average_precision[class_name] = average_precision_score(y_true_onehot[:, i], y_score[:, i])
        
        # Compute micro-average precision-recall
        average_precision['micro'] = average_precision_score(y_true_onehot.ravel(), y_score.ravel())
        
        # Compute macro-average precision-recall
        average_precision['macro'] = np.mean(list(average_precision.values())[:-1])  # Exclude 'micro'
        
        return average_precision
    
    def plot_precision_recall_curves(self, y_true, y_score, figsize=(10, 8), save_path=None):
        """
        Plot precision-recall curves for multi-class classification.
        
        Args:
            y_true (array-like): True labels
            y_score (array-like): Predicted probabilities
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the plot
        """
        # Convert y_true to one-hot encoding
        y_true_onehot = np.zeros((len(y_true), len(self.class_names)))
        for i, label in enumerate(y_true):
            y_true_onehot[i, label] = 1
        
        # Plot precision-recall curves
        plt.figure(figsize=figsize)
        
        # Plot precision-recall curve for each class
        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_score[:, i])
            ap = average_precision_score(y_true_onehot[:, i], y_score[:, i])
            plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {ap:.2f})')
        
        # Plot micro-average precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true_onehot.ravel(), y_score.ravel())
        ap = average_precision_score(y_true_onehot.ravel(), y_score.ravel())
        plt.plot(recall, precision, lw=2, label=f'Micro-average (AP = {ap:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Precision-recall curves saved to {save_path}")
        
        plt.show()
    
    def evaluate_modality_contributions(self, y_true, text_pred, voice_pred, fusion_pred):
        """
        Evaluate the contribution of each modality to the fusion model.
        
        Args:
            y_true (array-like): True labels
            text_pred (array-like): Text modality predictions
            voice_pred (array-like): Voice modality predictions
            fusion_pred (array-like): Fusion model predictions
            
        Returns:
            dict: Dictionary of metrics for each modality
        """
        # Compute metrics for each modality
        text_metrics = self.compute_basic_metrics(y_true, text_pred)
        voice_metrics = self.compute_basic_metrics(y_true, voice_pred)
        fusion_metrics = self.compute_basic_metrics(y_true, fusion_pred)
        
        # Compute improvement of fusion over individual modalities
        improvement = {
            'text_to_fusion_accuracy': fusion_metrics['accuracy'] - text_metrics['accuracy'],
            'voice_to_fusion_accuracy': fusion_metrics['accuracy'] - voice_metrics['accuracy'],
            'text_to_fusion_f1_macro': fusion_metrics['f1_macro'] - text_metrics['f1_macro'],
            'voice_to_fusion_f1_macro': fusion_metrics['f1_macro'] - voice_metrics['f1_macro']
        }
        
        # Compute agreement between modalities
        agreement = np.mean(text_pred == voice_pred)
        
        # Compute cases where fusion corrects individual modalities
        text_wrong_fusion_right = np.logical_and(text_pred != y_true, fusion_pred == y_true)
        voice_wrong_fusion_right = np.logical_and(voice_pred != y_true, fusion_pred == y_true)
        both_wrong_fusion_right = np.logical_and(text_wrong_fusion_right, voice_wrong_fusion_right)
        
        correction_rates = {
            'text_wrong_fusion_right': np.mean(text_wrong_fusion_right),
            'voice_wrong_fusion_right': np.mean(voice_wrong_fusion_right),
            'both_wrong_fusion_right': np.mean(both_wrong_fusion_right)
        }
        
        return {
            'text_metrics': text_metrics,
            'voice_metrics': voice_metrics,
            'fusion_metrics': fusion_metrics,
            'improvement': improvement,
            'modality_agreement': agreement,
            'correction_rates': correction_rates
        }
    
    def plot_modality_comparison(self, y_true, text_pred, voice_pred, fusion_pred, figsize=(12, 6), save_path=None):
        """
        Plot comparison of modality performance.
        
        Args:
            y_true (array-like): True labels
            text_pred (array-like): Text modality predictions
            voice_pred (array-like): Voice modality predictions
            fusion_pred (array-like): Fusion model predictions
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the plot
        """
        # Compute metrics for each modality
        text_metrics = self.compute_basic_metrics(y_true, text_pred)
        voice_metrics = self.compute_basic_metrics(y_true, voice_pred)
        fusion_metrics = self.compute_basic_metrics(y_true, fusion_pred)
        
        # Prepare data for plotting
        modalities = ['Text', 'Voice', 'Fusion']
        accuracy = [text_metrics['accuracy'], voice_metrics['accuracy'], fusion_metrics['accuracy']]
        f1_macro = [text_metrics['f1_macro'], voice_metrics['f1_macro'], fusion_metrics['f1_macro']]
        
        # Plot comparison
        plt.figure(figsize=figsize)
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.bar(modalities, accuracy, color=['blue', 'green', 'red'])
        plt.ylim([0, 1])
        plt.title('Accuracy Comparison')
        plt.ylabel('Accuracy')
        
        # Add value labels
        for i, v in enumerate(accuracy):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        # Plot F1 score
        plt.subplot(1, 2, 2)
        plt.bar(modalities, f1_macro, color=['blue', 'green', 'red'])
        plt.ylim([0, 1])
        plt.title('F1 Score (Macro) Comparison')
        plt.ylabel('F1 Score')
        
        # Add value labels
        for i, v in enumerate(f1_macro):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Modality comparison saved to {save_path}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # This is a demonstration of how the evaluation metrics would be used
    # In a real scenario, you would use actual predictions and labels
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    n_classes = 7
    
    # True labels
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # Predicted labels for each modality
    text_pred = np.random.randint(0, n_classes, n_samples)
    voice_pred = np.random.randint(0, n_classes, n_samples)
    
    # Make fusion predictions better than individual modalities
    fusion_pred = np.copy(y_true)
    # Introduce some errors
    error_indices = np.random.choice(n_samples, int(n_samples * 0.2), replace=False)
    fusion_pred[error_indices] = np.random.randint(0, n_classes, len(error_indices))
    
    # Predicted probabilities
    y_score = np.random.rand(n_samples, n_classes)
    # Normalize to sum to 1
    y_score = y_score / y_score.sum(axis=1, keepdims=True)
    # Make the true class more likely
    for i, label in enumerate(y_true):
        y_score[i, label] += 0.5
    y_score = y_score / y_score.sum(axis=1, keepdims=True)
    
    # Initialize evaluator
    evaluator = EmotionEvaluator()
    
    # Compute and print basic metrics
    metrics = evaluator.compute_basic_metrics(y_true, fusion_pred)
    print("Basic Metrics:")
    for metric, value in metrics.items():
        if not metric.startswith(('precision_', 'recall_', 'f1_')) or metric.endswith(('macro', 'weighted')):
            print(f"  {metric}: {value:.4f}")
    
    # Compute and print classification report
    report = evaluator.compute_classification_report(y_true, fusion_pred, output_dict=False)
    print("\nClassification Report:")
    print(report)
    
    # Evaluate modality contributions
    contributions = evaluator.evaluate_modality_contributions(y_true, text_pred, voice_pred, fusion_pred)
    print("\nModality Contributions:")
    print(f"  Text Accuracy: {contributions['text_metrics']['accuracy']:.4f}")
    print(f"  Voice Accuracy: {contributions['voice_metrics']['accuracy']:.4f}")
    print(f"  Fusion Accuracy: {contributions['fusion_metrics']['accuracy']:.4f}")
    print(f"  Improvement over Text: {contributions['improvement']['text_to_fusion_accuracy']:.4f}")
    print(f"  Improvement over Voice: {contributions['improvement']['voice_to_fusion_accuracy']:.4f}")
    print(f"  Modality Agreement: {contributions['modality_agreement']:.4f}")
    
    print("\nEvaluation metrics module initialized.")
    print("Available methods:")
    print("- compute_basic_metrics: Compute accuracy, precision, recall, F1 score")
    print("- compute_confusion_matrix: Compute confusion matrix")
    print("- plot_confusion_matrix: Plot confusion matrix")
    print("- compute_classification_report: Compute detailed classification report")
    print("- compute_roc_auc: Compute ROC AUC scores")
    print("- plot_roc_curves: Plot ROC curves")
    print("- compute_precision_recall: Compute precision-recall curves")
    print("- plot_precision_recall_curves: Plot precision-recall curves")
    print("- evaluate_modality_contributions: Evaluate contribution of each modality")
    print("- plot_modality_comparison: Plot comparison of modality performance")
