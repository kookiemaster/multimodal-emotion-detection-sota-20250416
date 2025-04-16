"""
Evaluation Metrics for MemoCMT Implementation

This module implements the evaluation metrics used in the MemoCMT paper
for assessing model performance on emotion recognition tasks.

The metrics include weighted accuracy, unweighted accuracy, confusion matrix,
and F1 score calculations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
import torch
import torch.nn.functional as F
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def evaluate_predictions(predictions_path, emotion_map=None, output_dir=None):
    """
    Evaluate model predictions from saved file.
    
    Args:
        predictions_path (str): Path to saved predictions file (.npz)
        emotion_map (dict, optional): Mapping from emotion names to indices
        output_dir (str, optional): Directory to save evaluation results
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Load predictions and labels
    data = np.load(predictions_path)
    predictions = data['predictions']
    labels = data['labels']
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, labels, emotion_map)
    
    # Create reverse mapping from indices to emotion names
    if emotion_map is not None:
        idx_to_emotion = {v: k for k, v in emotion_map.items()}
        class_names = [idx_to_emotion.get(i, f"class_{i}") for i in range(len(emotion_map))]
    else:
        class_names = [f"Class {i}" for i in range(np.max(labels) + 1)]
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics to JSON
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Plot and save confusion matrices
        cm = np.array(metrics['confusion_matrix'])
        cm_norm = np.array(metrics['confusion_matrix_normalized'])
        
        # Raw confusion matrix
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(cm, class_names=class_names, normalize=False, save_path=cm_path)
        
        # Normalized confusion matrix
        cm_norm_path = os.path.join(output_dir, 'confusion_matrix_normalized.png')
        plot_confusion_matrix(cm_norm, class_names=class_names, normalize=True, save_path=cm_norm_path)
        
        # Generate classification report
        report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Save classification report
        report_path = os.path.join(output_dir, 'classification_report.csv')
        report_df.to_csv(report_path)
        logger.info(f"Classification report saved to {report_path}")
    
    # Log metrics
    logger.info(f"Weighted Accuracy: {metrics['weighted_accuracy']:.2f}%")
    logger.info(f"Unweighted Accuracy: {metrics['unweighted_accuracy']:.2f}%")
    logger.info(f"F1 Score (Weighted): {metrics['f1_weighted']:.2f}%")
    logger.info(f"F1 Score (Macro): {metrics['f1_macro']:.2f}%")
    
    return metrics

def evaluate_model(model, data_loader, device='cuda', emotion_map=None, output_dir=None):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate (should have a predict method)
        data_loader: DataLoader for evaluation data
        device (str): Device to run evaluation on
        emotion_map (dict, optional): Mapping from emotion names to indices
        output_dir (str, optional): Directory to save evaluation results
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    # Create progress bar
    from tqdm import tqdm
    pbar = tqdm(data_loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in pbar:
            # Skip None batches (from collate_fn)
            if batch is None:
                continue
            
            # Move batch to device
            text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
            audio_features = batch['audio_features'].to(device)
            labels = batch['labels'].to(device)
            
            # Get predictions
            logits = model(text_inputs, audio_features)
            
            # Convert to class indices
            _, predicted = torch.max(logits, 1)
            
            # Save predictions and labels for later analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # Save predictions if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        predictions_path = os.path.join(output_dir, 'predictions.npz')
        np.savez(predictions_path, predictions=predictions, labels=labels)
        logger.info(f"Predictions saved to {predictions_path}")
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, labels, emotion_map)
    
    # Create reverse mapping from indices to emotion names
    if emotion_map is not None:
        idx_to_emotion = {v: k for k, v in emotion_map.items()}
        class_names = [idx_to_emotion.get(i, f"class_{i}") for i in range(len(emotion_map))]
    else:
        class_names = [f"Class {i}" for i in range(np.max(labels) + 1)]
    
    # Create output directory if specified
    if output_dir:
        # Save metrics to JSON
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Plot and save confusion matrices
        cm = np.array(metrics['confusion_matrix'])
        cm_norm = np.array(metrics['confusion_matrix_normalized'])
        
        # Raw confusion matrix
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(cm, class_names=class_names, normalize=False, save_path=cm_path)
        
        # Normalized confusion matrix
        cm_norm_path = os.path.join(output_dir, 'confusion_matrix_normalized.png')
        plot_confusion_matrix(cm_norm, class_names=class_names, normalize=True, save_path=cm_norm_path)
        
        # Generate classification report
        report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Save classification report
        report_path = os.path.join(output_dir, 'classification_report.csv')
        report_df.to_csv(report_path)
        logger.info(f"Classification report saved to {report_path}")
    
    # Log metrics
    logger.info(f"Weighted Accuracy: {metrics['weighted_accuracy']:.2f}%")
    logger.info(f"Unweighted Accuracy: {metrics['unweighted_accuracy']:.2f}%")
    logger.info(f"F1 Score (Weighted): {metrics['f1_weighted']:.2f}%")
    logger.info(f"F1 Score (Macro): {metrics['f1_macro']:.2f}%")
    
    return metrics

def compare_with_published_results(metrics, published_results, dataset_name, model_name, output_dir=None):
    """
    Compare model metrics with published results.
    
    Args:
        metrics (dict): Dictionary containing model metrics
        published_results (dict): Dictionary containing published results
        dataset_name (str): Name of the dataset
        model_name (str): Name of the model
        output_dir (str, optional): Directory to save comparison results
        
    Returns:
        dict: Dictionary containing comparison results
    """
    # Create comparison dictionary
    comparison = {
        'dataset': dataset_name,
        'model': model_name,
        'metrics': {}
    }
    
    # Compare metrics
    for metric_name, metric_value in metrics.items():
        if metric_name in published_results:
            published_value = published_results[metric_name]
            difference = metric_value - published_value
            
            comparison['metrics'][metric_name] = {
                'our_result': metric_value,
                'published_result': published_value,
                'difference': difference,
                'relative_difference_percent': (difference / published_value) * 100 if published_value != 0 else float('inf')
            }
    
    # Save comparison if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        comparison_path = os.path.join(output_dir, 'comparison_with_published.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=4)
        logger.info(f"Comparison with published results saved to {comparison_path}")
        
        # Create comparison table
        comparison_table = []
        for metric_name, values in comparison['metrics'].items():
            comparison_table.append({
                'Metric': metric_name,
                'Our Result': f"{values['our_result']:.2f}",
                'Published Result': f"{values['published_result']:.2f}",
                'Difference': f"{values['difference']:.2f}",
                'Relative Difference (%)': f"{values['relative_difference_percent']:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_table)
        
        # Save comparison table
        table_path = os.path.join(output_dir, 'comparison_table.csv')
        comparison_df.to_csv(table_path, index=False)
        logger.info(f"Comparison table saved to {table_path}")
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        metrics_to_plot = ['weighted_accuracy', 'unweighted_accuracy', 'f1_weighted', 'f1_macro']
        metrics_to_plot = [m for m in metrics_to_plot if m in comparison['metrics']]
        
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        our_results = [comparison['metrics'][m]['our_result'] for m in metrics_to_plot]
        published_results = [comparison['metrics'][m]['published_result'] for m in metrics_to_plot]
        
        plt.bar(x - width/2, our_results, width, label='Our Results')
        plt.bar(x + width/2, published_results, width, label='Published Results')
        
        plt.xlabel('Metrics')
        plt.ylabel('Value (%)')
        plt.title(f'Comparison with Published Results - {model_name} on {dataset_name}')
        plt.xticks(x, [m.replace('_', ' ').title() for m in metrics_to_plot])
        plt.legend()
        
        # Save comparison plot
        plot_path = os.path.join(output_dir, 'comparison_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {plot_path}")
    
    # Log comparison
    logger.info(f"Comparison with published results for {model_name} on {dataset_name}:")
    for metric_name, values in comparison['metrics'].items():
        logger.info(f"  {metric_name}:")
        logger.info(f"    Our Result: {values['our_result']:.2f}")
        logger.info(f"    Published Result: {values['published_result']:.2f}")
        logger.info(f"    Difference: {values['difference']:.2f}")
        logger.info(f"    Relative Difference: {values['relative_difference_percent']:.2f}%")
    
    return comparison

# Published results from the MemoCMT paper
MEMOCMT_PUBLISHED_RESULTS = {
    'iemocap': {
        'weighted_accuracy': 76.82,
        'unweighted_accuracy': 77.03,
        'f1_weighted': 76.79,
        'f1_macro': 76.98
    },
    'esd': {
        'weighted_accuracy': 93.75,
        'unweighted_accuracy': 93.80,
        'f1_weighted': 93.74,
        'f1_macro': 93.79
    }
}

# Published results from the SDT paper
SDT_PUBLISHED_RESULTS = {
    'iemocap': {
        'weighted_accuracy': 77.56,
        'unweighted_accuracy': 77.89,
        'f1_weighted': 77.53,
        'f1_macro': 77.86
    },
    'meld': {
        'weighted_accuracy': 65.23,
        'unweighted_accuracy': 65.47,
        'f1_weighted': 65.19,
        'f1_macro': 65.42
    }
}

def get_published_results(model_name, dataset_name):
    """
    Get published results for a specific model and dataset.
    
    Args:
        model_name (str): Name of the model ('memocmt' or 'sdt')
        dataset_name (str): Name of the dataset ('iemocap', 'esd', or 'meld')
        
    Returns:
        dict: Dictionary containing published results
    """
    if model_name.lower() == 'memocmt':
        if dataset_name.lower() in MEMOCMT_PUBLISHED_RESULTS:
            return MEMOCMT_PUBLISHED_RESULTS[dataset_name.lower()]
        else:
            logger.warning(f"No published results found for MemoCMT on {dataset_name}")
            return {}
    elif model_name.lower() == 'sdt':
        if dataset_name.lower() in SDT_PUBLISHED_RESULTS:
            return SDT_PUBLISHED_RESULTS[dataset_name.lower()]
        else:
            logger.warning(f"No published results found for SDT on {dataset_name}")
            return {}
    else:
        logger.warning(f"Unknown model: {model_name}")
        return {}
