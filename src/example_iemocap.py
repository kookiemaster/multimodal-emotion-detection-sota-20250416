"""
Example script for using COGMEN model with IEMOCAP data

This script demonstrates how to use the COGMEN model for multimodal emotion
recognition on the IEMOCAP dataset.

Usage:
    python example_iemocap.py

Requirements:
    - IEMOCAP_Final.csv file in the data/processed/iemocap directory
    - PyTorch
    - PyTorch Geometric
    - Sentence Transformers
    - pandas
    - numpy
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the COGMEN inference module
from src.cogmen_inference import COGMENInference, EMOTION_LABELS

def load_iemocap_sample(csv_path, sample_size=100):
    """
    Load a sample of IEMOCAP data for demonstration purposes.
    
    Args:
        csv_path: Path to IEMOCAP CSV file
        sample_size: Number of samples to load
        
    Returns:
        sample_df: DataFrame with IEMOCAP sample data
    """
    # Check if the CSV file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"IEMOCAP CSV file not found at {csv_path}")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded IEMOCAP data with {len(df)} utterances")
    
    # Get a balanced sample of emotions
    sample_df = pd.DataFrame()
    emotions = df['emotion'].unique()
    
    # Calculate samples per emotion
    samples_per_emotion = max(1, sample_size // len(emotions))
    
    # Get samples for each emotion
    for emotion in emotions:
        emotion_df = df[df['emotion'] == emotion]
        if len(emotion_df) > samples_per_emotion:
            emotion_sample = emotion_df.sample(samples_per_emotion, random_state=42)
        else:
            emotion_sample = emotion_df
        sample_df = pd.concat([sample_df, emotion_sample])
    
    # Shuffle the sample
    sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Limit to sample_size
    if len(sample_df) > sample_size:
        sample_df = sample_df.iloc[:sample_size]
    
    print(f"Created balanced sample with {len(sample_df)} utterances")
    
    return sample_df

def visualize_results(results_df, output_dir):
    """
    Visualize the prediction results.
    
    Args:
        results_df: DataFrame with prediction results
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Map emotion labels to indices
    emotion_to_idx = {v.lower(): k for k, v in EMOTION_LABELS.items()}
    
    # Get ground truth and predictions
    ground_truth = results_df['emotion'].str.lower().map(emotion_to_idx).tolist()
    predictions = results_df['predicted_emotion'].str.lower().map(emotion_to_idx).tolist()
    
    # Get emotion labels
    emotion_labels = [EMOTION_LABELS[i].lower() for i in range(len(EMOTION_LABELS))]
    
    # Create confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=range(len(EMOTION_LABELS)))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Plot emotion distribution
    plt.figure(figsize=(10, 6))
    emotion_counts = results_df['emotion'].str.lower().value_counts()
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.title('Emotion Distribution in Sample')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_distribution.png'))
    plt.close()
    
    # Plot accuracy by emotion
    plt.figure(figsize=(10, 6))
    accuracies = []
    for emotion in emotion_labels:
        emotion_df = results_df[results_df['emotion'].str.lower() == emotion]
        if len(emotion_df) > 0:
            accuracy = (emotion_df['emotion'].str.lower() == emotion_df['predicted_emotion'].str.lower()).mean() * 100
            accuracies.append(accuracy)
        else:
            accuracies.append(0)
    
    sns.barplot(x=emotion_labels, y=accuracies)
    plt.xlabel('Emotion')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy by Emotion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_emotion.png'))
    plt.close()
    
    # Generate classification report
    report = classification_report(ground_truth, predictions, 
                                  target_names=emotion_labels, 
                                  output_dict=True)
    
    # Save classification report as CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    print(f"Visualizations saved to {output_dir}")

def main():
    """
    Main function to demonstrate COGMEN inference on IEMOCAP data.
    """
    # Define paths
    iemocap_csv = "/home/ubuntu/multimodal-emotion-detection-sota-20250416/data/processed/iemocap/IEMOCAP_Final.csv"
    output_dir = "/home/ubuntu/multimodal-emotion-detection-sota-20250416/results/example_run"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load IEMOCAP sample
    try:
        sample_df = load_iemocap_sample(iemocap_csv, sample_size=100)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the IEMOCAP_Final.csv file is in the correct location.")
        return
    
    # Save sample to CSV
    sample_path = os.path.join(output_dir, "iemocap_sample.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"Saved sample to {sample_path}")
    
    # Initialize COGMEN inference
    print("\nInitializing COGMEN model...")
    cogmen = COGMENInference()
    
    # Process IEMOCAP sample and get predictions
    print("\nProcessing IEMOCAP sample and generating predictions...")
    results_df = cogmen.process_iemocap_csv(sample_path)
    
    # Save results to CSV
    results_path = os.path.join(output_dir, "cogmen_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved predictions to {results_path}")
    
    # Evaluate predictions
    print("\nEvaluating predictions...")
    metrics = cogmen.evaluate_predictions(results_df)
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
    print("\nPer-Class Metrics:")
    for emotion, class_metric in metrics['class_metrics'].items():
        print(f"{emotion.capitalize()}: Accuracy = {class_metric['accuracy']:.2f}%, Count = {class_metric['count']}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(results_df, output_dir)
    
    print("\nExample completed successfully!")
    print(f"All results and visualizations are saved in {output_dir}")
    
    # Compare with reported SOTA results
    print("\nComparison with reported SOTA results:")
    print("COGMEN paper reported 80.47% weighted accuracy on IEMOCAP")
    print(f"Our sample run achieved {metrics['overall_accuracy']:.2f}% accuracy")
    print("Note: Our results are based on a small sample and simplified implementation,")
    print("      so they may differ from the reported SOTA results.")

if __name__ == "__main__":
    main()
