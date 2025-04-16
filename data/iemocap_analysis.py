"""
Analysis script for the IEMOCAP CSV dataset.

This script analyzes the structure and content of the IEMOCAP_Final.csv file
to understand the data distribution and prepare for model implementation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Path to the CSV file
csv_path = '/home/ubuntu/upload/IEMOCAP_Final.csv'

def analyze_iemocap_csv():
    """Analyze the IEMOCAP CSV file structure and content."""
    print(f"Reading CSV file from: {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Basic information
    print(f"\nDataset shape: {df.shape}")
    print(f"Number of samples: {len(df)}")
    
    # Column information
    print("\nColumns in the dataset:")
    for col in df.columns:
        print(f"- {col}")
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Emotion distribution
    print("\nEmotion distribution:")
    emotion_counts = Counter(df['Major_emotion'].str.strip())
    for emotion, count in emotion_counts.most_common():
        print(f"- {emotion}: {count} ({count/len(df)*100:.2f}%)")
    
    # Session distribution
    print("\nSession distribution:")
    session_counts = Counter(df['Session'])
    for session, count in session_counts.most_common():
        print(f"- {session}: {count} ({count/len(df)*100:.2f}%)")
    
    # Speaker distribution
    print("\nSpeaker distribution:")
    speaker_counts = Counter(df['Speaker_id'].apply(lambda x: x.split('_')[0]))
    for speaker, count in speaker_counts.most_common():
        print(f"- {speaker}: {count} ({count/len(df)*100:.2f}%)")
    
    # Transcript length distribution
    df['transcript_length'] = df['Transcript'].apply(len)
    print("\nTranscript length statistics:")
    print(f"- Min: {df['transcript_length'].min()}")
    print(f"- Max: {df['transcript_length'].max()}")
    print(f"- Mean: {df['transcript_length'].mean():.2f}")
    print(f"- Median: {df['transcript_length'].median()}")
    
    # Dimensional ratings
    print("\nDimensional ratings statistics:")
    for dim in ['Arousal', 'Valence', 'Dominance']:
        if dim in df.columns:
            try:
                # Convert to numeric if possible
                values = pd.to_numeric(df[dim], errors='coerce')
                print(f"- {dim}:")
                print(f"  - Min: {values.min()}")
                print(f"  - Max: {values.max()}")
                print(f"  - Mean: {values.mean():.2f}")
                print(f"  - Median: {values.median()}")
            except:
                print(f"- {dim}: Could not convert to numeric for statistics")
    
    return df

def plot_emotion_distribution(df):
    """Plot the distribution of emotions in the dataset."""
    try:
        # Count emotions
        emotion_counts = Counter(df['Major_emotion'].str.strip())
        emotions = [e for e, _ in emotion_counts.most_common()]
        counts = [c for _, c in emotion_counts.most_common()]
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(emotions, counts)
        plt.title('Emotion Distribution in IEMOCAP Dataset')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        output_dir = os.path.dirname(csv_path)
        plt.savefig(os.path.join(output_dir, 'emotion_distribution.png'))
        print(f"Emotion distribution plot saved to {os.path.join(output_dir, 'emotion_distribution.png')}")
    except Exception as e:
        print(f"Error creating emotion distribution plot: {e}")

def main():
    """Main function to analyze the IEMOCAP CSV file."""
    print("Starting IEMOCAP CSV analysis...")
    df = analyze_iemocap_csv()
    
    try:
        plot_emotion_distribution(df)
    except Exception as e:
        print(f"Error in plotting: {e}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
