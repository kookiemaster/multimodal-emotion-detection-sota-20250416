#!/usr/bin/env python3
"""
Dataset Preprocessing Script for Multimodal Emotion Recognition

This script provides utilities for preprocessing the IEMOCAP, MELD, and ESD datasets
according to the specifications in the MemoCMT and SDT papers.
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
import librosa
import torch
from tqdm import tqdm
import logging
import shutil
from pathlib import Path
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_iemocap(data_root, output_dir):
    """
    Preprocess the IEMOCAP dataset according to MemoCMT paper specifications
    
    Args:
        data_root (str): Path to the IEMOCAP_full_release directory
        output_dir (str): Directory to save processed data
    """
    logger.info(f"Preprocessing IEMOCAP dataset from {data_root}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define emotion mapping (according to MemoCMT paper)
    emotion_map = {
        'ang': 'angry',
        'hap': 'happy',
        'exc': 'happy',  # Excitement is mapped to happy as per common practice
        'sad': 'sad',
        'neu': 'neutral',
        'fru': 'frustrated',
        'fea': 'fear',
        'sur': 'surprise',
        'dis': 'disgust',
        'oth': 'other'
    }
    
    # Emotions to keep (based on MemoCMT paper)
    target_emotions = ['angry', 'happy', 'sad', 'neutral']
    
    # Initialize lists to store data
    data = []
    
    # Process each session
    for session in range(1, 6):
        session_dir = os.path.join(data_root, f'Session{session}')
        
        # Process evaluation files
        eval_dir = os.path.join(session_dir, 'dialog', 'EmoEvaluation')
        wav_dir = os.path.join(session_dir, 'sentences', 'wav')
        
        for eval_file in os.listdir(eval_dir):
            if not eval_file.endswith('.txt'):
                continue
                
            dialog_id = eval_file.split('.')[0]
            
            # Read evaluation file
            eval_path = os.path.join(eval_dir, eval_file)
            with open(eval_path, 'r') as f:
                lines = f.readlines()
            
            # Extract emotion labels
            for line in lines:
                if line.startswith('['):
                    parts = line.strip().split('\t')
                    if len(parts) < 4:
                        continue
                    
                    utterance_id = parts[1]
                    emotion = parts[2]
                    
                    # Map emotion to standard categories
                    if emotion in emotion_map:
                        emotion = emotion_map[emotion]
                    else:
                        continue
                    
                    # Skip emotions not in target list
                    if emotion not in target_emotions:
                        continue
                    
                    # Find corresponding audio file
                    speaker = utterance_id.split('_')[0]
                    wav_file = f"{utterance_id}.wav"
                    wav_path = os.path.join(wav_dir, dialog_id, wav_file)
                    
                    if not os.path.exists(wav_path):
                        logger.warning(f"Audio file not found: {wav_path}")
                        continue
                    
                    # Find corresponding transcript
                    transcript = ""
                    for i, line in enumerate(lines):
                        if utterance_id in line and ":" in line:
                            transcript = line.split(":", 1)[1].strip()
                            break
                    
                    # Add to dataset
                    data.append({
                        'id': utterance_id,
                        'session': f"Session{session}",
                        'dialog_id': dialog_id,
                        'speaker': speaker,
                        'emotion': emotion,
                        'audio_path': wav_path,
                        'text': transcript
                    })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'iemocap_metadata.csv')
    df.to_csv(metadata_path, index=False)
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Create processed directory structure
    processed_audio_dir = os.path.join(output_dir, 'audio')
    os.makedirs(processed_audio_dir, exist_ok=True)
    
    # Copy and preprocess audio files
    logger.info("Processing audio files...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        src_path = row['audio_path']
        dst_path = os.path.join(processed_audio_dir, f"{row['id']}.wav")
        
        # Copy audio file
        shutil.copy2(src_path, dst_path)
        
        # Update path in DataFrame
        df.at[idx, 'audio_path'] = dst_path
    
    # Save updated metadata
    df.to_csv(metadata_path, index=False)
    
    # Create train/val/test splits (following MemoCMT paper)
    # Sessions 1-4 for training, Session 5 for testing
    train_df = df[df['session'].isin([f"Session{i}" for i in range(1, 5)])]
    test_df = df[df['session'] == "Session5"]
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'iemocap_train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'iemocap_test.csv'), index=False)
    
    logger.info(f"IEMOCAP preprocessing complete. Total samples: {len(df)}")
    logger.info(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Save statistics
    emotion_counts = df['emotion'].value_counts().to_dict()
    stats = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'emotion_distribution': emotion_counts
    }
    
    with open(os.path.join(output_dir, 'iemocap_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    return df

def preprocess_meld(data_root, output_dir):
    """
    Preprocess the MELD dataset according to SDT paper specifications
    
    Args:
        data_root (str): Path to the MELD.Raw directory
        output_dir (str): Directory to save processed data
    """
    logger.info(f"Preprocessing MELD dataset from {data_root}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define paths
    train_csv = os.path.join(data_root, 'train_sent_emo.csv')
    dev_csv = os.path.join(data_root, 'dev_sent_emo.csv')
    test_csv = os.path.join(data_root, 'test_sent_emo.csv')
    
    # Check if files exist
    for file_path in [train_csv, dev_csv, test_csv]:
        if not os.path.exists(file_path):
            logger.error(f"Required file not found: {file_path}")
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Load CSV files
    train_df = pd.read_csv(train_csv)
    dev_df = pd.read_csv(dev_csv)
    test_df = pd.read_csv(test_csv)
    
    # Process each split
    for split, df in [('train', train_df), ('dev', dev_df), ('test', test_df)]:
        logger.info(f"Processing {split} split...")
        
        # Create processed directory structure
        processed_audio_dir = os.path.join(output_dir, f'{split}_audio')
        os.makedirs(processed_audio_dir, exist_ok=True)
        
        # Process each row
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Get audio file path
            season = row['Season']
            episode = row['Episode']
            utterance_id = row['Utterance_ID']
            
            src_path = os.path.join(
                data_root, 
                split, 
                f'dia{episode}_utt{utterance_id}.mp4'
            )
            
            if not os.path.exists(src_path):
                logger.warning(f"Audio file not found: {src_path}")
                continue
            
            # Create destination path
            dst_path = os.path.join(
                processed_audio_dir, 
                f"dia{episode}_utt{utterance_id}.wav"
            )
            
            # Convert mp4 to wav using librosa
            try:
                y, sr = librosa.load(src_path, sr=None)
                librosa.output.write_wav(dst_path, y, sr)
                
                # Update path in DataFrame
                df.at[idx, 'audio_path'] = dst_path
            except Exception as e:
                logger.error(f"Error processing {src_path}: {e}")
                continue
        
        # Save processed metadata
        output_csv = os.path.join(output_dir, f'meld_{split}.csv')
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved {split} metadata to {output_csv}")
    
    # Combine all splits for statistics
    all_df = pd.concat([train_df, dev_df, test_df])
    
    # Save statistics
    emotion_counts = all_df['Emotion'].value_counts().to_dict()
    stats = {
        'total_samples': len(all_df),
        'train_samples': len(train_df),
        'dev_samples': len(dev_df),
        'test_samples': len(test_df),
        'emotion_distribution': emotion_counts
    }
    
    with open(os.path.join(output_dir, 'meld_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"MELD preprocessing complete. Total samples: {len(all_df)}")
    
    return all_df

def preprocess_esd(data_root, output_dir):
    """
    Preprocess the ESD dataset according to MemoCMT paper specifications
    
    Args:
        data_root (str): Path to the ESD directory
        output_dir (str): Directory to save processed data
    """
    logger.info(f"Preprocessing ESD dataset from {data_root}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define emotion mapping
    emotion_map = {
        '0': 'neutral',
        '1': 'happy',
        '2': 'angry',
        '3': 'sad',
        '4': 'surprise'
    }
    
    # Initialize lists to store data
    data = []
    
    # Process English speakers (as per MemoCMT paper)
    for speaker_id in range(0, 10):  # English speakers are 0-9
        speaker_dir = os.path.join(data_root, f"{speaker_id}")
        
        if not os.path.exists(speaker_dir):
            logger.warning(f"Speaker directory not found: {speaker_dir}")
            continue
        
        # Process each emotion
        for emotion_id, emotion_name in emotion_map.items():
            emotion_dir = os.path.join(speaker_dir, emotion_id)
            
            if not os.path.exists(emotion_dir):
                logger.warning(f"Emotion directory not found: {emotion_dir}")
                continue
            
            # Process each audio file
            for audio_file in os.listdir(emotion_dir):
                if not audio_file.endswith('.wav'):
                    continue
                
                audio_path = os.path.join(emotion_dir, audio_file)
                
                # Extract sentence ID
                sentence_id = audio_file.split('.')[0]
                
                # Add to dataset
                data.append({
                    'id': f"{speaker_id}_{emotion_id}_{sentence_id}",
                    'speaker': speaker_id,
                    'emotion': emotion_name,
                    'audio_path': audio_path,
                    'sentence_id': sentence_id
                })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'esd_metadata.csv')
    df.to_csv(metadata_path, index=False)
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Create processed directory structure
    processed_audio_dir = os.path.join(output_dir, 'audio')
    os.makedirs(processed_audio_dir, exist_ok=True)
    
    # Copy and preprocess audio files
    logger.info("Processing audio files...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        src_path = row['audio_path']
        dst_path = os.path.join(processed_audio_dir, f"{row['id']}.wav")
        
        # Copy audio file
        shutil.copy2(src_path, dst_path)
        
        # Update path in DataFrame
        df.at[idx, 'audio_path'] = dst_path
    
    # Save updated metadata
    df.to_csv(metadata_path, index=False)
    
    # Create train/val/test splits (following MemoCMT paper)
    # 80% train, 10% validation, 10% test
    speakers = df['speaker'].unique()
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(speakers)
    
    train_speakers = speakers[:8]  # 80% for training
    val_speakers = speakers[8:9]   # 10% for validation
    test_speakers = speakers[9:]   # 10% for testing
    
    train_df = df[df['speaker'].isin(train_speakers)]
    val_df = df[df['speaker'].isin(val_speakers)]
    test_df = df[df['speaker'].isin(test_speakers)]
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'esd_train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'esd_val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'esd_test.csv'), index=False)
    
    logger.info(f"ESD preprocessing complete. Total samples: {len(df)}")
    logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")
    
    # Save statistics
    emotion_counts = df['emotion'].value_counts().to_dict()
    stats = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'emotion_distribution': emotion_counts
    }
    
    with open(os.path.join(output_dir, 'esd_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Preprocess datasets for multimodal emotion recognition')
    parser.add_argument('--dataset', type=str, required=True, choices=['iemocap', 'meld', 'esd'],
                        help='Dataset to preprocess')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to the raw dataset directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save processed data (default: data/{dataset}/processed)')
    
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join('data', args.dataset, 'processed')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Preprocess the specified dataset
    if args.dataset == 'iemocap':
        preprocess_iemocap(args.data_root, args.output_dir)
    elif args.dataset == 'meld':
        preprocess_meld(args.data_root, args.output_dir)
    elif args.dataset == 'esd':
        preprocess_esd(args.data_root, args.output_dir)

if __name__ == "__main__":
    main()
