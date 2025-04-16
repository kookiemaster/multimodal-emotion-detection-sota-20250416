"""
Main script for training and evaluating the MemoCMT model on IEMOCAP dataset.

This script implements the complete pipeline for replicating the MemoCMT model
results on the IEMOCAP dataset as described in the paper.
"""

import os
import torch
import logging
import argparse
from datetime import datetime
from tqdm import tqdm

# Import MemoCMT components
from src.memocmt.text.text_model import TextEmotionModel, TextProcessor
from src.memocmt.voice.voice_model import VoiceEmotionModel, VoiceProcessor
from src.memocmt.fusion.fusion_model import MemoCMTFusion
from src.memocmt.data.data_pipeline import prepare_iemocap_data
from src.memocmt.models.training import train_memocmt_iemocap
from src.memocmt.evaluation.metrics import evaluate_predictions, compare_with_published_results, get_published_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("memocmt_iemocap.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate MemoCMT model on IEMOCAP dataset")
    
    parser.add_argument("--data_dir", type=str, default="data/processed/iemocap",
                        help="Directory containing processed IEMOCAP data")
    parser.add_argument("--output_dir", type=str, default="results/memocmt_iemocap",
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizer")
    parser.add_argument("--fusion_weight", type=float, default=0.5,
                        help="Weight for fusion loss")
    parser.add_argument("--text_weight", type=float, default=0.25,
                        help="Weight for text loss")
    parser.add_argument("--voice_weight", type=float, default=0.25,
                        help="Weight for voice loss")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on (cuda or cpu)")
    parser.add_argument("--huggingface_token", type=str, default=None,
                        help="HuggingFace token for downloading models")
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)

def main():
    """Main function for training and evaluating MemoCMT on IEMOCAP."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set HuggingFace token if provided
    if args.huggingface_token:
        os.environ["HF_TOKEN"] = args.huggingface_token
        logger.info("HuggingFace token set")
    
    # Define emotion mapping for IEMOCAP
    emotion_map = {
        'angry': 0,
        'happy': 1,
        'sad': 2,
        'neutral': 3
    }
    
    # Initialize text and voice processors
    logger.info("Initializing text and voice processors...")
    text_processor = TextProcessor(bert_model="bert-base-uncased", max_length=128)
    voice_processor = VoiceProcessor(sample_rate=16000, max_duration=10, use_hubert=True)
    
    # Prepare data
    logger.info(f"Preparing IEMOCAP data from {args.data_dir}...")
    try:
        train_loader, test_loader = prepare_iemocap_data(
            processed_dir=args.data_dir,
            text_processor=text_processor,
            voice_processor=voice_processor,
            batch_size=args.batch_size,
            num_workers=4,
            device=args.device
        )
        logger.info(f"Data preparation complete. Train: {len(train_loader.dataset)} samples, Test: {len(test_loader.dataset)} samples")
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise
    
    # Initialize models
    logger.info("Initializing models...")
    text_model = TextEmotionModel(num_classes=len(emotion_map), dropout_rate=0.1, bert_model="bert-base-uncased")
    voice_model = VoiceEmotionModel(num_classes=len(emotion_map), dropout_rate=0.1)
    fusion_model = MemoCMTFusion(feature_dim=256, num_classes=len(emotion_map), fusion_method='min', dropout_rate=0.1)
    
    # Train model
    logger.info("Starting training...")
    try:
        history, test_acc = train_memocmt_iemocap(
            text_model=text_model,
            voice_model=voice_model,
            fusion_model=fusion_model,
            train_loader=train_loader,
            test_loader=test_loader,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            fusion_weight=args.fusion_weight,
            modality_weights=(args.text_weight, args.voice_weight),
            num_epochs=args.num_epochs,
            patience=args.patience,
            save_dir=os.path.join(output_dir, "checkpoints"),
            device=args.device
        )
        logger.info(f"Training complete. Test accuracy: {test_acc:.2f}%")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    
    # Evaluate model
    logger.info("Evaluating model...")
    try:
        # Find the predictions file
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        predictions_path = os.path.join(checkpoints_dir, "test_predictions.npz")
        
        if os.path.exists(predictions_path):
            # Evaluate predictions
            metrics = evaluate_predictions(
                predictions_path=predictions_path,
                emotion_map=emotion_map,
                output_dir=os.path.join(output_dir, "evaluation")
            )
            
            # Compare with published results
            published_results = get_published_results("memocmt", "iemocap")
            comparison = compare_with_published_results(
                metrics=metrics,
                published_results=published_results,
                dataset_name="IEMOCAP",
                model_name="MemoCMT",
                output_dir=os.path.join(output_dir, "comparison")
            )
            
            logger.info("Evaluation complete")
        else:
            logger.warning(f"Predictions file not found: {predictions_path}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise
    
    logger.info(f"All results saved to {output_dir}")

if __name__ == "__main__":
    main()
