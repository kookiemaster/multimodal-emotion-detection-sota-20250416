"""
Script to train and evaluate the multimodal emotion detection model on IEMOCAP data.

This script runs the training pipeline with a small subset of the data
to demonstrate the functionality while working within resource constraints.
"""

import os
import torch
import json
from src.training_pipeline import train_memocmt_iemocap

# Set paths
CSV_PATH = "/home/ubuntu/upload/IEMOCAP_Final.csv"
OUTPUT_DIR = "/home/ubuntu/multimodal-emotion-detection-sota-20250416/results/iemocap_run"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set training parameters
params = {
    "csv_path": CSV_PATH,
    "output_dir": OUTPUT_DIR,
    "model_name": "bert-base-uncased",
    "num_classes": 8,
    "batch_size": 16,  # Smaller batch size for CPU
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "fusion_weight": 0.5,
    "modality_weights": (0.25, 0.25),
    "num_epochs": 3,  # Reduced epochs for demonstration
    "patience": 2,
    "max_samples": 500,  # Limited samples for CPU training
    "use_audio": True,  # Use both text and audio modalities
    "device": torch.device('cpu')  # Force CPU
}

# Save parameters
with open(os.path.join(OUTPUT_DIR, "params.json"), "w") as f:
    # Convert non-serializable objects to strings
    serializable_params = {k: str(v) if not isinstance(v, (int, float, bool, list, dict, str, type(None))) else v 
                          for k, v in params.items()}
    json.dump(serializable_params, f, indent=4)

print("Starting training with the following parameters:")
for k, v in params.items():
    print(f"{k}: {v}")

# Run training
try:
    metrics, history = train_memocmt_iemocap(**params)
    
    # Save metrics
    with open(os.path.join(OUTPUT_DIR, "final_metrics.json"), "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for k, v in metrics.items():
            if hasattr(v, 'tolist'):
                serializable_metrics[k] = v.tolist()
            else:
                serializable_metrics[k] = v
        json.dump(serializable_metrics, f, indent=4)
    
    # Save history
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    
    print("\nTraining completed successfully!")
    print(f"Results saved to {OUTPUT_DIR}")
    
    # Print final metrics
    print("\nFinal Metrics:")
    for key, value in metrics.items():
        if isinstance(value, list) and len(value) > 10:
            print(f"{key}: [...]")
        elif isinstance(value, list):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.2f}")
            
except Exception as e:
    print(f"Error during training: {e}")
