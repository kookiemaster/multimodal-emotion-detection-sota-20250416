"""
Main script for text emotion detection component

This script demonstrates the usage of the text emotion detection component.
"""

import os
import sys
import torch
from transformers import AutoTokenizer

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text.text_processor import TextEmotionProcessor
from text.text_data import prepare_example_data

def main():
    """
    Main function to demonstrate text emotion detection.
    """
    print("Initializing Text Emotion Detection Component...")
    
    # Set HuggingFace token if available
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("HuggingFace token found in environment variables.")
    else:
        print("No HuggingFace token found. Some models may not be accessible.")
    
    # Initialize text emotion processor
    processor = TextEmotionProcessor(model_name="microsoft/deberta-v3-small")
    
    # Load model
    success = processor.load_model()
    if not success:
        print("Failed to load model. Using a simpler approach for demonstration.")
        # In a real implementation, we would handle this more gracefully
    
    # Prepare example data
    texts, _ = prepare_example_data()
    
    # Process each text and predict emotion
    print("\nProcessing example texts:")
    print("-" * 50)
    
    for i, text in enumerate(texts):
        print(f"Example {i+1}: \"{text}\"")
        
        try:
            # Predict emotion
            result = processor.predict_emotion(text)
            
            # Print results
            print(f"Predicted emotion: {result['predicted_emotion']}")
            print(f"Confidence: {result['confidence']:.4f}")
            
            # Print top 3 emotions by probability
            top_emotions = sorted(
                result['all_probabilities'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            print("Top 3 emotions:")
            for emotion, prob in top_emotions:
                print(f"  - {emotion}: {prob:.4f}")
                
        except Exception as e:
            print(f"Error processing text: {e}")
        
        print("-" * 50)
    
    print("Text emotion detection demonstration completed.")

if __name__ == "__main__":
    main()
