"""
Text Emotion Detection Component

This module implements text preprocessing and emotion detection using DeBERTa.
"""

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TextEmotionProcessor:
    """
    A class for processing text and detecting emotions using DeBERTa.
    """
    
    def __init__(self, model_name="microsoft/deberta-v3-small", num_labels=7):
        """
        Initialize the text emotion processor.
        
        Args:
            model_name (str): The name of the pre-trained model to use
            num_labels (int): Number of emotion classes to predict
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set HuggingFace token if available
        if "HF_TOKEN" in os.environ:
            self.hf_token = os.environ["HF_TOKEN"]
        else:
            self.hf_token = None
            
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        
        # Emotion labels (standard set)
        self.emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        
    def load_model(self):
        """
        Load the pre-trained model and tokenizer.
        """
        try:
            print(f"Loading tokenizer from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
            
            print(f"Loading model from {self.model_name}...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=self.num_labels,
                token=self.hf_token
            )
            self.model.to(self.device)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_text(self, text):
        """
        Preprocess text for emotion detection.
        
        Args:
            text (str): Input text to process
            
        Returns:
            dict: Tokenized text ready for model input
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
            
        # Basic preprocessing
        text = text.strip().lower()
        
        # Tokenize the text
        encoded_input = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        return encoded_input
    
    def predict_emotion(self, text):
        """
        Predict emotion from text.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary containing emotion predictions and probabilities
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess the text
        inputs = self.preprocess_text(text)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Convert to numpy for easier handling
        probs = probabilities.cpu().numpy()[0]
        
        # Get the predicted class
        predicted_class = np.argmax(probs)
        predicted_emotion = self.emotion_labels[predicted_class]
        
        # Create result dictionary
        result = {
            "predicted_emotion": predicted_emotion,
            "confidence": float(probs[predicted_class]),
            "all_probabilities": {emotion: float(prob) for emotion, prob in zip(self.emotion_labels, probs)}
        }
        
        return result
    
    def batch_predict(self, texts):
        """
        Predict emotions for a batch of texts.
        
        Args:
            texts (list): List of input texts
            
        Returns:
            list: List of prediction dictionaries
        """
        results = []
        for text in texts:
            result = self.predict_emotion(text)
            results.append(result)
        return results


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = TextEmotionProcessor()
    
    # Load model
    processor.load_model()
    
    # Example texts
    example_texts = [
        "I'm so happy today! Everything is going well.",
        "I'm feeling really sad and disappointed about the news.",
        "That makes me so angry! How could they do that?",
        "I'm a bit nervous about the upcoming presentation."
    ]
    
    # Make predictions
    for text in example_texts:
        result = processor.predict_emotion(text)
        print(f"Text: {text}")
        print(f"Predicted emotion: {result['predicted_emotion']} (Confidence: {result['confidence']:.4f})")
        print("All probabilities:", {k: f"{v:.4f}" for k, v in result['all_probabilities'].items()})
        print()
