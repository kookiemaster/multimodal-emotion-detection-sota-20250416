"""
Minimal Example for Multimodal Emotion Detection

This script demonstrates the core functionality of the multimodal emotion detection system
without requiring the full PyTorch installation.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

class MinimalEmotionDetector:
    """
    A minimal implementation of the multimodal emotion detection system.
    """
    
    def __init__(self):
        """
        Initialize the minimal emotion detector.
        """
        self.emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        print(f"Initialized MinimalEmotionDetector with {len(self.emotion_labels)} emotion classes")
    
    def process_text(self, text):
        """
        Process text input and predict emotions.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Emotion prediction result
        """
        print(f"Processing text: '{text}'")
        
        # Simulate text processing
        # In a real implementation, this would use the DeBERTa model
        
        # Simple rule-based prediction for demonstration
        text_lower = text.lower()
        
        # Check for emotion keywords
        emotion_keywords = {
            "angry": ["angry", "mad", "furious", "annoyed", "irritated"],
            "disgust": ["disgust", "disgusted", "gross", "revolting"],
            "fear": ["fear", "afraid", "scared", "terrified", "worried", "anxious"],
            "happy": ["happy", "joy", "delighted", "pleased", "glad", "excited"],
            "neutral": ["neutral", "normal", "fine", "okay", "ok"],
            "sad": ["sad", "unhappy", "depressed", "down", "miserable"],
            "surprise": ["surprise", "surprised", "shocked", "astonished", "amazed"]
        }
        
        # Count emotion keywords
        emotion_scores = {emotion: 0 for emotion in self.emotion_labels}
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_scores[emotion] += 1
        
        # If no keywords found, default to neutral
        if sum(emotion_scores.values()) == 0:
            emotion_scores["neutral"] = 1
        
        # Convert to probabilities
        total = sum(emotion_scores.values())
        probabilities = {emotion: score / total for emotion, score in emotion_scores.items()}
        
        # Get the predicted emotion
        predicted_emotion = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_emotion]
        
        return {
            "predicted_emotion": predicted_emotion,
            "confidence": confidence,
            "all_probabilities": probabilities
        }
    
    def process_audio(self, audio_path):
        """
        Process audio input and predict emotions.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            dict: Emotion prediction result
        """
        print(f"Processing audio: '{audio_path}'")
        
        # Simulate audio processing
        # In a real implementation, this would use the Semi-CNN model
        
        # For demonstration, generate random predictions with a bias
        # based on the filename if it contains emotion keywords
        
        audio_lower = audio_path.lower()
        
        # Initialize with equal probabilities
        probabilities = {emotion: 1.0 / len(self.emotion_labels) for emotion in self.emotion_labels}
        
        # Check for emotion keywords in filename
        for emotion in self.emotion_labels:
            if emotion in audio_lower:
                # Bias towards this emotion
                probabilities[emotion] += 0.5
        
        # Normalize probabilities
        total = sum(probabilities.values())
        probabilities = {emotion: prob / total for emotion, prob in probabilities.items()}
        
        # Add some randomness
        np.random.seed(int(datetime.now().timestamp()))
        for emotion in self.emotion_labels:
            probabilities[emotion] += np.random.uniform(-0.1, 0.1)
            probabilities[emotion] = max(0, probabilities[emotion])
        
        # Normalize again
        total = sum(probabilities.values())
        probabilities = {emotion: prob / total for emotion, prob in probabilities.items()}
        
        # Get the predicted emotion
        predicted_emotion = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_emotion]
        
        return {
            "predicted_emotion": predicted_emotion,
            "confidence": confidence,
            "all_probabilities": probabilities
        }
    
    def fuse_predictions(self, text_result, audio_result):
        """
        Fuse text and audio predictions.
        
        Args:
            text_result (dict): Text prediction result
            audio_result (dict): Audio prediction result
            
        Returns:
            dict: Fused prediction result
        """
        print("Fusing text and audio predictions")
        
        # Extract probabilities
        text_probs = text_result["all_probabilities"]
        audio_probs = audio_result["all_probabilities"]
        
        # Compute attention weights based on confidence
        text_conf = text_result["confidence"]
        audio_conf = audio_result["confidence"]
        
        # Apply softmax to get normalized attention weights
        attention_sum = text_conf + audio_conf
        text_weight = text_conf / attention_sum
        audio_weight = audio_conf / attention_sum
        
        print(f"Attention weights: text={text_weight:.2f}, audio={audio_weight:.2f}")
        
        # Fuse probabilities using attention weights
        fused_probs = {}
        for emotion in self.emotion_labels:
            fused_probs[emotion] = text_weight * text_probs[emotion] + audio_weight * audio_probs[emotion]
        
        # Get the predicted emotion
        predicted_emotion = max(fused_probs, key=fused_probs.get)
        confidence = fused_probs[predicted_emotion]
        
        return {
            "predicted_emotion": predicted_emotion,
            "confidence": confidence,
            "all_probabilities": fused_probs,
            "text_prediction": text_result["predicted_emotion"],
            "text_confidence": text_result["confidence"],
            "audio_prediction": audio_result["predicted_emotion"],
            "audio_confidence": audio_result["confidence"]
        }
    
    def predict(self, text, audio_path):
        """
        Predict emotion from text and audio.
        
        Args:
            text (str): Input text
            audio_path (str): Path to audio file
            
        Returns:
            dict: Prediction result
        """
        # Process text
        text_result = self.process_text(text)
        
        # Process audio
        audio_result = self.process_audio(audio_path)
        
        # Fuse predictions
        fused_result = self.fuse_predictions(text_result, audio_result)
        
        return {
            "text": text_result,
            "audio": audio_result,
            "fusion": fused_result
        }


def create_dummy_audio(emotion="happy"):
    """
    Create a dummy audio file for testing.
    
    Args:
        emotion (str): Emotion to encode in the filename
        
    Returns:
        str: Path to the created audio file
    """
    # Create directory if it doesn't exist
    os.makedirs("dummy_audio", exist_ok=True)
    
    # Create a dummy file
    file_path = f"dummy_audio/{emotion}_test.wav"
    
    # Write a simple text file as a placeholder
    # In a real implementation, this would create an actual audio file
    with open(file_path, "w") as f:
        f.write(f"This is a dummy audio file for {emotion} emotion.")
    
    print(f"Created dummy audio file: {file_path}")
    return file_path


def main():
    """
    Main function to demonstrate the minimal emotion detection system.
    """
    print("Multimodal Emotion Detection - Minimal Example")
    print("=" * 50)
    
    # Initialize detector
    detector = MinimalEmotionDetector()
    
    # Example inputs
    examples = [
        {
            "text": "I'm feeling really happy today!",
            "emotion": "happy"
        },
        {
            "text": "I'm so angry right now, this is frustrating!",
            "emotion": "angry"
        },
        {
            "text": "I'm feeling a bit sad and disappointed.",
            "emotion": "sad"
        },
        {
            "text": "That's surprising, I didn't expect that!",
            "emotion": "surprise"
        },
        {
            "text": "I'm feeling neutral, just a normal day.",
            "emotion": "neutral"
        }
    ]
    
    # Process each example
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print("-" * 50)
        
        # Create dummy audio
        audio_path = create_dummy_audio(example["emotion"])
        
        # Predict emotion
        result = detector.predict(example["text"], audio_path)
        
        # Print results
        print("\nResults:")
        print(f"Text: '{example['text']}'")
        print(f"Text prediction: {result['text']['predicted_emotion']} (Confidence: {result['text']['confidence']:.4f})")
        print(f"Audio prediction: {result['audio']['predicted_emotion']} (Confidence: {result['audio']['confidence']:.4f})")
        print(f"Fused prediction: {result['fusion']['predicted_emotion']} (Confidence: {result['fusion']['confidence']:.4f})")
        
        # Print top 3 emotions for fusion
        top_emotions = sorted(
            result['fusion']['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        print("\nTop 3 emotions (fusion):")
        for emotion, prob in top_emotions:
            print(f"  - {emotion}: {prob:.4f}")
        
        print("-" * 50)
    
    print("\nMinimal example completed successfully!")


if __name__ == "__main__":
    main()
