"""
Main script for voice emotion detection component

This script demonstrates the usage of the voice emotion detection component.
"""

import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voice.voice_processor import VoiceEmotionProcessor
from voice.voice_data import prepare_example_data

def main():
    """
    Main function to demonstrate voice emotion detection.
    """
    print("Initializing Voice Emotion Detection Component...")
    
    # Initialize voice emotion processor
    processor = VoiceEmotionProcessor()
    
    # Load model
    success = processor.load_model()
    if not success:
        print("Failed to load model. Using a simpler approach for demonstration.")
    
    # Prepare example data
    audio_paths, _ = prepare_example_data()
    
    # Process each audio file and predict emotion
    print("\nProcessing example audio files:")
    print("-" * 50)
    
    for i, audio_path in enumerate(audio_paths):
        print(f"Example {i+1}: \"{audio_path}\"")
        
        try:
            # Predict emotion
            result = processor.predict_emotion(audio_path)
            
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
            print(f"Error processing audio: {e}")
        
        print("-" * 50)
    
    print("Voice emotion detection demonstration completed.")

def create_test_audio():
    """
    Create a test audio file with different emotional segments.
    """
    print("Creating test audio file...")
    
    # Define parameters
    sample_rate = 16000
    duration = 5  # seconds
    
    # Create a simple sine wave with varying frequency to simulate different emotions
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create a signal with varying frequency
    signal = np.zeros_like(t)
    
    # First segment: low frequency (sad)
    mask1 = t < 1.5
    signal[mask1] = 0.5 * np.sin(2 * np.pi * 220 * t[mask1])
    
    # Second segment: medium frequency (neutral)
    mask2 = (t >= 1.5) & (t < 3.0)
    signal[mask2] = 0.5 * np.sin(2 * np.pi * 440 * t[mask2])
    
    # Third segment: high frequency (happy)
    mask3 = t >= 3.0
    signal[mask3] = 0.5 * np.sin(2 * np.pi * 880 * t[mask3])
    
    # Add some noise
    signal += 0.1 * np.random.randn(*signal.shape)
    
    # Save as WAV file
    output_path = "test_audio.wav"
    sf.write(output_path, signal, sample_rate)
    
    print(f"Test audio file created: {output_path}")
    return output_path

if __name__ == "__main__":
    # Create a test audio file
    test_audio_path = create_test_audio()
    
    # Run the main demonstration
    main()
    
    # Process the test audio file
    print("\nProcessing test audio file with varying emotional content:")
    print("-" * 50)
    
    processor = VoiceEmotionProcessor()
    processor.load_model()
    
    try:
        result = processor.predict_emotion(test_audio_path)
        print(f"Predicted emotion: {result['predicted_emotion']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        print("All probabilities:")
        for emotion, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {emotion}: {prob:.4f}")
    except Exception as e:
        print(f"Error processing test audio: {e}")
