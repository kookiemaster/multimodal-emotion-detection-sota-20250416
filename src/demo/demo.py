"""
Demo Application for Multimodal Emotion Detection

This module implements a simple demo application for multimodal emotion detection.
"""

import os
import sys
import argparse
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text.text_processor import TextEmotionProcessor
from voice.voice_processor import VoiceEmotionProcessor
from fusion.fusion import MultimodalFusion

class EmotionDetectionDemo:
    """
    Demo application for multimodal emotion detection.
    """
    
    def __init__(self):
        """
        Initialize the demo application.
        """
        self.text_processor = None
        self.voice_processor = None
        self.fusion = None
        self.emotion_colors = {
            "angry": "red",
            "disgust": "purple",
            "fear": "gray",
            "happy": "green",
            "neutral": "blue",
            "sad": "brown",
            "surprise": "orange"
        }
        
        # Set HuggingFace token if available
        if "HF_TOKEN" in os.environ:
            self.hf_token = os.environ["HF_TOKEN"]
            print("HuggingFace token found in environment variables.")
        else:
            self.hf_token = None
            print("No HuggingFace token found. Some models may not be accessible.")
    
    def initialize_models(self):
        """
        Initialize the emotion detection models.
        
        Returns:
            bool: True if models initialized successfully, False otherwise
        """
        print("Initializing text emotion detection model...")
        self.text_processor = TextEmotionProcessor()
        text_success = self.text_processor.load_model()
        
        print("Initializing voice emotion detection model...")
        self.voice_processor = VoiceEmotionProcessor()
        voice_success = self.voice_processor.load_model()
        
        if text_success and voice_success:
            print("Initializing multimodal fusion...")
            self.fusion = MultimodalFusion(
                text_processor=self.text_processor,
                voice_processor=self.voice_processor,
                fusion_method='attention'
            )
            return True
        else:
            print("Failed to initialize models.")
            return False
    
    def process_input(self, text, audio_path):
        """
        Process text and audio input.
        
        Args:
            text (str): Input text
            audio_path (str): Path to audio file
            
        Returns:
            dict: Dictionary containing emotion predictions
        """
        if self.text_processor is None or self.voice_processor is None or self.fusion is None:
            raise ValueError("Models not initialized. Call initialize_models() first.")
        
        print(f"Processing text: '{text}'")
        print(f"Processing audio: {audio_path}")
        
        # Get text emotion prediction
        text_result = self.text_processor.predict_emotion(text)
        
        # Get voice emotion prediction
        voice_result = self.voice_processor.predict_emotion(audio_path)
        
        # Get fused prediction
        fusion_result = self.fusion.fuse_predictions(text_result, voice_result)
        
        return {
            'text': text_result,
            'voice': voice_result,
            'fusion': fusion_result
        }
    
    def visualize_results(self, results):
        """
        Visualize emotion detection results.
        
        Args:
            results (dict): Dictionary containing emotion predictions
        """
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Plot text emotion
        plt.subplot(1, 3, 1)
        self._plot_emotion_probabilities(
            results['text']['all_probabilities'],
            title=f"Text Emotion: {results['text']['predicted_emotion']}",
            color=self.emotion_colors.get(results['text']['predicted_emotion'], 'blue')
        )
        
        # Plot voice emotion
        plt.subplot(1, 3, 2)
        self._plot_emotion_probabilities(
            results['voice']['all_probabilities'],
            title=f"Voice Emotion: {results['voice']['predicted_emotion']}",
            color=self.emotion_colors.get(results['voice']['predicted_emotion'], 'blue')
        )
        
        # Plot fusion emotion
        plt.subplot(1, 3, 3)
        self._plot_emotion_probabilities(
            results['fusion']['all_probabilities'],
            title=f"Fusion Emotion: {results['fusion']['predicted_emotion']}",
            color=self.emotion_colors.get(results['fusion']['predicted_emotion'], 'blue')
        )
        
        plt.tight_layout()
        plt.show()
    
    def _plot_emotion_probabilities(self, probabilities, title, color):
        """
        Plot emotion probabilities.
        
        Args:
            probabilities (dict): Dictionary of emotion probabilities
            title (str): Plot title
            color (str): Bar color
        """
        # Sort emotions by probability
        emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        labels = [e[0] for e in emotions]
        values = [e[1] for e in emotions]
        
        # Plot horizontal bar chart
        bars = plt.barh(labels, values, color=color)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}', va='center'
            )
        
        plt.xlim(0, 1)
        plt.title(title)
        plt.xlabel('Probability')
    
    def create_test_audio(self, emotion, duration=3.0, sample_rate=16000):
        """
        Create a test audio file with a specific emotion.
        
        Args:
            emotion (str): Emotion to simulate
            duration (float): Duration in seconds
            sample_rate (int): Sample rate
            
        Returns:
            str: Path to the created audio file
        """
        print(f"Creating test audio file for emotion: {emotion}")
        
        # Create directory for test audio
        os.makedirs("test_audio", exist_ok=True)
        
        # Define parameters based on emotion
        if emotion == "happy":
            # Higher frequency, more variations
            freq = 440
            variations = 10
        elif emotion == "sad":
            # Lower frequency, fewer variations
            freq = 220
            variations = 3
        elif emotion == "angry":
            # Higher frequency, sharp variations
            freq = 660
            variations = 15
        elif emotion == "fear":
            # Trembling effect
            freq = 330
            variations = 20
        else:
            # Neutral
            freq = 440
            variations = 5
        
        # Create time array
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Create base signal
        signal = 0.5 * np.sin(2 * np.pi * freq * t)
        
        # Add variations based on emotion
        for i in range(1, variations + 1):
            # Add harmonics
            signal += 0.1 / i * np.sin(2 * np.pi * (freq * i) * t)
            
            # Add amplitude modulation
            if emotion in ["fear", "angry"]:
                signal *= 1 + 0.2 * np.sin(2 * np.pi * 5 * t)
        
        # Add noise
        noise_level = 0.1
        if emotion == "angry":
            noise_level = 0.2
        elif emotion == "fear":
            noise_level = 0.15
        
        signal += noise_level * np.random.randn(*signal.shape)
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        # Save as WAV file
        output_path = f"test_audio/{emotion}.wav"
        sf.write(output_path, signal, sample_rate)
        
        print(f"Test audio file created: {output_path}")
        return output_path
    
    def run_interactive_demo(self):
        """
        Run an interactive demo.
        """
        if not self.initialize_models():
            print("Failed to initialize models. Exiting.")
            return
        
        print("\nWelcome to the Multimodal Emotion Detection Demo!")
        print("=" * 50)
        print("This demo allows you to detect emotions from text and voice.")
        print("You can enter text and provide an audio file, or use test audio.")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. Process text and audio")
            print("2. Process text with test audio")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == '1':
                text = input("\nEnter text: ")
                audio_path = input("Enter path to audio file: ")
                
                if not os.path.exists(audio_path):
                    print(f"Audio file not found: {audio_path}")
                    continue
                
                results = self.process_input(text, audio_path)
                self.visualize_results(results)
                
            elif choice == '2':
                text = input("\nEnter text: ")
                
                print("\nSelect emotion for test audio:")
                print("1. Happy")
                print("2. Sad")
                print("3. Angry")
                print("4. Fear")
                print("5. Neutral")
                
                emotion_choice = input("\nEnter your choice (1-5): ")
                emotion_map = {
                    '1': "happy",
                    '2': "sad",
                    '3': "angry",
                    '4': "fear",
                    '5': "neutral"
                }
                
                if emotion_choice in emotion_map:
                    emotion = emotion_map[emotion_choice]
                    audio_path = self.create_test_audio(emotion)
                    
                    results = self.process_input(text, audio_path)
                    self.visualize_results(results)
                else:
                    print("Invalid choice.")
                
            elif choice == '3':
                print("\nExiting demo. Goodbye!")
                break
                
            else:
                print("Invalid choice. Please try again.")


def main():
    """
    Main function to run the demo application.
    """
    parser = argparse.ArgumentParser(description='Multimodal Emotion Detection Demo')
    parser.add_argument('--text', type=str, help='Input text')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--interactive', action='store_true', help='Run interactive demo')
    
    args = parser.parse_args()
    
    demo = EmotionDetectionDemo()
    
    if args.interactive:
        demo.run_interactive_demo()
    elif args.text and args.audio:
        if not demo.initialize_models():
            print("Failed to initialize models. Exiting.")
            return
        
        results = demo.process_input(args.text, args.audio)
        demo.visualize_results(results)
    else:
        print("Please provide both text and audio, or use interactive mode.")
        print("Example: python demo.py --text 'I am happy' --audio happy.wav")
        print("         python demo.py --interactive")


if __name__ == "__main__":
    main()
