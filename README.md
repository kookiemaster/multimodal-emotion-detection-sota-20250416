# Multimodal Emotion Detection

A state-of-the-art multimodal emotion detection system that combines text and voice modalities for improved emotion recognition.

## Overview

This project implements a multimodal emotion detection system that leverages both text and voice inputs to recognize emotions. By combining these modalities, the system achieves more robust and accurate emotion recognition compared to single-modality approaches.

The system uses:
- **Text Modality**: DeBERTa-based model for text emotion recognition
- **Voice Modality**: Semi-CNN architecture for speech emotion recognition
- **Fusion Strategy**: Late fusion with attention mechanism

## Features

- Text emotion detection using DeBERTa
- Voice emotion detection using Semi-CNN
- Multimodal fusion with attention mechanism
- Comprehensive evaluation metrics
- Interactive demo application
- Support for both individual and combined modality analysis

## Project Structure

```
multimodal-emotion-detection/
├── data/                      # Data directory
│   ├── raw/                   # Raw data
│   └── processed/             # Processed data
├── docs/                      # Documentation
├── notebooks/                 # Jupyter notebooks
├── src/                       # Source code
│   ├── data/                  # Data processing utilities
│   ├── demo/                  # Demo application
│   ├── evaluation/            # Evaluation metrics
│   ├── fusion/                # Multimodal fusion
│   ├── models/                # Model definitions and training
│   ├── text/                  # Text emotion detection
│   ├── utils/                 # Utility functions
│   └── voice/                 # Voice emotion detection
├── tests/                     # Tests
├── .gitignore                 # Git ignore file
├── README.md                  # Project README
├── approach.md                # Approach documentation
└── todo.md                    # Todo list
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kookiemaster/multimodal-emotion-detection-sota-20250416.git
cd multimodal-emotion-detection-sota-20250416
```

2. Install dependencies:
```bash
pip install torch transformers librosa numpy pandas scikit-learn matplotlib
```

3. Set up environment variables (optional):
```bash
export HF_TOKEN=your_huggingface_token
```

## Usage

### Text Emotion Detection

```python
from src.text.text_processor import TextEmotionProcessor

# Initialize processor
processor = TextEmotionProcessor()
processor.load_model()

# Predict emotion from text
text = "I'm feeling really happy today!"
result = processor.predict_emotion(text)
print(f"Predicted emotion: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.4f}")
```

### Voice Emotion Detection

```python
from src.voice.voice_processor import VoiceEmotionProcessor

# Initialize processor
processor = VoiceEmotionProcessor()
processor.load_model()

# Predict emotion from audio
audio_path = "path/to/audio.wav"
result = processor.predict_emotion(audio_path)
print(f"Predicted emotion: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.4f}")
```

### Multimodal Fusion

```python
from src.text.text_processor import TextEmotionProcessor
from src.voice.voice_processor import VoiceEmotionProcessor
from src.fusion.fusion import MultimodalFusion

# Initialize processors
text_processor = TextEmotionProcessor()
voice_processor = VoiceEmotionProcessor()
text_processor.load_model()
voice_processor.load_model()

# Initialize fusion
fusion = MultimodalFusion(text_processor, voice_processor, fusion_method='attention')

# Predict emotion from text and audio
text = "I'm feeling really happy today!"
audio_path = "path/to/audio.wav"
result = fusion.predict_emotion(text, audio_path)
print(f"Predicted emotion: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.4f}")
```

### Demo Application

Run the interactive demo:

```bash
python src/demo/demo.py --interactive
```

Or process specific inputs:

```bash
python src/demo/demo.py --text "I'm feeling really happy today!" --audio path/to/audio.wav
```

## Approach

This project implements a hybrid architecture inspired by the MIST framework but focused specifically on text and voice modalities:

1. **Text Modality**: Uses a DeBERTa-based model for text emotion recognition
2. **Voice Modality**: Implements a Semi-CNN architecture for speech emotion recognition
3. **Fusion Strategy**: Uses late fusion with attention mechanism to dynamically weight the importance of each modality

For more details, see [approach.md](approach.md).

## Evaluation

The system is evaluated using various metrics:

- Accuracy, precision, recall, F1 score
- Confusion matrix
- ROC curves and AUC
- Precision-recall curves
- Modality contribution analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The MIST framework for multimodal emotion recognition
- HuggingFace for providing pre-trained models
- Various open-source libraries used in this project
