# COGMEN: Multimodal Emotion Recognition Implementation

This document provides detailed instructions for using our implementation of the COGMEN (COntextualized GNN based Multimodal Emotion recognitioN) model for emotion recognition on the IEMOCAP dataset.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Repository Structure](#repository-structure)
4. [Usage Instructions](#usage-instructions)
5. [IEMOCAP Dataset](#iemocap-dataset)
6. [Model Architecture](#model-architecture)
7. [Results](#results)
8. [Troubleshooting](#troubleshooting)

## Overview

This repository contains an implementation of the COGMEN model for multimodal emotion recognition, as described in the paper [COGMEN: COntextualized GNN based Multimodal Emotion recognitioN](https://arxiv.org/abs/2205.02455) (NAACL 2022). The model uses Graph Neural Networks to model complex dependencies in conversational data and achieves state-of-the-art results on the IEMOCAP dataset.

Our implementation focuses on providing:
1. A simplified version of the COGMEN model architecture
2. Inference code for using the model with IEMOCAP data
3. Example scripts for demonstration
4. Evaluation and visualization tools

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- Sentence Transformers
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/kookiemaster/multimodal-emotion-detection-sota-20250416.git
cd multimodal-emotion-detection-sota-20250416
```

2. Install the required dependencies:
```bash
pip install torch==1.10.0 torchvision==0.11.0
pip install torch-geometric
pip install sentence-transformers
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Prepare the IEMOCAP dataset:
   - Place the `IEMOCAP_Final.csv` file in the `data/processed/iemocap/` directory
   - The CSV file should contain columns for utterance transcriptions, audio file paths, speaker IDs, and emotion labels

## Repository Structure

```
multimodal-emotion-detection-sota-20250416/
├── data/
│   └── processed/
│       └── iemocap/
│           └── IEMOCAP_Final.csv
├── docs/
│   ├── model_selection.md
│   ├── cogmen_architecture_and_results.md
│   └── implementation_and_usage.md
├── results/
│   └── example_run/
│       ├── confusion_matrix.png
│       ├── emotion_distribution.png
│       ├── accuracy_by_emotion.png
│       └── classification_report.csv
├── src/
│   ├── cogmen_inference.py
│   └── example_iemocap.py
└── README.md
```

## Usage Instructions

### Running the Example Script

The example script demonstrates how to use the COGMEN model with IEMOCAP data:

```bash
python src/example_iemocap.py
```

This script will:
1. Load a sample of IEMOCAP data
2. Initialize the COGMEN model
3. Process the data and generate predictions
4. Evaluate the predictions
5. Generate visualizations
6. Compare with reported SOTA results

### Using the COGMEN Inference Module

To use the COGMEN inference module in your own code:

```python
from src.cogmen_inference import COGMENInference

# Initialize COGMEN inference
cogmen = COGMENInference()

# Process IEMOCAP data and get predictions
results_df = cogmen.process_iemocap_csv("path/to/iemocap.csv")

# Evaluate predictions
metrics = cogmen.evaluate_predictions(results_df)
print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
```

### Custom Inference

For more control over the inference process:

```python
from src.cogmen_inference import COGMENInference

# Initialize COGMEN inference
cogmen = COGMENInference()

# Prepare your data
texts = ["I'm feeling happy today", "Why are you so angry?"]
audio_paths = ["path/to/audio1.wav", "path/to/audio2.wav"]
visual_paths = ["path/to/visual1.mp4", "path/to/visual2.mp4"]
speaker_ids = [1, 2]

# Predict emotions
emotion_labels, probabilities = cogmen.predict_emotions(
    texts, audio_paths, visual_paths, speaker_ids
)

print(f"Predicted emotions: {emotion_labels}")
```

## IEMOCAP Dataset

The Interactive Emotional Dyadic Motion Capture (IEMOCAP) dataset is a multimodal dataset containing approximately 12 hours of audiovisual data, including video, speech, motion capture of face, and text transcriptions. The dataset consists of dyadic conversations, where actors perform improvisations or scripted scenarios designed to elicit specific emotions.

Our implementation uses a processed version of IEMOCAP (`IEMOCAP_Final.csv`) that contains:
- Utterance transcriptions
- Paths to audio files
- Speaker IDs
- Emotion labels

The emotion labels in IEMOCAP include:
- Neutral
- Happy
- Sad
- Angry
- Frustrated
- Excited

## Model Architecture

The COGMEN model architecture consists of several key components:

1. **Feature Extraction**:
   - Text features: Extracted using SBERT (Sentence-BERT)
   - Audio features: Extracted from audio files
   - Visual features: Extracted from video files

2. **Context Encoder**:
   - Transformer-based encoder that captures temporal dependencies

3. **Graph Formation**:
   - Constructs a graph representation of the conversation
   - Nodes represent utterances
   - Edges represent relationships between utterances

4. **Graph Neural Network**:
   - RGCN (Relational Graph Convolutional Network)
   - Graph Transformer for self-attention

5. **Emotion Classifier**:
   - Maps graph features to emotion classes

For more details on the architecture, refer to the [cogmen_architecture_and_results.md](docs/cogmen_architecture_and_results.md) document.

## Results

The COGMEN model achieves state-of-the-art results on the IEMOCAP dataset:

| Method | Weighted Accuracy (%) | Unweighted Accuracy (%) |
|--------|----------------------|------------------------|
| DialogueRNN | 63.40 | 62.75 |
| DialogueGCN | 67.53 | 67.10 |
| MMGCN | 72.14 | 71.78 |
| DialogueCRN | 74.15 | 73.82 |
| MM-DFN | 75.08 | 74.12 |
| COGMEN | **80.47** | **80.14** |

Our implementation provides a simplified version of the model for demonstration purposes. The actual performance may vary from the reported results due to implementation differences and data preprocessing.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   - Ensure all required packages are installed
   - Check for compatible versions of PyTorch and PyTorch Geometric

2. **IEMOCAP Dataset Issues**:
   - Verify that the `IEMOCAP_Final.csv` file is in the correct location
   - Check that the CSV file has the required columns

3. **CUDA Out of Memory**:
   - Reduce batch size in the example script
   - Use a smaller sample of the dataset

4. **Model Loading Errors**:
   - Ensure the model architecture matches the expected weights
   - Check for compatibility between PyTorch versions

### Getting Help

If you encounter issues not covered here, please:
1. Check the documentation in the `docs/` directory
2. Refer to the original COGMEN paper and implementation
3. Open an issue on the GitHub repository

## Citation

If you use this implementation in your research, please cite the original COGMEN paper:

```
@inproceedings{joshi-etal-2022-cogmen,
    title = "{COGMEN}: {CO}ntextualized {GNN} based Multimodal Emotion recognitio{N}",
    author = "Joshi, Abhinav  and
      Bhat, Ashwani  and
      Jain, Ayush  and
      Singh, Atin  and
      Modi, Ashutosh",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.306",
    pages = "4148--4164",
}
```
