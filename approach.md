# Selected Approach for Multimodal Emotion Detection

Based on our research of state-of-the-art methods for multimodal emotion detection combining voice and text modalities, we have selected the following approach:

## Overall Architecture

We will implement a hybrid architecture inspired by the MIST framework but focused specifically on text and voice modalities. Our approach will use:

1. **Text Modality**: DeBERTa-based model for text emotion recognition
2. **Voice Modality**: Semi-CNN architecture for speech emotion recognition
3. **Fusion Strategy**: Late fusion with attention mechanism

## Text Emotion Recognition Component

For the text modality, we will use a DeBERTa-based model which has shown superior performance in capturing contextual information and semantic relationships in text:

- **Base Model**: DeBERTa-v3-small (to accommodate disk space constraints)
- **Fine-tuning**: The model will be fine-tuned on emotion classification tasks
- **Preprocessing**: Standard NLP preprocessing including tokenization, cleaning, and normalization
- **Output**: Emotion probabilities across standard emotion categories (happy, sad, angry, neutral, etc.)

## Voice Emotion Recognition Component

For the voice modality, we will implement a Semi-CNN architecture which has shown good performance in speech emotion recognition tasks:

- **Feature Extraction**: Log-mel spectrograms from audio signals
- **Architecture**: Time Distributed Convolutional Neural Network with LSTM layers
- **Processing**: Rolling window approach for processing spectrograms
- **Output**: Emotion probabilities across standard emotion categories

## Multimodal Fusion Strategy

We will implement a late fusion strategy with an attention mechanism:

- **Late Fusion**: Both modalities will be processed independently, and their outputs will be combined at the decision level
- **Attention Mechanism**: To dynamically weight the importance of each modality based on confidence scores
- **Weighted Averaging**: Final prediction will be a weighted average of individual modality predictions

## Implementation Considerations

Given the disk space constraints encountered during environment setup, we will:

1. Use smaller model variants where possible
2. Implement modular architecture to allow independent testing of components
3. Use efficient data loading techniques to minimize memory usage
4. Leverage pre-trained models with minimal fine-tuning when appropriate

## Datasets

For training and evaluation, we will use publicly available datasets:

1. **Text**: Stream-of-consciousness dataset or similar text emotion datasets
2. **Voice**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

## Evaluation Metrics

We will evaluate our model using:

- Accuracy
- F1-score (weighted and per-class)
- Confusion matrix
- Cross-modal evaluation to assess each modality's contribution

This approach balances state-of-the-art performance with practical implementation considerations, focusing on the voice and text modalities as specified in the project requirements.
