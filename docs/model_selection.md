# Model Selection: State-of-the-Art Multimodal Emotion Recognition for IEMOCAP

After researching state-of-the-art multimodal emotion recognition models available on Hugging Face, I've identified two promising candidates for our implementation:

## 1. SpeechBrain's emotion-recognition-wav2vec2-IEMOCAP

**Model Link**: [speechbrain/emotion-recognition-wav2vec2-IEMOCAP](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP)

**Architecture**: Fine-tuned wav2vec2 (base) model using SpeechBrain

**Performance**: 
- Accuracy on IEMOCAP test set: 78.7% (Avg: 75.3%)
- Release date: October 19, 2021

**Modalities**: Audio only (not multimodal)

**Advantages**:
- Well-documented implementation with clear usage examples
- Built on the robust wav2vec2 architecture
- Specifically trained on IEMOCAP
- Easy to use with SpeechBrain's inference interface

**Limitations**:
- Only uses audio modality, not multimodal (doesn't incorporate text)
- Not the absolute latest SOTA model

## 2. COGMEN: COntextualized GNN based Multimodal Emotion recognitioN

**Model Link**: [NAACL2022/cogmen](https://huggingface.co/NAACL2022/cogmen)

**Architecture**: Graph Neural Network (GNN) based architecture that models complex dependencies in conversations

**Performance**:
- Claims state-of-the-art results on IEMOCAP and MOSEI datasets
- Published in NAACL 2022

**Modalities**: Multimodal (audio, text, visual)

**Advantages**:
- True multimodal approach incorporating audio, text, and visual features
- Models both local information (inter/intra dependency between speakers) and global context
- More recent publication (2022)
- Official PyTorch implementation available

**Limitations**:
- More complex architecture requiring additional dependencies (PyG, SBERT)
- Less straightforward implementation for inference

## Selected Model: COGMEN

**Rationale for Selection**:

1. **Multimodal Approach**: COGMEN is a true multimodal model that incorporates audio, text, and visual features, which aligns better with our goal of implementing state-of-the-art multimodal emotion recognition.

2. **Performance Claims**: COGMEN claims to achieve state-of-the-art results on IEMOCAP, which is our target dataset.

3. **Recency**: As a more recent publication (2022), COGMEN likely incorporates more advanced techniques compared to the SpeechBrain model (2021).

4. **Architecture Innovation**: The use of Graph Neural Networks to model complex dependencies in conversations represents an innovative approach to emotion recognition.

5. **Official Implementation**: Having the official PyTorch implementation available on Hugging Face makes it feasible to implement and use with our IEMOCAP dataset.

While the SpeechBrain model offers excellent documentation and ease of use, COGMEN's multimodal approach and claimed SOTA performance make it the more appropriate choice for our project. We will implement COGMEN and document its architecture, results, and usage instructions in our repository.
