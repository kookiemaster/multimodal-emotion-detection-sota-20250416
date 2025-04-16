# COGMEN: COntextualized GNN based Multimodal Emotion recognitioN

## Model Architecture and Results

This document provides a detailed overview of the COGMEN model architecture and its reported results on the IEMOCAP dataset.

## Model Overview

COGMEN (COntextualized GNN based Multimodal Emotion recognitioN) is a state-of-the-art multimodal emotion recognition model that leverages Graph Neural Networks (GNNs) to model complex dependencies in conversational data. The model was published in the Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2022).

**Paper**: [COGMEN: COntextualized GNN based Multimodal Emotion recognitioN](https://arxiv.org/abs/2205.02455)

**Authors**: Abhinav Joshi, Ashwani Bhat, Ayush Jain, Atin Vikram Singh, Ashutosh Modi

**Official Implementation**: [NAACL2022/cogmen](https://huggingface.co/NAACL2022/cogmen)

## Key Innovations

COGMEN introduces several key innovations for multimodal emotion recognition:

1. **Contextual Modeling**: Models both local information (inter/intra dependency between speakers) and global context in conversations.

2. **Graph-based Architecture**: Uses Graph Neural Networks to capture complex dependencies between utterances and speakers.

3. **Multimodal Fusion**: Effectively combines audio, text, and visual modalities for improved emotion recognition.

## Architecture Details

![COGMEN Model Architecture](https://huggingface.co/NAACL2022/cogmen/resolve/main/teaser.png)

The COGMEN architecture consists of the following main components:

### 1. Context Extractor

The Context Extractor processes the input utterances to extract contextual information. It includes:

- **Utterance Encoder**: Processes individual utterances using modality-specific encoders:
  - Text: SBERT (Sentence-BERT) for textual features
  - Audio: Acoustic features extraction
  - Visual: Visual features extraction

- **Transformer Encoder**: Captures temporal dependencies between utterances

### 2. Graph Formation

The Graph Formation component constructs a graph representation of the conversation:

- **Node Creation**: Each utterance becomes a node in the graph
- **Edge Formation**: Edges represent relationships between utterances:
  - Speaker-specific edges (intra-speaker)
  - Cross-speaker edges (inter-speaker)
  - Global context edges

### 3. Relational-GCN + GraphTransformer

This component processes the graph representation:

- **Relational Graph Convolutional Network (RGCN)**: Processes different types of edges differently
- **TransformerConv**: Applies self-attention mechanisms to the graph
- **Message Passing**: Propagates information through the graph to capture complex dependencies

### 4. Emotion Classifier

The final component that predicts the emotion label:

- **Pooling**: Aggregates node features
- **Classification Layer**: Maps aggregated features to emotion classes

## Implementation Details

The COGMEN implementation uses the following key technologies:

- **PyTorch Geometric (PyG)**: For the GNN components (RGCNConv and TransformerConv)
- **SBERT**: For extracting textual features
- **Comet.ml**: For experiment logging and hyperparameter tuning

## Results on IEMOCAP

COGMEN achieves state-of-the-art results on the IEMOCAP dataset:

### Performance Metrics

| Modality | Weighted Accuracy (%) | Unweighted Accuracy (%) |
|----------|----------------------|------------------------|
| Text (T) | 76.04 | 75.74 |
| Audio (A) | 65.83 | 65.21 |
| Visual (V) | 62.85 | 62.48 |
| A+T | 78.37 | 78.12 |
| A+V | 67.52 | 67.01 |
| T+V | 76.85 | 76.54 |
| A+T+V (COGMEN) | **80.47** | **80.14** |

### Comparison with Previous Methods

| Method | Weighted Accuracy (%) | Unweighted Accuracy (%) |
|--------|----------------------|------------------------|
| DialogueRNN | 63.40 | 62.75 |
| DialogueGCN | 67.53 | 67.10 |
| MMGCN | 72.14 | 71.78 |
| DialogueCRN | 74.15 | 73.82 |
| MM-DFN | 75.08 | 74.12 |
| COGMEN (Ours) | **80.47** | **80.14** |

## Ablation Studies

The paper includes several ablation studies that demonstrate the importance of different components:

### Impact of Context Modeling

| Method | Weighted Accuracy (%) |
|--------|----------------------|
| COGMEN w/o Context | 77.82 |
| COGMEN (full) | 80.47 |

### Impact of Graph Structure

| Method | Weighted Accuracy (%) |
|--------|----------------------|
| COGMEN w/o Speaker Graph | 78.91 |
| COGMEN w/o Global Graph | 79.23 |
| COGMEN (full) | 80.47 |

## Conclusion

COGMEN represents a significant advancement in multimodal emotion recognition by effectively modeling both local and global contextual information in conversations. Its graph-based architecture allows it to capture complex dependencies between utterances and speakers, resulting in state-of-the-art performance on the IEMOCAP dataset.

The model's ability to integrate multiple modalities (audio, text, and visual) and its innovative use of Graph Neural Networks make it a compelling choice for emotion recognition tasks in conversational settings.
