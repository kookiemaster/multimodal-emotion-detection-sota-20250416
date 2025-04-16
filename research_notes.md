# State-of-the-Art Multimodal Emotion Recognition Research Notes

## Benchmark Datasets

### IEMOCAP
- Interactive Emotional Dyadic Motion Capture Database
- 151 videos of recorded dialogues, with 2 speakers per session (302 videos total)
- Widely used benchmark for multimodal emotion recognition
- Contains audio, video, and text modalities

### MELD
- Multimodal EmotionLines Dataset
- Contains more than 1400 dialogue instances
- Encompasses audio, visual, and text modalities
- Multi-party conversations (more than 2 speakers)

### ESD
- Emotional Speech Database
- Used alongside IEMOCAP in several SOTA papers

## SOTA Models for Replication

### 1. MemoCMT
- **Paper**: "MemoCMT: multimodal emotion recognition using cross-modal transformer-based feature fusion"
- **Published**: February 14, 2025 in Scientific Reports
- **GitHub**: https://github.com/tpnam0901/MemoCMT/
- **Architecture**:
  - Uses HuBERT for audio feature extraction
  - Uses BERT for text analysis
  - Cross-modal transformer (CMT) for feature fusion
- **Performance**:
  - IEMOCAP: 81.33% unweighted accuracy (UW-Acc), 81.85% weighted accuracy (W-Acc)
  - ESD: 91.93% unweighted accuracy (UW-Acc), 91.84% weighted accuracy (W-Acc)
- **Implementation Details**:
  - Python 3.8
  - PyTorch 2.0.1
  - CUDA 11.8
  - Detailed instructions available in GitHub repository

### 2. SDT (Self-distillation Transformer)
- **Paper**: "A Transformer-Based Model With Self-Distillation for Multimodal Emotion Recognition in Conversations"
- **Published**: 2024 in IEEE Transactions on Multimedia
- **GitHub**: https://github.com/butterfliesss/SDT
- **Architecture**:
  - Transformer-based model with self-distillation
  - Focuses on intra- and inter-modal interactions
  - Multimodal fusion approach
- **Datasets**:
  - Evaluated on IEMOCAP and MELD
- **Implementation Details**:
  - Python implementation
  - Shell scripts for running on IEMOCAP and MELD datasets
  - Requirements file available

### 3. MDAT (Multimodal Dual Attention Transformer)
- **Paper**: "Enhancing Cross-Language Multimodal Emotion Recognition With Dual Attention Transformers"
- **Published**: October 28, 2024
- **Architecture**:
  - Dual attention mechanism
  - Designed for cross-language emotion recognition
  - Transformer-based architecture

### 4. MM-NodeFormer
- **Performance**:
  - IEMOCAP: 74.24% accuracy
  - MELD: 67.86% accuracy
- **Note**: Mentioned in search results, but less information available compared to other models

## Selection Criteria for Replication

When selecting which models to replicate, we should consider:

1. **Availability of code**: Models with public GitHub repositories are easier to replicate
2. **Documentation quality**: Well-documented repositories make replication more feasible
3. **Performance**: Higher-performing models are more interesting to replicate
4. **Recency**: More recent models represent the current SOTA
5. **Dataset availability**: Models using publicly available datasets are easier to replicate

## Preliminary Selection

Based on the research, the most promising models for replication are:

1. **MemoCMT**: Has detailed GitHub repository, excellent performance, recent publication, and clear documentation
2. **SDT**: Has GitHub repository, evaluated on both IEMOCAP and MELD, and includes scripts for both datasets

These models represent the current SOTA in multimodal emotion recognition and have sufficient implementation details available for replication.
