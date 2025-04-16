# Selected SOTA Models for Replication

After thorough research of state-of-the-art multimodal emotion recognition models, we have selected the following models for replication:

## Primary Model: MemoCMT

### Selection Rationale
1. **Recent Publication**: Published in February 2025 in Scientific Reports, making it one of the most recent SOTA models
2. **Excellent Performance**: Achieved 81.33% unweighted accuracy on IEMOCAP and 91.93% on ESD
3. **Complete Implementation**: GitHub repository available with detailed instructions
4. **Well-Documented Architecture**: Clear description of model components (HuBERT, BERT, CMT)
5. **Reproducibility Focus**: Authors explicitly mention reproducibility in their paper

### Implementation Details
- **Repository**: https://github.com/tpnam0901/MemoCMT/
- **Architecture**: Cross-modal transformer using HuBERT for audio and BERT for text
- **Fusion Method**: Min aggregation (best performing variant)
- **Datasets**: IEMOCAP and ESD
- **Metrics to Replicate**: Unweighted accuracy (UW-Acc) and weighted accuracy (W-Acc)

### Target Performance
- IEMOCAP: 81.33% UW-Acc, 81.85% W-Acc
- ESD: 91.93% UW-Acc, 91.84% W-Acc

## Secondary Model: SDT (Self-distillation Transformer)

### Selection Rationale
1. **Complementary Approach**: Uses self-distillation, providing a different perspective from MemoCMT
2. **Different Datasets**: Evaluated on both IEMOCAP and MELD (unlike MemoCMT which uses ESD)
3. **Available Implementation**: GitHub repository with execution scripts
4. **Published in Reputable Venue**: IEEE Transactions on Multimedia (2024)
5. **Focus on Conversations**: Specifically designed for conversational emotion recognition

### Implementation Details
- **Repository**: https://github.com/butterfliesss/SDT
- **Architecture**: Transformer with self-distillation for intra- and inter-modal interactions
- **Datasets**: IEMOCAP and MELD
- **Execution**: Separate scripts for IEMOCAP and MELD datasets

## Implementation Strategy

We will implement these models in the following order:

1. **MemoCMT on IEMOCAP**: As our primary focus, we'll first replicate MemoCMT on the IEMOCAP dataset
2. **MemoCMT on ESD**: Next, we'll extend to the ESD dataset to complete MemoCMT replication
3. **SDT on IEMOCAP**: This will allow direct comparison with MemoCMT on the same dataset
4. **SDT on MELD**: Finally, we'll implement SDT on MELD to cover a different dataset

This approach allows us to:
- Focus on one model architecture at a time
- Compare models on the same dataset (IEMOCAP)
- Cover multiple datasets (IEMOCAP, ESD, MELD)
- Replicate different fusion approaches (cross-modal transformer vs. self-distillation)

## Evaluation Plan

For each model and dataset combination, we will:
1. Train using the exact hyperparameters from the original papers
2. Evaluate using the same metrics (UW-Acc and W-Acc)
3. Compare our results with the published results
4. Analyze any discrepancies
5. Visualize performance and attention mechanisms

This comprehensive approach will ensure thorough replication of state-of-the-art multimodal emotion recognition models as requested.
