# Multimodal Emotion Detection SOTA Replication

This repository contains implementations of state-of-the-art multimodal emotion detection models, specifically focusing on replicating the results of:

1. **MemoCMT**: A cross-modal transformer-based approach (2025)
2. **SDT**: A self-distillation transformer approach (2024)

## Project Structure

```
multimodal-emotion-detection-sota-20250416/
├── data/                      # Dataset directory
│   ├── iemocap/               # IEMOCAP dataset
│   │   ├── raw/               # Raw data
│   │   └── processed/         # Processed data
│   ├── meld/                  # MELD dataset
│   │   ├── raw/               # Raw data
│   │   └── processed/         # Processed data
│   └── esd/                   # ESD dataset
│       ├── raw/               # Raw data
│       └── processed/         # Processed data
├── docs/                      # Documentation
├── notebooks/                 # Jupyter notebooks for analysis
├── results/                   # Results and visualizations
│   ├── memocmt/               # Results for MemoCMT
│   └── sdt/                   # Results for SDT
├── scripts/                   # Utility scripts
├── src/                       # Source code
│   ├── memocmt/               # MemoCMT implementation
│   │   ├── audio/             # Audio processing
│   │   ├── text/              # Text processing
│   │   ├── fusion/            # Fusion mechanisms
│   │   ├── models/            # Model definitions
│   │   └── utils/             # Utilities
│   └── sdt/                   # SDT implementation
│       ├── models/            # Model definitions
│       └── utils/             # Utilities
└── tests/                     # Tests
```

## Selected Models

### MemoCMT

MemoCMT is a state-of-the-art multimodal emotion recognition system that incorporates a cross-modal transformer (CMT) to effectively capture both local and global contexts in speech signals and text transcripts.

- **Paper**: "MemoCMT: multimodal emotion recognition using cross-modal transformer-based feature fusion"
- **Published**: February 14, 2025 in Scientific Reports
- **Original Repository**: https://github.com/tpnam0901/MemoCMT/
- **Performance**:
  - IEMOCAP: 81.33% unweighted accuracy, 81.85% weighted accuracy
  - ESD: 91.93% unweighted accuracy, 91.84% weighted accuracy

### SDT (Self-distillation Transformer)

SDT is a transformer-based model with self-distillation for multimodal emotion recognition in conversations.

- **Paper**: "A Transformer-Based Model With Self-Distillation for Multimodal Emotion Recognition in Conversations"
- **Published**: 2024 in IEEE Transactions on Multimedia
- **Original Repository**: https://github.com/butterfliesss/SDT
- **Datasets**: IEMOCAP and MELD

## Datasets

### IEMOCAP

The Interactive Emotional Dyadic Motion Capture Database (IEMOCAP) consists of 151 videos of recorded dialogues, with 2 speakers per session for a total of 302 videos.

### MELD

The Multimodal EmotionLines Dataset (MELD) contains more than 1400 dialogue instances with audio, visual, and text modalities.

### ESD

The Emotional Speech Database (ESD) is used alongside IEMOCAP in the MemoCMT paper.

## Implementation Strategy

We are implementing these models in the following order:

1. MemoCMT on IEMOCAP
2. MemoCMT on ESD
3. SDT on IEMOCAP
4. SDT on MELD

## Research Notes

For detailed research notes on the selected models and datasets, see [research_notes.md](research_notes.md).

## Todo List

For a comprehensive list of tasks for this project, see [todo.md](todo.md).

## Selected Models

For detailed information about the selected models and rationale, see [selected_models.md](selected_models.md).

## License

This project is for research purposes only. All implementations are based on the original papers and repositories, with appropriate citations.
