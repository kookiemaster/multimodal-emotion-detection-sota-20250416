# Dataset Access and Preparation Guide

This document provides instructions for accessing and preparing the datasets required for replicating the state-of-the-art multimodal emotion recognition models.

## IEMOCAP Dataset

The Interactive Emotional Dyadic Motion Capture (IEMOCAP) database is a private dataset that requires an access request.

### Access Instructions
1. Visit the official IEMOCAP website: https://sail.usc.edu/iemocap/iemocap_release.htm
2. Fill out the electronic release form on the website
3. Contact Anfeng Xu (anfengxu@usc.edu) if you have any questions
4. Wait for approval and download instructions

### Dataset Structure
- 151 videos of recorded dialogues
- 2 speakers per session (total of 302 videos)
- Contains audio, video, and text modalities
- Approximately 12 hours of data

### Preprocessing Steps
Once obtained, the dataset should be processed according to the MemoCMT paper specifications:
```bash
cd scripts && python preprocess.py -ds IEMOCAP --data_root ./data/IEMOCAP_full_release
```

## MELD Dataset

The Multimodal EmotionLines Dataset (MELD) is publicly available.

### Access Instructions
1. Download the raw data from: http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
2. Alternative download link: https://affective-meld.github.io/
3. The dataset is also available on Kaggle: https://www.kaggle.com/datasets/zaber666/meld-dataset

### Dataset Structure
- More than 1400 dialogue instances
- Audio, visual, and text modalities
- Multi-party conversations (more than 2 speakers)
- Data stored in .mp4 format

### Preprocessing Steps
For SDT model replication, use the provided script:
```bash
cd scripts && python preprocess_meld.py --data_root ./data/MELD.Raw
```

## ESD Dataset

The Emotional Speech Database (ESD) is publicly available.

### Access Instructions
1. Download from GitHub: https://github.com/HLTSingapore/Emotional-Speech-Data
2. Direct Google Drive link available on the GitHub page
3. Alternative download: https://hltsingapore.github.io/ESD/download.html
4. Also available on Kaggle: https://www.kaggle.com/datasets/nguyenthanhlim/emotional-speech-dataset-esd

### Dataset Structure
- 350 parallel utterances
- 10 native English and 10 native Chinese speakers
- 5 emotion categories (neutral, happy, angry, sad, surprise)
- Used in the MemoCMT paper alongside IEMOCAP

### Preprocessing Steps
For MemoCMT model replication, use the provided script:
```bash
cd scripts && python preprocess.py -ds ESD --data_root ./data/ESD
```

## Data Privacy and Usage

When working with these datasets, especially IEMOCAP which requires a formal request, please adhere to the following guidelines:

1. Use the data solely for research purposes as specified in the access agreements
2. Do not redistribute the datasets without permission
3. Properly cite the original dataset papers in any publications
4. Follow the specific terms of use provided by each dataset's creators

## Dataset Citations

### IEMOCAP
```
Busso, C., Bulut, M., Lee, C.C., Kazemzadeh, A., Mower, E., Kim, S., Chang, J.N., Lee, S. and Narayanan, S.S., 2008. IEMOCAP: Interactive emotional dyadic motion capture database. Language resources and evaluation, 42(4), pp.335-359.
```

### MELD
```
Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., & Mihalcea, R. (2019). MELD: A multimodal multi-party dataset for emotion recognition in conversations. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (pp. 527-536).
```

### ESD
```
Zhou, K., Sisman, B., & Li, H. (2022). Emotional voice conversion: Theory, databases and ESD. Speech Communication, 137, 1-18.
```

## Next Steps

After downloading and preparing the datasets:

1. Place the raw data in the corresponding `data/{dataset}/raw` directories
2. Run the preprocessing scripts to prepare the data for model training
3. Verify the processed data matches the format expected by the models
4. Proceed with model implementation and training
