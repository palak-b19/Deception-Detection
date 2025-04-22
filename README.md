# Deception Detection: Lie Detection in Diplomacy via Machine Learning

## Project Overview
This project focuses on detecting deception in diplomatic communication within the game of Diplomacy using machine learning and deep learning techniques. The research is done on the "It Takes Two to Lie: One to Lie, and One to Listen" dataset (Peskov et al., 2020) to analyze 17,289 annotated messages from 12 Diplomacy games, with approximately 5% labeled as deceptive.

## Codes
Implementation of the EnhancedDeceptionDetector model, baseline models (e.g., Context LSTM + Power, Bagofwords, TF-IDF), and preprocessing scripts.
- `EnhancedDeceptionDetector.py`: Main model integrating Sentence-BERT embeddings, linguistic features, and bidirectional LSTM with self-attention.
- `preprocessing.py`: Scripts for data preprocessing, including categorical mapping and linguistic feature extraction.
- `train.py`: Training loop with hyperparameter tuning and checkpointing.
- `evaluate.py`: Evaluation scripts for computing accuracy, macro F1, and lie F1 scores.

## Dataset
References the 2020 ACL Diplomacy dataset, split into:
- `train.jsonl` (9 games)
- `validation.jsonl` (1 game)
- `test.jsonl` (2 games)
- Dataset: [2020 ACL Diplomacy GitHub Repository]([https://github.com/DenisPeskoff/2020_acl_diplomacy/tree/master/data])

## Required Libraries
- PyTorch
- SpaCy
- NLTK
- VADER
- Sentence-BERT (`paraphrase-MiniLM-L6-v2`)

## Model Details

### EnhancedDeceptionDetector
- Combines Sentence-BERT embeddings (384D), linguistic features (6D), and contextual metadata (e.g., game scores, countries).
- Uses bidirectional LSTM with self-attention and weighted loss to handle class imbalance (5% deceptive messages).
- Achieves 81.75% accuracy, 57.71 macro F1, and 24.60 lie F1 score.

### Baseline Models
- Context LSTM + Power: 55.13 macro F1.
- Context LSTM: 53.7 macro F1.
- Other approaches: Bagofwords, TF-IDF, and LLM Feedback Loop (26.51 lie F1).

### Preprocessing
- Linguistic features: Word count, lexical diversity, self-references, group references, negative emotion, modifiers.
- Categorical mapping for countries, seasons, and years.
- Oversampling and weighted loss to address class imbalance.

## References
- Peskov et al. (2020). *It Takes Two to Lie: One to Lie, and One to Listen*. ACL.
- Mendels et al. (2017). *Hybrid Acoustic-Lexical Deep Learning Approach for Deception Detection*. Interspeech.
- Prome et al. (2024). *Deception Detection Using Machine Learning and Deep Learning Techniques*. Neurocomputing.
- Banerjee et al. (2024). *LLMs are Superior Feedback Providers*. arXiv.
