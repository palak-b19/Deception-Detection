# Deception Detection Model - Instructions

## Setup & Requirements

1. Install required libraries:
   ```bash
   pip install torch sentence-transformers spacy nltk scikit-learn matplotlib seaborn tqdm
   python -m spacy download en_core_web_sm
   ```

2. Ensure you have the following data files:
   - `/kaggle/input/dataset-deception/train.jsonl`
   - `/kaggle/input/dataset-deception/validation.jsonl`
   - `/kaggle/input/dataset-deception/test.jsonl`

## Running the Model

### Basic Usage

1. Execute the entire script to train and evaluate the model:
   ```bash
   python dataset_deception_self_attention.py
   ```

2. The script will:
   - Preprocess the data
   - Train the model with oversampling of deceptive messages
   - Perform grid search for hyperparameters
   - Generate and save checkpoints
   - Evaluate the model on test data

### Configuration Options

You can modify these key variables at the bottom of the script:

- `use_ling_features`: Set to `True` to use linguistic features, `False` to disable them
- `epochs`: Number of training epochs (default: 15)
- `patience`: Early stopping patience (default: 3)
- `oversample_factor`: Factor for oversampling deceptive messages (default: 2)

### Checkpoint Management

- Checkpoints are saved to the `checkpoints` directory
- `best_checkpoint.pt` contains the model with highest validation macro F1 score
- Individual epoch checkpoints are saved with metrics in their filenames
- To resume training from a checkpoint:
  ```python
  checkpoint = torch.load('checkpoints/best_checkpoint.pt')
  train_and_evaluate(model, train_data, val_data, test_data, checkpoint=checkpoint)
  ```

### Evaluating Saved Models

To evaluate a saved checkpoint:
```python
checkpoint_path = 'checkpoints/best_checkpoint.pt'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
test_loss, test_metrics = evaluate_checkpoint(model, test_data)
```

## Output

The model outputs metrics including:
- Precision, recall, and F1 scores for both truthful and deceptive classes
- Micro and macro-averaged metrics
- Confusion matrix
- Validation and test losses
