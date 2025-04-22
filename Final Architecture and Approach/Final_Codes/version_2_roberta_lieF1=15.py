import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import jsonlines
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm.notebook import tqdm
from collections import deque

print("Starting")

# Dataset class to handle Diplomacy game data with conversational context
class DiplomacyDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128, context_window=5):
        self.data = []
        self.context_window = context_window
        print(f"Loading dataset from {file_path}...")
        with jsonlines.open(file_path, 'r') as reader:
            dialogs_list = list(reader)
            total_messages = sum(len(dialogs['messages']) for dialogs in dialogs_list)
            
            for dialogs in tqdm(dialogs_list, desc="Processing dialogs", position=0):
                message_data = list(zip(
                    dialogs['messages'],
                    dialogs['sender_labels'],
                    dialogs['game_score'],
                    dialogs['game_score_delta'],
                    dialogs['relative_message_index']
                ))
                # Track conversation history per sender-receiver pair
                history = deque(maxlen=context_window)
                for msg, sender_label, game_score, game_score_delta, rel_idx in message_data:
                    try:
                        game_state = [float(game_score), float(game_score_delta)]
                    except (ValueError, TypeError):
                        continue

                    if sender_label is True:
                        label = 0  # Truthful
                    elif sender_label is False:
                        label = 1  # Deceptive
                    else:
                        continue

                    # Add current message to history
                    history.append(msg)
                    # Create context (previous messages, padded if needed)
                    context = list(history)[:-1]  # Exclude current message
                    context = [''] * (context_window - 1 - len(context)) + context  # Pad with empty strings
                    context_text = " ".join(context)  # Concatenate for simplicity

                    self.data.append({
                        'text': msg,
                        'context': context_text,
                        'label': label,
                        'game_state': game_state
                    })

        self.tokenizer = tokenizer
        self.max_length = max_length
        # Compute weights for sampling
        self.weights = [1.0 if item['label'] == 1 else 0.05 for item in self.data]  # Higher weight for deceptive

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        context = item['context']
        label = item['label']
        game_state = item['game_state']

        # Combine context and current message
        combined_text = context + " [SEP] " + text if context.strip() else text
        encoding = self.tokenizer(
            combined_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Don't move to device here - do it in training loop
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Keep as numpy arrays or Python lists initially
        game_state = torch.tensor(game_state, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'game_state': game_state,
            'label': label
        }

# Model architecture
class DeceptionDetectionModel(nn.Module):
    def __init__(self, roberta_model, game_state_dim, hidden_dim=800, dropout=0.3):
        super(DeceptionDetectionModel, self).__init__()
        self.roberta = roberta_model
        self.game_state_encoder = nn.Sequential(
            nn.Linear(game_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Fix the transformer implementation
        self.fusion_layer = nn.Sequential(
            nn.Linear(768 + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, game_state):
        text_embeds = self.roberta(input_ids=input_ids, attention_mask=attention_mask).pooler_output  # (batch_size, 768)
        game_embeds = self.game_state_encoder(game_state)  # (batch_size, 32)
        fused = torch.cat((text_embeds, game_embeds), dim=1)  # (batch_size, 800)
        fused = self.fusion_layer(fused)  # (batch_size, hidden_dim//2)
        logits = self.classifier(fused)
        probs = torch.sigmoid(logits)
        return probs

# Focal loss with adjusted parameters
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.75):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        bce_loss = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = self.alpha * focal_term * bce_loss
        return loss.mean()

# Training function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training batches", position=1, leave=True):
        # Move tensors to device here
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        game_state = batch['game_state'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, game_state)
        loss = criterion(outputs.squeeze(1), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    preds, labels_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation batches", position=1, leave=True):
            # Move tensors to device here
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            game_state = batch['game_state'].to(device)
            label = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, game_state)
            preds.extend(outputs.squeeze(1).cpu().numpy())
            labels_list.extend(label.cpu().numpy())
    preds = np.array(preds) > 0.5
    labels_array = np.array(labels_list)
    macro_f1 = f1_score(labels_array, preds, average='macro')
    lie_f1 = f1_score(labels_array, preds, pos_label=1, zero_division=0)  # Handle zero cases
    accuracy = accuracy_score(labels_array, preds)
    return macro_f1, lie_f1, accuracy

# Main function
def main():
    # Hyperparameters
    batch_size = 32
    num_epochs = 15  # Extended to match paper
    learning_rate = 2e-5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaModel.from_pretrained('roberta-base')
    # Only move the model to device after defining it
    roberta_model = roberta_model.to(device)
    
    game_state_dim = 2  # game_score, score_delta
    model = DeceptionDetectionModel(roberta_model, game_state_dim).to(device)

    # Load datasets with correct paths for Kaggle
    train_dataset = DiplomacyDataset('/kaggle/input/dataset-deception/train.jsonl', tokenizer)
    val_dataset = DiplomacyDataset('/kaggle/input/dataset-deception/validation.jsonl', tokenizer)
    test_dataset = DiplomacyDataset('/kaggle/input/dataset-deception/test.jsonl', tokenizer)

    # Weighted sampler for class imbalance
    weights = torch.tensor(train_dataset.weights, dtype=torch.float32)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,  # This helps speed up data transfer to GPU
        num_workers=2     # Use multiple workers for data loading
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2
    )

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = FocalLoss()
    
    # Use a scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # Training loop
    best_lie_f1 = 0
    patience = 10
    patience_counter = 0
    
    # Use specific path for saving model checkpoint on Kaggle
    model_save_path = '/kaggle/working/best_model.pt'
    
    for epoch in tqdm(range(num_epochs), desc="Epochs", position=0):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        macro_f1, lie_f1, accuracy = evaluate(model, val_loader, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Macro F1: {macro_f1:.4f}, Lie F1: {lie_f1:.4f}, Accuracy: {accuracy:.4f}')
        
        # Update the learning rate based on validation performance
        scheduler.step(lie_f1)

        # Early stopping based on lie F1
        if lie_f1 > best_lie_f1:
            best_lie_f1 = lie_f1
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with Lie F1: {lie_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(model_save_path))
    print("Running final test evaluation...")
    macro_f1, lie_f1, accuracy = evaluate(model, test_loader, device)
    print(f'Test Macro F1: {macro_f1:.4f}, Lie F1: {lie_f1:.4f}, Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()



"""
Starting
Using device: cuda
tokenizer_config.json: 100%
 25.0/25.0 [00:00<00:00, 2.40kB/s]
vocab.json: 100%
 899k/899k [00:00<00:00, 3.69MB/s]
merges.txt: 100%
 456k/456k [00:00<00:00, 23.4MB/s]
tokenizer.json: 100%
 1.36M/1.36M [00:00<00:00, 4.17MB/s]
config.json: 100%
 481/481 [00:00<00:00, 63.6kB/s]
Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
model.safetensors: 100%
 499M/499M [00:01<00:00, 369MB/s]
Some weights of RobertaModel were not initialized from the model checkpoint at roberta-base and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Loading dataset from /kaggle/input/dataset-deception/train.jsonl...
Processing dialogs: 100%
 189/189 [00:00<00:00, 4644.50it/s]
Loading dataset from /kaggle/input/dataset-deception/validation.jsonl...
Processing dialogs: 100%
 21/21 [00:00<00:00, 1590.19it/s]
Loading dataset from /kaggle/input/dataset-deception/test.jsonl...
Processing dialogs: 100%
 42/42 [00:00<00:00, 2620.62it/s]
Epochs:  73%
 11/15 [33:08<11:03, 165.78s/it]
Training batches: 100%
 411/411 [02:39<00:00,  3.08it/s]
Evaluation batches: 100%
 45/45 [00:04<00:00,  9.29it/s]
Epoch 1/15, Loss: 0.1798, Macro F1: 0.5026, Lie F1: 0.0945, Accuracy: 0.8376
New best model saved with Lie F1: 0.0945
Training batches: 100%
 411/411 [02:40<00:00,  3.09it/s]
Evaluation batches: 100%
 45/45 [00:04<00:00,  9.29it/s]
Epoch 2/15, Loss: 0.1657, Macro F1: 0.4803, Lie F1: 0.1089, Accuracy: 0.7458
New best model saved with Lie F1: 0.1089
Training batches: 100%
 411/411 [02:40<00:00,  3.10it/s]
Evaluation batches: 100%
 45/45 [00:04<00:00,  9.31it/s]
Epoch 3/15, Loss: 0.0876, Macro F1: 0.5196, Lie F1: 0.0863, Accuracy: 0.9103
Training batches: 100%
 411/411 [02:40<00:00,  3.10it/s]
Evaluation batches: 100%
 45/45 [00:04<00:00,  9.31it/s]
Epoch 4/15, Loss: 0.0560, Macro F1: 0.5191, Lie F1: 0.0682, Accuracy: 0.9421
Training batches: 100%
 411/411 [02:40<00:00,  3.09it/s]
Evaluation batches: 100%
 45/45 [00:04<00:00,  9.28it/s]
Epoch 5/15, Loss: 0.0387, Macro F1: 0.5332, Lie F1: 0.1043, Accuracy: 0.9273
Training batches: 100%
 411/411 [02:40<00:00,  3.10it/s]
Evaluation batches: 100%
 45/45 [00:04<00:00,  9.33it/s]
Epoch 6/15, Loss: 0.0375, Macro F1: 0.5364, Lie F1: 0.1042, Accuracy: 0.9393
Training batches: 100%
 411/411 [02:40<00:00,  3.09it/s]
Evaluation batches: 100%
 45/45 [00:04<00:00,  9.24it/s]
Epoch 7/15, Loss: 0.0268, Macro F1: 0.5111, Lie F1: 0.0709, Accuracy: 0.9075
Training batches: 100%
 411/411 [02:40<00:00,  3.08it/s]
Evaluation batches: 100%
 45/45 [00:04<00:00,  9.25it/s]
Epoch 8/15, Loss: 0.0160, Macro F1: 0.5236, Lie F1: 0.0862, Accuracy: 0.9251
Training batches: 100%
 411/411 [02:40<00:00,  3.09it/s]
Evaluation batches: 100%
 45/45 [00:04<00:00,  9.29it/s]
Epoch 9/15, Loss: 0.0095, Macro F1: 0.5133, Lie F1: 0.0526, Accuracy: 0.9492
Training batches: 100%
 411/411 [02:40<00:00,  3.09it/s]
Evaluation batches: 100%
 45/45 [00:04<00:00,  9.30it/s]
Epoch 10/15, Loss: 0.0082, Macro F1: 0.5107, Lie F1: 0.0494, Accuracy: 0.9456
Training batches: 100%
 411/411 [02:40<00:00,  3.09it/s]
Evaluation batches: 100%
 45/45 [00:04<00:00,  9.30it/s]
Epoch 11/15, Loss: 0.0045, Macro F1: 0.5044, Lie F1: 0.0312, Accuracy: 0.9562
Training batches: 100%
 411/411 [02:40<00:00,  3.09it/s]
Evaluation batches: 100%
 45/45 [00:04<00:00,  9.28it/s]
Epoch 12/15, Loss: 0.0039, Macro F1: 0.5143, Lie F1: 0.0541, Accuracy: 0.9506
Early stopping triggered after {epoch+1} epochs
/tmp/ipykernel_31/3473975214.py:274: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(model_save_path))
Running final test evaluation...
Evaluation batches: 100%
 86/86 [00:09<00:00,  9.28it/s]
Test Macro F1: 0.4885, Lie F1: 0.1595, Accuracy: 0.7001
"""
