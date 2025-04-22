
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
