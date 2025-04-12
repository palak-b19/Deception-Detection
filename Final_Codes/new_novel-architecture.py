import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader
import jsonlines
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm  # Import tqdm for progress bars

print("starting")

# Dataset class to handle Diplomacy game data
class DiplomacyDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128, device='cpu'):
        self.data = []
        print(f"Loading dataset from {file_path}...")
        with jsonlines.open(file_path, 'r') as reader:
            dialogs_list = list(reader)  # Convert iterator to list for tqdm
            for dialogs in tqdm(dialogs_list, desc="Processing dialogs"):
                # Create list of tuples for messages and features to use with tqdm
                message_data = list(zip(
                    dialogs['messages'], 
                    dialogs['sender_labels'], 
                    dialogs['game_score'], 
                    dialogs['game_score_delta']
                ))
                # Iterate over messages with tqdm
                for msg, sender_label, game_score, game_score_delta in tqdm(
                    message_data, 
                    desc="Processing messages", 
                    leave=False
                ):
                    # Construct game_state as a list of floats
                    try:
                        game_state = [float(game_score), float(game_score_delta)]
                    except (ValueError, TypeError):
                        # Skip if conversion fails
                        continue

                    # Convert sender_label to binary label (0 for truthful, 1 for deceptive)
                    if sender_label is True:
                        label = 0  # Truthful
                    elif sender_label is False:
                        label = 1  # Deceptive
                    else:
                        continue  # Skip if label is None or invalid

                    self.data.append({
                        'text': msg,
                        'label': label,
                        'game_state': game_state
                    })
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        game_state = item['game_state']
        
        # Tokenize and move tensors to device
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0).to(self.device)
        attention_mask = encoding['attention_mask'].squeeze(0).to(self.device)
        
        # Create tensors directly on device
        game_state = torch.tensor(game_state, dtype=torch.float32, device=self.device)
        label = torch.tensor(label, dtype=torch.float32, device=self.device)
        
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
        # Encode game-state features with a 2-layer MLP
        self.game_state_encoder = nn.Sequential(
            nn.Linear(game_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # Transformer encoder layer for fusion
        self.fusion_transformer = nn.TransformerEncoderLayer(d_model=768 + 32, nhead=4, dim_feedforward=2048)
        self.classifier = nn.Linear(768 + 32, 1)  # Output a single logit
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, game_state):
        # Text encoding using RoBERTa
        text_embeds = self.roberta(input_ids=input_ids, attention_mask=attention_mask).pooler_output  # (batch_size, 768)

        # Game-state encoding
        game_embeds = self.game_state_encoder(game_state)  # (batch_size, 32)

        # Fusion of text and game-state features
        fused = torch.cat((text_embeds, game_embeds), dim=1)  # (batch_size, 800)
        fused = self.fusion_transformer(fused.unsqueeze(0)).squeeze(0)  # (batch_size, 800)
        fused = self.dropout(fused)

        # Classification
        logits = self.classifier(fused)  # (batch_size, 1)
        probs = torch.sigmoid(logits)    # Probability of deception
        return probs

# Focal loss to handle class imbalance
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
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
    # Add tqdm progress bar for training batches
    for batch in tqdm(dataloader, desc="Training batches"):
        input_ids = batch['input_ids']  # Already on device
        attention_mask = batch['attention_mask']  # Already on device
        game_state = batch['game_state']  # Already on device
        labels = batch['label']  # Already on device

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
    preds, labels = [], []
    with torch.no_grad():
        # Add tqdm progress bar for evaluation batches
        for batch in tqdm(dataloader, desc="Evaluation batches"):
            input_ids = batch['input_ids']  # Already on device
            attention_mask = batch['attention_mask']  # Already on device
            game_state = batch['game_state']  # Already on device
            label = batch['label']  # Already on device

            outputs = model(input_ids, attention_mask, game_state)
            # Move outputs to CPU for numpy conversion
            preds.extend(outputs.squeeze(1).cpu().numpy())
            labels.extend(label.cpu().numpy())
    preds = np.array(preds) > 0.5  # Threshold at 0.5
    labels = np.array(labels)
    macro_f1 = f1_score(labels, preds, average='macro')
    lie_f1 = f1_score(labels, preds, pos_label=1)
    accuracy = accuracy_score(labels, preds)
    return macro_f1, lie_f1, accuracy

# Main function to run the training and evaluation
def main():
    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 2e-5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer and RoBERTa model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)  # Move to device

    # Game state dimension is 2 (game_score, game_score_delta)
    game_state_dim = 2
    model = DeceptionDetectionModel(roberta_model, game_state_dim).to(device)  # Ensure model is on device

    # Load datasets
    train_dataset = DiplomacyDataset('/kaggle/input/dataset-deception/train.jsonl', tokenizer, device=device)
    val_dataset = DiplomacyDataset('/kaggle/input/dataset-deception/validation.jsonl', tokenizer,device=device)
    test_dataset = DiplomacyDataset('/kaggle/input/dataset-deception/test.jsonl', tokenizer, device=device)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=False  # Disable pin_memory since tensors are on GPU
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        pin_memory=False  # Disable pin_memory since tensors are on GPU
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        pin_memory=False  # Disable pin_memory since tensors are on GPU
    )

    # Optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = FocalLoss()

    # Training loop with tqdm for epochs
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        macro_f1, lie_f1, accuracy = evaluate(model, val_loader, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Macro F1: {macro_f1:.4f}, Lie F1: {lie_f1:.4f}, Accuracy: {accuracy:.4f}')

    # Test evaluation with tqdm
    print("Running final test evaluation...")
    macro_f1, lie_f1, accuracy = evaluate(model, test_loader, device)
    print(f'Test Macro F1: {macro_f1:.4f}, Lie F1: {lie_f1:.4f}, Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()
