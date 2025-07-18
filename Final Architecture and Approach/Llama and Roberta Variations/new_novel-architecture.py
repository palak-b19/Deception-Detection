import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import jsonlines
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm.notebook import tqdm
import collections

print("Starting")

# Dataset class with conversational context
class DiplomacyDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128, context_size=3, device='cpu'):
        self.data = []
        self.context_size = context_size  # Number of previous messages to include
        print(f"Loading dataset from {file_path}...")
        with jsonlines.open(file_path, 'r') as reader:
            dialogs_list = list(reader)
            total_messages = sum(len(dialogs['messages']) for dialogs in dialogs_list)
            
            for dialogs in tqdm(dialogs_list, desc="Processing dialogs", position=0):
                messages = dialogs['messages']
                sender_labels = dialogs['sender_labels']
                game_scores = dialogs['game_score']
                score_deltas = dialogs['game_score_delta']
                
                # Create context windows
                for i in range(len(messages)):
                    # Get current message and up to context_size previous messages
                    context_messages = []
                    for j in range(max(0, i - self.context_size), i + 1):
                        context_messages.append(messages[j])
                    # Join context messages with a separator
                    text = " [SEP] ".join(context_messages)
                    
                    # Get label and game state for current message
                    try:
                        game_state = [float(game_scores[i]), float(score_deltas[i])]
                    except (ValueError, TypeError):
                        continue
                    
                    if sender_labels[i] is True:
                        label = 0  # Truthful
                    elif sender_labels[i] is False:
                        label = 1  # Deceptive
                    else:
                        continue
                    
                    self.data.append({
                        'text': text,
                        'label': label,
                        'game_state': game_state
                    })

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

        # Compute class weights for sampler
        labels = [item['label'] for item in self.data]
        class_counts = collections.Counter(labels)
        total = len(labels)
        weights = [total / (len(class_counts) * class_counts[label]) for label in labels]
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        game_state = item['game_state']
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0).to(self.device)
        attention_mask = encoding['attention_mask'].squeeze(0).to(self.device)
        game_state = torch.tensor(game_state, dtype=torch.float32, device=self.device)
        label = torch.tensor(label, dtype=torch.float32, device=self.device)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'game_state': game_state,
            'label': label
        }

# Model architecture (unchanged)
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
        self.fusion_transformer = nn.TransformerEncoderLayer(d_model=768 + 32, nhead=4, dim_feedforward=2048)
        self.classifier = nn.Linear(768 + 32, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, game_state):
        text_embeds = self.roberta(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        game_embeds = self.game_state_encoder(game_state)
        fused = torch.cat((text_embeds, game_embeds), dim=1)
        fused = self.fusion_transformer(fused.unsqueeze(0)).squeeze(0)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        probs = torch.sigmoid(logits)
        return probs

# Focal loss with adjusted alpha
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.75):
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
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        game_state = batch['game_state']
        labels = batch['label']

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
        for batch in tqdm(dataloader, desc="Evaluation batches", position=1, leave=True):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            game_state = batch['game_state']
            label = batch['label']

            outputs = model(input_ids, attention_mask, game_state)
            preds.extend(outputs.squeeze(1).cpu().numpy())
            labels.extend(label.cpu().numpy())
    preds = np.array(preds) > 0.5
    labels = np.array(labels)
    macro_f1 = f1_score(labels, preds, average='macro')
    lie_f1 = f1_score(labels, preds, pos_label=1)
    accuracy = accuracy_score(labels, preds)
    return macro_f1, lie_f1, accuracy

# Main function
def main():
    # Hyperparameters
    batch_size = 16  # Reduced from 32
    num_epochs = 15  # Increased to match paper
    learning_rate = 2e-5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
    game_state_dim = 2  # game_score, score_delta
    model = DeceptionDetectionModel(roberta_model, game_state_dim).to(device)

    # Load datasets
    train_dataset = DiplomacyDataset('/kaggle/input/dataset-deception/train.jsonl', tokenizer, device=device)
    val_dataset = DiplomacyDataset('/kaggle/input/dataset-deception/validation.jsonl', tokenizer, device=device)
    test_dataset = DiplomacyDataset('/kaggle/input/dataset-deception/test.jsonl', tokenizer, device=device)

    # Weighted sampler for class imbalance
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(train_dataset.weights, len(train_dataset)),
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=False
    )

    # Optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = FocalLoss()

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Epochs", position=0):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        macro_f1, lie_f1, accuracy = evaluate(model, val_loader, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Macro F1: {macro_f1:.4f}, Lie F1: {lie_f1:.4f}, Accuracy: {accuracy:.4f}')

    # Test evaluation
    print("Running final test evaluation...")
    macro_f1, lie_f1, accuracy = evaluate(model, test_loader, device)
    print(f'Test Macro F1: {macro_f1:.4f}, Lie F1: {lie_f1:.4f}, Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()
