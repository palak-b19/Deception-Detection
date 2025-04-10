import json
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
from sklearn.metrics import f1_score
import numpy as np

# --- Data Loading Functions ---

def load_data(file_path):
    """Load JSONL file into a list of message dictionaries."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def group_into_conversations(data):
    """Group messages into conversations by game_id, speaker, and receiver."""
    conversations = defaultdict(list)
    for msg in data:
        # Only include messages with valid sender_labels
        if msg['sender_labels'] in ['true', 'false']:
            key = (msg['game_id'], msg['speakers'], msg['receivers'])
            conversations[key].append(msg)
    # Sort each conversation by relative_message_index
    for key in conversations:
        conversations[key].sort(key=lambda x: x['relative_message_index'])
    return list(conversations.values())

def load_glove(glove_path):
    """Load GloVe embeddings into a dictionary."""
    word2vec = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]])
            word2vec[word] = vector
    return word2vec

def build_vocab(conversations, glove_words, min_freq=1):
    """Build vocabulary from training data, including only words in GloVe."""
    word_freq = Counter()
    for conv in conversations:
        for msg in conv:
            words = msg['messages'].lower().split()  # Simple tokenization
            word_freq.update(words)
    
    vocab = ['<PAD>', '<UNK>'] + [word for word, freq in word_freq.items() 
                                 if freq >= min_freq and word in glove_words]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word2idx

def create_embedding_matrix(vocab, word2vec, embedding_dim):
    """Create embedding matrix with GloVe vectors."""
    embedding_matrix = torch.zeros((len(vocab), embedding_dim))
    for idx, word in enumerate(vocab):
        if word in word2vec:
            embedding_matrix[idx] = word2vec[word]
        elif word == '<PAD>':
            embedding_matrix[idx] = torch.zeros(embedding_dim)
        else:
            embedding_matrix[idx] = torch.randn(embedding_dim)
    return embedding_matrix

# --- Dataset Preparation ---

def prepare_conversation(conv, word2idx):
    """Prepare messages, game features, and labels for a conversation."""
    messages = []
    game_features = []
    labels = []
    for msg in conv:
        text = msg['messages'].lower().split()
        token_ids = [word2idx.get(word, word2idx['<UNK>']) for word in text]
        messages.append(token_ids)
        # Game features: game_score and score_delta
        gf = [float(msg['game_score']), float(msg['score_delta'])]
        game_features.append(gf)
        # Map 'true' (lie) to 1, 'false' (truthful) to 0
        label = 1 if msg['sender_labels'] == 'true' else 0
        labels.append(label)
    return messages, game_features, labels

class DiplomacyDataset(Dataset):
    def __init__(self, conversations, word2idx):
        self.data = []
        for conv in conversations:
            messages, game_features, labels = prepare_conversation(conv, word2idx)
            self.data.append((messages, game_features, labels))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """Collate a batch of conversations with padding."""
    batch_size = len(batch)
    max_seq_len = max(len(conv[0]) for conv in batch)
    max_msg_len = max(max(len(msg) for msg in conv[0]) for conv in batch)
    num_game_features = 2  # game_score, score_delta
    
    padded_messages = torch.zeros(max_seq_len, batch_size, max_msg_len, dtype=torch.long)
    padded_game_features = torch.zeros(max_seq_len, batch_size, num_game_features)
    padded_labels = torch.zeros(max_seq_len, batch_size, dtype=torch.float)
    conv_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for b, (messages, game_features, labels) in enumerate(batch):
        conv_lengths[b] = len(messages)
        for t, (msg, gf, label) in enumerate(zip(messages, game_features, labels)):
            msg_len = len(msg)
            padded_messages[t, b, :msg_len] = torch.tensor(msg)
            padded_game_features[t, b] = torch.tensor(gf)
            padded_labels[t, b] = label
    
    return padded_messages, padded_game_features, padded_labels, conv_lengths

# --- Model Definition ---

class HierarchicalLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_game_features, pos_weight):
        super(HierarchicalLSTM, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False  # Fixed embeddings as in baseline
        
        # Message encoder: Bidirectional LSTM with max pooling
        self.message_encoder = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True)
        # Conversation encoder: Unidirectional LSTM
        self.conversation_encoder = nn.LSTM(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)  # Dropout as per config
        self.classifier = nn.Linear(hidden_dim + num_game_features, 1)
        self.pos_weight = torch.tensor(pos_weight)
    
    def forward(self, messages, game_features, conv_lengths):
        """
        messages: (seq_len, batch, max_msg_len)
        game_features: (seq_len, batch, num_game_features)
        conv_lengths: (batch,)
        """
        seq_len, batch_size = messages.size(0), messages.size(1)
        
        # Encode each message
        encoded_messages = []
        for t in range(seq_len):
            msg = messages[t]  # (batch, max_msg_len)
            embedded = self.embedding(msg)  # (batch, max_msg_len, embedding_dim)
            msg_lengths = (msg != 0).sum(1).cpu()  # (batch,)
            packed = pack_padded_sequence(embedded, msg_lengths, batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.message_encoder(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  # (batch, max_msg_len, hidden_dim)
            # Max pooling over time
            encoded_msg = lstm_out.max(dim=1)[0]  # (batch, hidden_dim)
            encoded_messages.append(encoded_msg)
        
        encoded_messages = torch.stack(encoded_messages)  # (seq_len, batch, hidden_dim)
        
        # Conversation encoder
        packed_conv = pack_padded_sequence(encoded_messages, conv_lengths, enforce_sorted=False)
        conv_output, _ = self.conversation_encoder(packed_conv)
        conv_output, _ = pad_packed_sequence(conv_output)  # (seq_len, batch, hidden_dim)
        conv_output = self.dropout(conv_output)
        
        # Concatenate with game features
        combined = torch.cat([conv_output, game_features], dim=2)  # (seq_len, batch, hidden_dim + num_game_features)
        
        # Classification
        logits = self.classifier(combined).squeeze(2)  # (seq_len, batch)
        return logits
    
    def loss(self, logits, labels, conv_lengths):
        """Compute BCE loss with pos_weight, masking padded positions."""
        mask = torch.arange(logits.size(0), device=logits.device).unsqueeze(1) < conv_lengths
        logits_flat = logits[mask]
        labels_flat = labels[mask]
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        return loss_fn(logits_flat, labels_flat)

# --- Evaluation Function ---

def evaluate(model, loader, device):
    """Compute macro F1 score on the dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            messages, game_features, labels, conv_lengths = [x.to(device) for x in batch]
            logits = model(messages, game_features, conv_lengths)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            mask = torch.arange(logits.size(0), device=device).unsqueeze(1) < conv_lengths
            preds_flat = preds[mask].cpu().numpy()
            labels_flat = labels[mask].cpu().numpy()
            
            all_preds.extend(preds_flat)
            all_labels.extend(labels_flat)
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return macro_f1

# --- Main Execution ---

def main():
    # File paths (adjust based on your Kaggle dataset paths)
    train_path = '/kaggle/input/diplomacy/train.jsonl'
    val_path = '/kaggle/input/diplomacy/validation.jsonl'
    test_path = '/kaggle/input/diplomacy/test.jsonl'
    glove_path = '/kaggle/input/glove/glove.twitter.27B.200d.txt'  # Upload GloVe file to Kaggle
    
    # Load and prepare data
    train_data = load_data(train_path)
    val_data = load_data(val_path)
    test_data = load_data(test_path)
    
    train_convs = group_into_conversations(train_data)
    val_convs = group_into_conversations(val_data)
    test_convs = group_into_conversations(test_data)
    
    word2vec = load_glove(glove_path)
    vocab, word2idx = build_vocab(train_convs, word2vec.keys())
    embedding_matrix = create_embedding_matrix(vocab, word2vec, embedding_dim=200)
    
    train_dataset = DiplomacyDataset(train_convs, word2idx)
    val_dataset = DiplomacyDataset(val_convs, word2idx)
    test_dataset = DiplomacyDataset(test_convs, word2idx)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HierarchicalLSTM(
        embedding_matrix=embedding_matrix,
        hidden_dim=200,
        num_game_features=2,  # game_score, score_delta
        pos_weight=10.0  # As per baseline config
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    
    # Training loop with early stopping
    best_val_f1 = 0.0
    patience = 10
    patience_counter = 0
    num_epochs = 15  # Adjust as needed
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            messages, game_features, labels, conv_lengths = [x.to(device) for x in batch]
            optimizer.zero_grad()
            logits = model(messages, game_features, conv_lengths)
            loss = model.loss(logits, labels, conv_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Validation
        val_f1 = evaluate(model, val_loader, device)
        print(f"Validation Macro F1: {val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Optionally save model: torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Test evaluation
    test_f1 = evaluate(model, test_loader, device)
    print(f"Test Macro F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()
