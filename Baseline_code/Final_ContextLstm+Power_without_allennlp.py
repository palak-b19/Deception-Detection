import json
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm  # For progress tracking

# --- Data Loading Functions ---

def load_conversations(file_path):
    """
    Load JSONL file where each line is a conversation with keys containing lists.
    """
    conversations = []
    with open(file_path, 'r') as f:
        for line in f:
            conversations.append(json.loads(line))
    return conversations

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
    """
    Build vocabulary from conversation data.
    Iterates over each conversation's messages (which are lists of strings).
    """
    word_freq = Counter()
    for conv in conversations:
        for message in conv['messages']:
            words = message.lower().split()  # Simple tokenization
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
    """
    Prepare messages, game features, and labels for a conversation.
    Expects conv keys as lists: 'messages', 'game_score', 'game_score_delta', 'sender_labels'.
    Inserts a dummy token if a message tokenizes to an empty list.
    """
    messages = []
    game_features = []
    labels = []
    for text, game_score, score_delta, sender_label in zip(
            conv['messages'], conv['game_score'], conv['game_score_delta'], conv['sender_labels']):
        # Tokenize the message
        token_ids = [word2idx.get(word, word2idx['<UNK>']) for word in text.lower().split()]
        # If the message is empty, insert a dummy token (e.g., <UNK>)
        if len(token_ids) == 0:
            token_ids = [word2idx['<UNK>']]
        messages.append(token_ids)
        # Convert game features to float
        gf = [float(game_score), float(score_delta)]
        game_features.append(gf)
        # Convert sender_label to int: True/"true" -> 1, else 0
        label = 1 if (sender_label is True or str(sender_label).lower() == "true") else 0
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
    """Collate a batch of conversations with padding.
    
       If a conversation is empty, we replace it with a dummy message.
    """
    batch_size = len(batch)
    num_game_features = 2  # game_score and score_delta
    
    # Determine maximum number of messages (sequence length) in the batch
    max_seq_len = max(len(conv[0]) if len(conv[0]) > 0 else 1 for conv in batch)
    # Determine maximum message length (number of tokens)
    max_msg_len = max(
        (max((len(msg) for msg in conv[0]), default=0) if len(conv[0]) > 0 else 0)
        for conv in batch
    )
    if max_msg_len == 0:
        max_msg_len = 1

    padded_messages = torch.zeros(max_seq_len, batch_size, max_msg_len, dtype=torch.long)
    padded_game_features = torch.zeros(max_seq_len, batch_size, num_game_features)
    padded_labels = torch.zeros(max_seq_len, batch_size, dtype=torch.float)
    conv_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for b, (messages, game_features, labels) in enumerate(batch):
        # If a conversation has no messages, create a dummy padded message.
        if len(messages) == 0:
            messages = [[0]]
            game_features = [[0.0] * num_game_features]
            labels = [0.0]
        conv_lengths[b] = len(messages)
        for t, (msg, gf, label) in enumerate(zip(messages, game_features, labels)):
            msg_len = len(msg)
            padded_messages[t, b, :msg_len] = torch.tensor(msg, dtype=torch.long)
            padded_game_features[t, b] = torch.tensor(gf, dtype=torch.float)
            padded_labels[t, b] = label
    
    return padded_messages, padded_game_features, padded_labels, conv_lengths

# --- Model Definition ---

class HierarchicalLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_game_features, pos_weight):
        super(HierarchicalLSTM, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False  # Fixed embeddings
        
        # Message encoder: Bidirectional LSTM with max pooling over tokens
        self.message_encoder = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True)
        # Conversation encoder: Unidirectional LSTM over message representations
        self.conversation_encoder = nn.LSTM(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        # Combine conversation encoding with game features for classification
        self.classifier = nn.Linear(hidden_dim + num_game_features, 1)
        self.pos_weight = torch.tensor(pos_weight)
    
    def forward(self, messages, game_features, conv_lengths):
        """
        messages: (seq_len, batch, max_msg_len)
        game_features: (seq_len, batch, num_game_features)
        conv_lengths: (batch,)
        """
        seq_len, batch_size = messages.size(0), messages.size(1)
        
        # Encode each message in the conversation individually
        encoded_messages = []
        for t in range(seq_len):
            msg = messages[t]  # (batch, max_msg_len)
            embedded = self.embedding(msg)  # (batch, max_msg_len, embedding_dim)
            # Count non-zero tokens; ensure at least one element to avoid error
            msg_lengths = (msg != 0).sum(1).cpu()
            # Check for any zeros in lengths; if so, set them to 1
            msg_lengths = torch.clamp(msg_lengths, min=1)
            packed = pack_padded_sequence(embedded, msg_lengths, batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.message_encoder(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  # (batch, max_msg_len, hidden_dim)
            # Use max pooling over time (word dimension)
            encoded_msg = lstm_out.max(dim=1)[0]  # (batch, hidden_dim)
            encoded_messages.append(encoded_msg)
        
        encoded_messages = torch.stack(encoded_messages)  # (seq_len, batch, hidden_dim)
        
        # Encode the sequence of message representations (i.e., the conversation)
        packed_conv = pack_padded_sequence(encoded_messages, conv_lengths, enforce_sorted=False)
        conv_output, _ = self.conversation_encoder(packed_conv)
        conv_output, _ = pad_packed_sequence(conv_output)  # (seq_len, batch, hidden_dim)
        conv_output = self.dropout(conv_output)
        
        # Concatenate conversation encoding with game features
        combined = torch.cat([conv_output, game_features], dim=2)  # (seq_len, batch, hidden_dim + num_game_features)
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
    # File paths (adjust these paths as necessary)
    train_path = 'data/train.jsonl'
    val_path   = 'data/validation.jsonl'
    test_path  = 'data/test.jsonl'
    glove_path = 'data/glove.twitter.27B.200d.txt'  # Ensure the file is in the data folder
    
    # Load conversations (each file contains conversation objects)
    train_convs = load_conversations(train_path)
    val_convs   = load_conversations(val_path)
    test_convs  = load_conversations(test_path)
    
    # Load GloVe embeddings
    word2vec = load_glove(glove_path)
    
    # Build vocabulary based on training conversations
    vocab, word2idx = build_vocab(train_convs, word2vec.keys())
    embedding_matrix = create_embedding_matrix(vocab, word2vec, embedding_dim=200)
    
    # Create datasets
    train_dataset = DiplomacyDataset(train_convs, word2idx)
    val_dataset   = DiplomacyDataset(val_convs, word2idx)
    test_dataset  = DiplomacyDataset(test_convs, word2idx)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)
    
    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HierarchicalLSTM(
        embedding_matrix=embedding_matrix,
        hidden_dim=200,
        num_game_features=2,  # game_score and score_delta
        pos_weight=10.0       # Pos weight as per baseline configuration
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
        # Wrap training loop with tqdm for a progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
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
        
        # Early stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Optionally, save the model state: torch.save(model.state_dict(), 'best_model.pt')
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

# import json
# import torch
# import torch.nn as nn
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from torch.utils.data import Dataset, DataLoader
# from collections import Counter
# from sklearn.metrics import f1_score
# import numpy as np
# from tqdm import tqdm

# # --- Data Loading Functions ---

# def load_conversations(file_path):
#     conversations = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             conversations.append(json.loads(line))
#     return conversations

# def load_glove(glove_path):
#     word2vec = {}
#     with open(glove_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             values = line.split()
#             word = values[0]
#             vector = torch.tensor([float(val) for val in values[1:]])
#             word2vec[word] = vector
#     return word2vec

# def build_vocab(conversations, glove_words, min_freq=1):
#     word_freq = Counter()
#     for conv in conversations:
#         for message in conv['messages']:
#             words = message.lower().split()
#             word_freq.update(words)
    
#     vocab = ['<PAD>', '<UNK>'] + [word for word, freq in word_freq.items() if freq >= min_freq and word in glove_words]
#     word2idx = {word: idx for idx, word in enumerate(vocab)}
#     return vocab, word2idx

# def create_embedding_matrix(vocab, word2vec, embedding_dim):
#     embedding_matrix = torch.zeros((len(vocab), embedding_dim))
#     for idx, word in enumerate(vocab):
#         if word in word2vec:
#             embedding_matrix[idx] = word2vec[word]
#         elif word == '<PAD>':
#             embedding_matrix[idx] = torch.zeros(embedding_dim)
#         else:
#             embedding_matrix[idx] = torch.randn(embedding_dim)
#     return embedding_matrix

# # --- Dataset Preparation ---

# def prepare_conversation(conv, word2idx):
#     messages, game_features, labels = [], [], []
#     for text, game_score, score_delta, sender_label in zip(conv['messages'], conv['game_score'], conv['game_score_delta'], conv['sender_labels']):
#         token_ids = [word2idx.get(word, word2idx['<UNK>']) for word in text.lower().split()]
#         if len(token_ids) == 0:
#             token_ids = [word2idx['<UNK>']]
#         messages.append(token_ids)
#         gf = [float(game_score), float(score_delta)]
#         game_features.append(gf)
#         label = 1 if (sender_label is True or str(sender_label).lower() == "true") else 0
#         labels.append(label)
#     return messages, game_features, labels

# class DiplomacyDataset(Dataset):
#     def __init__(self, conversations, word2idx):
#         self.data = [prepare_conversation(conv, word2idx) for conv in conversations]
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return self.data[idx]

# def collate_fn(batch):
#     batch_size = len(batch)
#     num_game_features = 2
#     max_seq_len = max(len(conv[0]) if len(conv[0]) > 0 else 1 for conv in batch)
#     max_msg_len = max((max((len(msg) for msg in conv[0]), default=0) if len(conv[0]) > 0 else 0) for conv in batch)
#     max_msg_len = max(max_msg_len, 1)

#     padded_messages = torch.zeros(max_seq_len, batch_size, max_msg_len, dtype=torch.long)
#     padded_game_features = torch.zeros(max_seq_len, batch_size, num_game_features)
#     padded_labels = torch.zeros(max_seq_len, batch_size, dtype=torch.float)
#     conv_lengths = torch.zeros(batch_size, dtype=torch.long)
    
#     for b, (messages, game_features, labels) in enumerate(batch):
#         if len(messages) == 0:
#             messages, game_features, labels = [[0]], [[0.0] * num_game_features], [0.0]
#         conv_lengths[b] = len(messages)
#         for t, (msg, gf, label) in enumerate(zip(messages, game_features, labels)):
#             msg_len = len(msg)
#             padded_messages[t, b, :msg_len] = torch.tensor(msg, dtype=torch.long)
#             padded_game_features[t, b] = torch.tensor(gf, dtype=torch.float)
#             padded_labels[t, b] = label
#     return padded_messages, padded_game_features, padded_labels, conv_lengths

# # --- Model Definition ---

# class HierarchicalLSTM(nn.Module):
#     def __init__(self, embedding_matrix, hidden_dim, num_game_features, pos_weight):
#         super(HierarchicalLSTM, self).__init__()
#         vocab_size, embedding_dim = embedding_matrix.shape
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.embedding.weight.data.copy_(embedding_matrix)
#         self.embedding.weight.requires_grad = False
        
#         self.message_encoder = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True)
#         self.conversation_encoder = nn.LSTM(hidden_dim, hidden_dim)
#         self.dropout = nn.Dropout(0.3)
#         self.classifier = nn.Linear(hidden_dim + num_game_features, 1)
#         self.pos_weight = torch.tensor(pos_weight)
    
#     def forward(self, messages, game_features, conv_lengths):
#         seq_len, batch_size = messages.size(0), messages.size(1)
#         encoded_messages = []
#         for t in range(seq_len):
#             msg = messages[t]
#             embedded = self.embedding(msg)
#             msg_lengths = (msg != 0).sum(1).cpu()
#             msg_lengths = torch.clamp(msg_lengths, min=1)
#             packed = pack_padded_sequence(embedded, msg_lengths, batch_first=True, enforce_sorted=False)
#             lstm_out, _ = self.message_encoder(packed)
#             lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
#             encoded_msg = lstm_out.max(dim=1)[0]
#             encoded_messages.append(encoded_msg)
#         encoded_messages = torch.stack(encoded_messages)
#         packed_conv = pack_padded_sequence(encoded_messages, conv_lengths, enforce_sorted=False)
#         conv_output, _ = self.conversation_encoder(packed_conv)
#         conv_output, _ = pad_packed_sequence(conv_output)
#         conv_output = self.dropout(conv_output)
#         combined = torch.cat([conv_output, game_features], dim=2)
#         logits = self.classifier(combined).squeeze(2)
#         return logits
    
#     def loss(self, logits, labels, conv_lengths):
#         mask = torch.arange(logits.size(0), device=logits.device).unsqueeze(1) < conv_lengths
#         logits_flat = logits[mask]
#         labels_flat = labels[mask]
#         loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
#         return loss_fn(logits_flat, labels_flat)

# # --- Evaluation Function ---

# def evaluate(model, loader, device):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for batch in loader:
#             messages, game_features, labels, conv_lengths = [x.to(device) for x in batch]
#             logits = model(messages, game_features, conv_lengths)
#             preds = (torch.sigmoid(logits) > 0.5).float()
#             mask = torch.arange(logits.size(0), device=device).unsqueeze(1) < conv_lengths
#             preds_flat = preds[mask].cpu().numpy()
#             labels_flat = labels[mask].cpu().numpy()
#             all_preds.extend(preds_flat)
#             all_labels.extend(labels_flat)
    
#     macro_f1 = f1_score(all_labels, all_preds, average='macro')
#     micro_f1 = f1_score(all_labels, all_preds, average='binary')
#     return macro_f1, micro_f1

# # --- Main Execution ---

# def main():
#     train_path = 'data/train.jsonl'
#     val_path   = 'data/validation.jsonl'
#     test_path  = 'data/test.jsonl'
#     glove_path = 'data/glove.twitter.27B.200d.txt'
    
#     train_convs = load_conversations(train_path)
#     val_convs = load_conversations(val_path)
#     test_convs = load_conversations(test_path)
    
#     glove = load_glove(glove_path)
#     vocab, word2idx = build_vocab(train_convs + val_convs + test_convs, glove)
#     embedding_dim = list(glove.values())[0].shape[0]
#     embedding_matrix = create_embedding_matrix(vocab, glove, embedding_dim)
    
#     train_dataset = DiplomacyDataset(train_convs, word2idx)
#     val_dataset = DiplomacyDataset(val_convs, word2idx)
#     test_dataset = DiplomacyDataset(test_convs, word2idx)
    
#     train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
#     val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)
#     test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     all_labels = [label for _, _, labels in train_dataset for label in labels]
#     pos_weight = len(all_labels) / sum(all_labels)
    
#     model = HierarchicalLSTM(embedding_matrix, hidden_dim=128, num_game_features=2, pos_weight=pos_weight).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
#     for epoch in range(10):
#         model.train()
#         total_loss = 0
#         for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
#             messages, game_features, labels, conv_lengths = [x.to(device) for x in batch]
#             optimizer.zero_grad()
#             logits = model(messages, game_features, conv_lengths)
#             loss = model.loss(logits, labels, conv_lengths)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
        
#         val_macro_f1, val_f1 = evaluate(model, val_loader, device)
#         print(f"Epoch {epoch+1} - Loss: {total_loss:.4f} | Val Macro F1: {val_macro_f1:.4f} | Val F1: {val_f1:.4f}")
    
#     test_macro_f1, test_f1 = evaluate(model, test_loader, device)
#     print(f"\nTest Macro F1: {test_macro_f1:.4f}")
#     print(f"Test F1: {test_f1:.4f}")

# if __name__ == "__main__":
#     main()
