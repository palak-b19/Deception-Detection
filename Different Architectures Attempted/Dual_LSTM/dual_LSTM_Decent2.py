
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import json
from tqdm import tqdm
import logging
import time
from sklearn.metrics import precision_recall_fscore_support

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load BERT model and tokenizer
logger.info("Loading BERT model and tokenizer...")
start_time = time.time()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
logger.info(f"BERT loaded in {time.time() - start_time:.2f} seconds")

# Add [EMPTY] token
if '[EMPTY]' not in tokenizer.get_vocab():
    tokenizer.add_tokens(['[EMPTY]'])
    bert_model.resize_token_embeddings(len(tokenizer))

# Model Definition
class DualLSTMDeceptionDetector(nn.Module):
    def __init__(self, num_countries, country_emb_dim, hidden_dim, bert_dim=768):
        super(DualLSTMDeceptionDetector, self).__init__()
        self.num_countries = num_countries
        # Embedding with padding index
        self.country_embedding = nn.Embedding(num_countries + 1, country_emb_dim, padding_idx=num_countries)
        self.sender_lstm = nn.LSTM(country_emb_dim + bert_dim, hidden_dim, batch_first=True)
        self.receiver_lstm = nn.LSTM(country_emb_dim + bert_dim, hidden_dim, batch_first=True)
        # Single classifier for sender_labels using combined context
        self.classifier = nn.Linear(hidden_dim * 2, 1)  # Concat sender + receiver hidden states

    def forward(self, messages, senders, receivers, lengths):
        batch_size, seq_len = senders.shape
        message_embs = self.get_message_embeddings(messages, batch_size, seq_len)
        
        # Prepare inputs
        sender_embs = self.country_embedding(senders)
        receiver_embs = self.country_embedding(receivers)
        sender_input = torch.cat([message_embs, sender_embs], dim=2)
        receiver_input = torch.cat([message_embs, receiver_embs], dim=2)
        
        # Pack sequences to handle padding efficiently
        sender_input_packed = pack_padded_sequence(sender_input, lengths, batch_first=True, enforce_sorted=False)
        receiver_input_packed = pack_padded_sequence(receiver_input, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM outputs
        sender_out_packed, _ = self.sender_lstm(sender_input_packed)
        receiver_out_packed, _ = self.receiver_lstm(receiver_input_packed)
        sender_out, _ = torch.nn.utils.rnn.pad_packed_sequence(sender_out_packed, batch_first=True)
        receiver_out, _ = torch.nn.utils.rnn.pad_packed_sequence(receiver_out_packed, batch_first=True)
        
        # Combine sender and receiver context
        combined_out = torch.cat([sender_out, receiver_out], dim=2)  # (batch_size, seq_len, hidden_dim * 2)
        final_pred = self.classifier(combined_out)  # (batch_size, seq_len, 1)
        
        return final_pred

    def get_message_embeddings(self, messages, batch_size, seq_len):
        message_embs = []
        for batch_idx in range(batch_size):
            seq_embs = []
            for seq_idx in range(seq_len):
                if seq_idx < len(messages[batch_idx]):
                    msg = messages[batch_idx][seq_idx] or '[EMPTY]'
                    inputs = tokenizer(msg, return_tensors='pt', padding=True, truncation=True, max_length=128)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = bert_model(**inputs).last_hidden_state[:, 0, :]
                    seq_embs.append(outputs.squeeze(0))
                else:
                    seq_embs.append(torch.zeros(768, device=device))
            message_embs.append(torch.stack(seq_embs))
        return torch.stack(message_embs)

# Data Preprocessing
def preprocess_data(data_file, country_map=None):
    logger.info(f"Preprocessing {data_file}")
    with open(data_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    if country_map is None:
        countries = set()
        for game in data:
            countries.update(game['speakers'])
            countries.update(game['receivers'])
        country_map = {c: i for i, c in enumerate(sorted(countries))}
        logger.info(f"Created country map with {len(country_map)} countries")
    
    conversations = []
    for game in data:
        messages = [msg if msg else '[EMPTY]' for msg in game['messages']]
        senders = [country_map[s] for s in game['speakers']]
        receivers = [country_map[r] for r in game['receivers']]
        sender_labels = [1 if l else 0 for l in game['sender_labels']]
        receiver_labels = [-1 if l == "NOANNOTATION" else (1 if l else 0) for l in game['receiver_labels']]
        
        conversations.append({
            'messages': messages,
            'speakers': torch.tensor(senders, dtype=torch.long),
            'receivers': torch.tensor(receivers, dtype=torch.long),
            'sender_labels': torch.tensor(sender_labels, dtype=torch.float32),
            'receiver_labels': torch.tensor(receiver_labels, dtype=torch.float32)
        })
    
    # Filter out conversations with zero messages
    original_count = len(conversations)
    conversations = [conv for conv in conversations if len(conv['messages']) > 0]
    logger.info(f"Filtered out {original_count - len(conversations)} conversations with zero messages from {data_file}")
    
    return conversations, country_map

# Compute Metrics
def compute_metrics(preds, labels, mask=None):
    if mask is not None:
        preds = preds[mask.bool()]
        labels = labels[mask.bool()]
    preds = (preds > 0.5).float()
    labels = labels.float()
    
    precision, recall, fscore, _ = precision_recall_fscore_support(labels.cpu(), preds.cpu(), average=None, labels=[0, 1], zero_division=0)
    micro = precision_recall_fscore_support(labels.cpu(), preds.cpu(), average='micro', zero_division=0)[:3]
    macro = precision_recall_fscore_support(labels.cpu(), preds.cpu(), average='macro', zero_division=0)[:3]
    
    return {
        'True_precision': precision[1], 'False_precision': precision[0],
        'True_recall': recall[1], 'False_recall': recall[0],
        'True_fscore': fscore[1], 'False_fscore': fscore[0],
        'micro_precision': micro[0], 'micro_recall': micro[1], 'micro_fscore': micro[2],
        'macro_precision': macro[0], 'macro_recall': macro[1], 'macro_fscore': macro[2]
    }

# Training and Evaluation
def train_and_evaluate(model, train_data, val_data, test_data, epochs=5, batch_size=4):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Compute pos_weight for class imbalance
    all_labels = []
    for conv in train_data:
        labels = conv['sender_labels']
        all_labels.extend([l.item() for l in labels if l != -1])
    num_lies = sum(l == 1 for l in all_labels)
    num_truths = len(all_labels) - num_lies
    pos_weight = num_truths / num_lies if num_lies > 0 else 1.0
    logger.info(f"Class stats - Lies: {num_lies}, Truths: {num_truths}, pos_weight: {pos_weight:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device), reduction='none')
    
    best_val_loss = float('inf')
    best_metrics = None
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch = train_data[i:i+batch_size]
            messages = [c['messages'] for c in batch]
            senders = pad_sequence([c['speakers'] for c in batch], batch_first=True, padding_value=model.num_countries).to(device)
            receivers = pad_sequence([c['receivers'] for c in batch], batch_first=True, padding_value=model.num_countries).to(device)
            sender_labels = pad_sequence([c['sender_labels'] for c in batch], batch_first=True, padding_value=-1).to(device)
            lengths = [len(c['messages']) for c in batch]
            
            # Safety check for lengths
            assert all(l > 0 for l in lengths), f"Found zero length in batch at index {i}"
            
            optimizer.zero_grad()
            final_pred = model(messages, senders, receivers, lengths)
            loss = criterion(final_pred.squeeze(-1), sender_labels)
            mask = (sender_labels != -1)
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / (len(train_data) / batch_size)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                messages = [c['messages'] for c in batch]
                senders = pad_sequence([c['speakers'] for c in batch], batch_first=True, padding_value=model.num_countries).to(device)
                receivers = pad_sequence([c['receivers'] for c in batch], batch_first=True, padding_value=model.num_countries).to(device)
                sender_labels = pad_sequence([c['sender_labels'] for c in batch], batch_first=True, padding_value=-1).to(device)
                lengths = [len(c['messages']) for c in batch]
                
                final_pred = model(messages, senders, receivers, lengths)
                loss = criterion(final_pred.squeeze(-1), sender_labels)
                mask = (sender_labels != -1)
                loss = (loss * mask).sum() / (mask.sum() + 1e-8)
                val_loss += loss.item()
                
                all_preds.append(final_pred.squeeze(-1)[mask])
                all_labels.append(sender_labels[mask])
        
        avg_val_loss = val_loss / (len(val_data) / batch_size)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(all_preds, all_labels)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_metrics = metrics
            best_model_state = model.state_dict().copy()
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    logger.info(f"Best Val Loss: {best_val_loss:.4f}")
    
    # Test Evaluation
    model.eval()
    test_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            messages = [c['messages'] for c in batch]
            senders = pad_sequence([c['speakers'] for c in batch], batch_first=True, padding_value=model.num_countries).to(device)
            receivers = pad_sequence([c['receivers'] for c in batch], batch_first=True, padding_value=model.num_countries).to(device)
            sender_labels = pad_sequence([c['sender_labels'] for c in batch], batch_first=True, padding_value=-1).to(device)
            lengths = [len(c['messages']) for c in batch]
            
            final_pred = model(messages, senders, receivers, lengths)
            loss = criterion(final_pred.squeeze(-1), sender_labels)
            mask = (sender_labels != -1)
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            test_loss += loss.item()
            
            all_preds.append(final_pred.squeeze(-1)[mask])
            all_labels.append(sender_labels[mask])
    
    avg_test_loss = test_loss / (len(test_data) / batch_size)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    test_metrics = compute_metrics(all_preds, all_labels)
    
    return {
        'training_loss': avg_train_loss,
        'validation_loss': avg_val_loss,
        'best_validation_loss': best_val_loss,
        'test_loss': avg_test_loss,
        'best_validation_metrics': best_metrics,
        'test_metrics': test_metrics
    }

# Main Execution
if __name__ == "__main__":
    train_file = "/kaggle/input/dataset-deception/train.jsonl"
    val_file = "/kaggle/input/dataset-deception/validation.jsonl"
    test_file = "/kaggle/input/dataset-deception/test.jsonl"
    
    train_data, country_map = preprocess_data(train_file)
    val_data, _ = preprocess_data(val_file, country_map)
    test_data, _ = preprocess_data(test_file, country_map)
    
    num_countries = len(country_map)
    model = DualLSTMDeceptionDetector(num_countries=num_countries, country_emb_dim=16, hidden_dim=128).to(device)
    logger.info(f"Model initialized with {num_countries} countries")
    
    results = train_and_evaluate(model, train_data, val_data, test_data, epochs=5, batch_size=4)
    
    print("\n=== Per-Class Metrics (for 'True' and 'False' classes) ===")
    print("Precision")
    print(f"True_precision: {results['test_metrics']['True_precision']:.4f}")
    print(f"False_precision: {results['test_metrics']['False_precision']:.4f}")
    print("Recall")
    print(f"True_recall: {results['test_metrics']['True_recall']:.4f}")
    print(f"False_recall: {results['test_metrics']['False_recall']:.4f}")
    print("F1-Score")
    print(f"True_fscore: {results['test_metrics']['True_fscore']:.4f}")
    print(f"False_fscore: {results['test_metrics']['False_fscore']:.4f}")
    
    print("\n📊 Micro-Averaged Metrics")
    print(f"micro_precision: {results['test_metrics']['micro_precision']:.4f}")
    print(f"micro_recall: {results['test_metrics']['micro_recall']:.4f}")
    print(f"micro_fscore: {results['test_metrics']['micro_fscore']:.4f}")
    
    print("\n📈 Macro-Averaged Metrics")
    print(f"macro_precision: {results['test_metrics']['macro_precision']:.4f}")
    print(f"macro_recall: {results['test_metrics']['macro_recall']:.4f}")
    print(f"macro_fscore: {results['test_metrics']['macro_fscore']:.4f}")
    
    print("\n🧠 Loss")
    print(f"training_loss: {results['training_loss']:.4f}")
    print(f"validation_loss: {results['validation_loss']:.4f}")
    print(f"best_validation_loss: {results['best_validation_loss']:.4f}")
    print(f"test_loss: {results['test_loss']:.4f}")
    
    print("\n🧪 Best Validation Metrics")
    for metric in ['True_precision', 'False_precision', 'True_recall', 'False_recall', 
                   'True_fscore', 'False_fscore', 'micro_precision', 'micro_recall', 
                   'micro_fscore', 'macro_precision', 'macro_recall', 'macro_fscore']:
        print(f"best_validation_{metric}: {results['best_validation_metrics'][metric]:.4f}")
