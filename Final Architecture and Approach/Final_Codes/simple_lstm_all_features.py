
import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
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

# Load BERT tokenizer
logger.info("Loading BERT tokenizer...")
start_time = time.time()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
logger.info(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")

# Add [EMPTY] token
if '[EMPTY]' not in tokenizer.get_vocab():
    tokenizer.add_tokens(['[EMPTY]'])

# Model Definition
class SimpleDeceptionDetector(nn.Module):
    def __init__(self, vocab_size, num_countries, num_seasons, num_years, 
                 embedding_dim=100, country_emb_dim=16, season_emb_dim=8, year_emb_dim=8, hidden_dim=64):
        super(SimpleDeceptionDetector, self).__init__()
        # Word embedding for messages
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Embeddings for categorical features
        self.country_embedding = nn.Embedding(num_countries + 1, country_emb_dim, padding_idx=num_countries)
        self.season_embedding = nn.Embedding(num_seasons + 1, season_emb_dim, padding_idx=num_seasons)
        self.year_embedding = nn.Embedding(num_years + 1, year_emb_dim, padding_idx=num_years)
        # LSTM input: text + sender + receiver + season + year + game_score + game_score_delta
        input_dim = embedding_dim + country_emb_dim + country_emb_dim + season_emb_dim + year_emb_dim + 1 + 1
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, messages_list, senders, receivers, seasons, years, game_scores, game_score_deltas, lengths):
        batch_size = len(messages_list)
        # Generate message embeddings
        message_embs_list = []
        for batch_idx in range(batch_size):
            msgs = messages_list[batch_idx]
            embs = []
            for msg in msgs:
                if msg == '[EMPTY]':
                    emb = torch.zeros(self.word_embedding.embedding_dim, device=device)
                else:
                    inputs = tokenizer(msg, return_tensors='pt', padding=True, truncation=True, max_length=128)
                    token_ids = inputs['input_ids'].squeeze(0).to(device)
                    word_embs = self.word_embedding(token_ids)
                    attention_mask = (token_ids != 0).float()
                    if attention_mask.sum() > 0:
                        message_emb = word_embs.sum(dim=0) / attention_mask.sum()
                    else:
                        message_emb = torch.zeros(self.word_embedding.embedding_dim, device=device)
                    embs.append(message_emb)
            message_embs_list.append(torch.stack(embs))
        message_embs_padded = pad_sequence(message_embs_list, batch_first=True, padding_value=0)
        # Embed categorical and numerical features
        senders_emb = self.country_embedding(senders)
        receivers_emb = self.country_embedding(receivers)
        seasons_emb = self.season_embedding(seasons)
        years_emb = self.year_embedding(years)
        game_scores = game_scores.unsqueeze(-1)
        game_score_deltas = game_score_deltas.unsqueeze(-1)
        # Combine all features
        combined = torch.cat([message_embs_padded, senders_emb, receivers_emb, seasons_emb, years_emb, 
                             game_scores, game_score_deltas], dim=2)
        # LSTM processing
        combined_packed = pack_padded_sequence(combined, lengths, batch_first=True, enforce_sorted=False)
        lstm_out_packed, _ = self.lstm(combined_packed)
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
        # Classification
        final_pred = self.classifier(lstm_out)
        return final_pred

# Data Preprocessing
def preprocess_data(data_file, country_map=None, season_map=None, year_map=None):
    """Preprocess JSONL data into tensors with mappings for categorical features."""
    logger.info(f"Preprocessing {data_file}")
    with open(data_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Create mappings if not provided
    if country_map is None:
        countries = set()
        for game in data:
            countries.update(game['speakers'])
            countries.update(game['receivers'])
        country_map = {c: i for i, c in enumerate(sorted(countries))}
        logger.info(f"Created country map with {len(country_map)} countries")
    
    if season_map is None:
        seasons = set()
        for game in data:
            seasons.update(game['seasons'])
        season_map = {s: i for i, s in enumerate(sorted(seasons))}
        logger.info(f"Created season map with {len(season_map)} seasons")
    
    if year_map is None:
        years = set()
        for game in data:
            years.update(game['years'])
        year_map = {y: i for i, y in enumerate(sorted(years))}
        logger.info(f"Created year map with {len(year_map)} years")
    
    conversations = []
    for game in data:
        messages = [msg if msg else '[EMPTY]' for msg in game['messages']]
        senders = [country_map[s] for s in game['speakers']]
        receivers = [country_map[r] for r in game['receivers']]
        seasons = [season_map[s] for s in game['seasons']]
        years = [year_map[y] for y in game['years']]
        game_scores = [float(gs) for gs in game['game_score']]
        game_score_deltas = [float(gsd) for gsd in game['game_score_delta']]
        sender_labels = [1 if l else 0 for l in game['sender_labels']]
        receiver_labels = [-1 if l == "NOANNOTATION" else (1 if l else 0) for l in game['receiver_labels']]
        
        conversations.append({
            'messages': messages,
            'speakers': torch.tensor(senders, dtype=torch.long),
            'receivers': torch.tensor(receivers, dtype=torch.long),
            'seasons': torch.tensor(seasons, dtype=torch.long),
            'years': torch.tensor(years, dtype=torch.long),
            'game_scores': torch.tensor(game_scores, dtype=torch.float32),
            'game_score_deltas': torch.tensor(game_score_deltas, dtype=torch.float32),
            'sender_labels': torch.tensor(sender_labels, dtype=torch.float32),
            'receiver_labels': torch.tensor(receiver_labels, dtype=torch.float32)
        })
    
    # Filter out empty conversations
    original_count = len(conversations)
    conversations = [conv for conv in conversations if len(conv['messages']) > 0]
    logger.info(f"Filtered out {original_count - len(conversations)} empty conversations from {data_file}")
    
    return conversations, country_map, season_map, year_map

# Compute Metrics
def compute_metrics(preds, labels, mask=None):
    """Calculate precision, recall, and F1 scores."""
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
    """Train the model and evaluate on validation and test sets."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Compute pos_weight for class imbalance
    all_labels = [l.item() for conv in train_data for l in conv['sender_labels'] if l != -1]
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
            senders = pad_sequence([c['speakers'] for c in batch], batch_first=True, padding_value=model.country_embedding.padding_idx).to(device)
            receivers = pad_sequence([c['receivers'] for c in batch], batch_first=True, padding_value=model.country_embedding.padding_idx).to(device)
            seasons = pad_sequence([c['seasons'] for c in batch], batch_first=True, padding_value=model.season_embedding.padding_idx).to(device)
            years = pad_sequence([c['years'] for c in batch], batch_first=True, padding_value=model.year_embedding.padding_idx).to(device)
            game_scores = pad_sequence([c['game_scores'] for c in batch], batch_first=True, padding_value=0).to(device)
            game_score_deltas = pad_sequence([c['game_score_deltas'] for c in batch], batch_first=True, padding_value=0).to(device)
            sender_labels = pad_sequence([c['sender_labels'] for c in batch], batch_first=True, padding_value=-1).to(device)
            lengths = [len(c['messages']) for c in batch]
            
            optimizer.zero_grad()
            final_pred = model(messages, senders, receivers, seasons, years, game_scores, game_score_deltas, lengths)
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
                senders = pad_sequence([c['speakers'] for c in batch], batch_first=True, padding_value=model.country_embedding.padding_idx).to(device)
                receivers = pad_sequence([c['receivers'] for c in batch], batch_first=True, padding_value=model.country_embedding.padding_idx).to(device)
                seasons = pad_sequence([c['seasons'] for c in batch], batch_first=True, padding_value=model.season_embedding.padding_idx).to(device)
                years = pad_sequence([c['years'] for c in batch], batch_first=True, padding_value=model.year_embedding.padding_idx).to(device)
                game_scores = pad_sequence([c['game_scores'] for c in batch], batch_first=True, padding_value=0).to(device)
                game_score_deltas = pad_sequence([c['game_score_deltas'] for c in batch], batch_first=True, padding_value=0).to(device)
                sender_labels = pad_sequence([c['sender_labels'] for c in batch], batch_first=True, padding_value=-1).to(device)
                lengths = [len(c['messages']) for c in batch]
                
                final_pred = model(messages, senders, receivers, seasons, years, game_scores, game_score_deltas, lengths)
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
            senders = pad_sequence([c['speakers'] for c in batch], batch_first=True, padding_value=model.country_embedding.padding_idx).to(device)
            receivers = pad_sequence([c['receivers'] for c in batch], batch_first=True, padding_value=model.country_embedding.padding_idx).to(device)
            seasons = pad_sequence([c['seasons'] for c in batch], batch_first=True, padding_value=model.season_embedding.padding_idx).to(device)
            years = pad_sequence([c['years'] for c in batch], batch_first=True, padding_value=model.year_embedding.padding_idx).to(device)
            game_scores = pad_sequence([c['game_scores'] for c in batch], batch_first=True, padding_value=0).to(device)
            game_score_deltas = pad_sequence([c['game_score_deltas'] for c in batch], batch_first=True, padding_value=0).to(device)
            sender_labels = pad_sequence([c['sender_labels'] for c in batch], batch_first=True, padding_value=-1).to(device)
            lengths = [len(c['messages']) for c in batch]
            
            final_pred = model(messages, senders, receivers, seasons, years, game_scores, game_score_deltas, lengths)
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
    # File paths (adjust as needed)
    train_file = "/kaggle/input/dataset-deception/train.jsonl"
    val_file = "/kaggle/input/dataset-deception/validation.jsonl"
    test_file = "/kaggle/input/dataset-deception/test.jsonl"
    
    # Preprocess data
    train_data, country_map, season_map, year_map = preprocess_data(train_file)
    val_data, _, _, _ = preprocess_data(val_file, country_map, season_map, year_map)
    test_data, _, _, _ = preprocess_data(test_file, country_map, season_map, year_map)
    
    # Initialize model
    num_countries = len(country_map)
    num_seasons = len(season_map)
    num_years = len(year_map)
    vocab_size = len(tokenizer)
    model = SimpleDeceptionDetector(
        vocab_size=vocab_size,
        num_countries=num_countries, 
        num_seasons=num_seasons, 
        num_years=num_years, 
        embedding_dim=100, 
        country_emb_dim=16, 
        season_emb_dim=8, 
        year_emb_dim=8, 
        hidden_dim=64
    ).to(device)
    logger.info(f"Model initialized with vocab_size={vocab_size}, {num_countries} countries, {num_seasons} seasons, {num_years} years")
    
    # Train and evaluate
    results = train_and_evaluate(model, train_data, val_data, test_data, epochs=10, batch_size=4)
    
    # Print results
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

