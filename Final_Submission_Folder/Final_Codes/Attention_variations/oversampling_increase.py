import torch.nn.functional as F

class EnhancedDeceptionDetector(nn.Module):
    def __init__(self, num_countries, num_seasons, num_years, 
                 sentence_emb_dim=384, ling_feature_dim=6, country_emb_dim=16, 
                 season_emb_dim=8, year_emb_dim=8, hidden_dim=64, dropout=0.4, 
                 max_seq_len=100, pos_enc_dim=64):
        super(EnhancedDeceptionDetector, self).__init__()
        self.ling_feature_dim = ling_feature_dim
        self.hidden_dim = hidden_dim
        self.pos_enc_dim = pos_enc_dim
        
        # Embeddings
        self.country_embedding = nn.Embedding(num_countries + 1, country_emb_dim, padding_idx=num_countries)
        self.season_embedding = nn.Embedding(num_seasons + 1, season_emb_dim, padding_idx=num_seasons)
        self.year_embedding = nn.Embedding(num_years + 1, year_emb_dim, padding_idx=num_years)
        
        # Positional Encoding
        self.max_seq_len = max_seq_len
        self.pos_enc_dim = min(pos_enc_dim, hidden_dim)
        self.positional_encoding = self._init_positional_encoding(max_seq_len, self.pos_enc_dim)
        
        # Input dimension
        input_dim = sentence_emb_dim + ling_feature_dim + (country_emb_dim * 2) + season_emb_dim + year_emb_dim + 2
        self.ling_projection = nn.Linear(ling_feature_dim, ling_feature_dim) if ling_feature_dim > 0 else None
        self.input_projection = nn.Linear(input_dim + self.pos_enc_dim, hidden_dim)
        
        # LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        
        # Self-Attention
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Classifier (applied to each timestep)
        self.classifier = nn.Linear(hidden_dim * 2, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _init_positional_encoding(self, max_seq_len, pos_enc_dim):
        """Initialize fixed sinusoidal positional encodings."""
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, pos_enc_dim, 2).float() * (-np.log(10000.0) / pos_enc_dim))
        pos_enc = torch.zeros(max_seq_len, pos_enc_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        return pos_enc.to(device)
    
    def forward(self, messages_list, ling_features, senders, receivers, seasons, years, 
                game_scores, game_score_deltas, lengths):
        batch_size = len(messages_list)
        
        # Generate message embeddings
        message_embs_list = []
        for batch_idx in range(batch_size):
            msgs = messages_list[batch_idx]
            msgs = ['' if msg == '[EMPTY]' else msg for msg in msgs]
            with torch.no_grad():
                embs = sentence_bert.encode(
                    msgs, 
                    convert_to_tensor=True, 
                    device=device, 
                    show_progress_bar=False,
                    batch_size=32
                )
            message_embs_list.append(embs)
        
        # Pad message embeddings
        message_embs_padded = pad_sequence(message_embs_list, batch_first=True, padding_value=0)
        
        # Pad linguistic features
        if self.ling_feature_dim > 0:
            ling_features_padded = pad_sequence(ling_features, batch_first=True, padding_value=0).to(device)
            ling_features_padded = self.ling_projection(ling_features_padded)
        else:
            ling_features_padded = torch.zeros(batch_size, message_embs_padded.size(1), 0, device=device)
        
        # Embed categorical and numerical features
        senders_emb = self.country_embedding(senders)
        receivers_emb = self.country_embedding(receivers)
        seasons_emb = self.season_embedding(seasons)
        years_emb = self.year_embedding(years)
        game_scores = game_scores.unsqueeze(-1)
        game_score_deltas = game_score_deltas.unsqueeze(-1)
        
        # Combine features
        combined = torch.cat([
            message_embs_padded, 
            ling_features_padded,
            senders_emb, 
            receivers_emb, 
            seasons_emb, 
            years_emb, 
            game_scores, 
            game_score_deltas
        ], dim=2)
        
        # Add positional encoding and adjust lengths if necessary
        seq_len = combined.size(1)
        lengths_tensor = torch.tensor(lengths, device=device, dtype=torch.long)
        if seq_len > self.max_seq_len:
            logger.warning(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}. Truncating.")
            combined = combined[:, :self.max_seq_len, :]
            seq_len = self.max_seq_len
            lengths_tensor = torch.clamp(lengths_tensor, max=self.max_seq_len)
        pos_enc = self.positional_encoding[:, :seq_len, :].expand(batch_size, -1, -1)
        if self.pos_enc_dim < self.hidden_dim:
            pos_enc = torch.cat([pos_enc, torch.zeros(batch_size, seq_len, self.hidden_dim - self.pos_enc_dim, device=device)], dim=2)
        combined = torch.cat([combined, pos_enc], dim=2)
        
        # Project to hidden dimension
        combined = self.input_projection(combined)
        combined = self.dropout(combined)
        
        # LSTM processing
        packed_input = pack_padded_sequence(combined, lengths_tensor.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out_packed, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True, total_length=seq_len)
        
        # Normalize LSTM output
        lstm_out = self.layer_norm(lstm_out)  # Shape: [batch_size, seq_len, hidden_dim * 2]
        
        # Self-Attention
        # Compute attention scores
        attn_scores = self.attention(lstm_out)  # Shape: [batch_size, seq_len, 1]
        # Create padding mask
        mask = torch.arange(seq_len, device=device)[None, :] >= lengths_tensor[:, None]  # Shape: [batch_size, seq_len]
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), float('-inf'))  # Shape: [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # Shape: [batch_size, seq_len, 1]
        # Apply attention weights to LSTM output
        attn_out = lstm_out * attn_weights  # Shape: [batch_size, seq_len, hidden_dim * 2]
        
        # Normalize attention output
        attn_out = self.layer_norm(attn_out)  # Shape: [batch_size, seq_len, hidden_dim * 2]
        
        # Classify
        final_pred = self.classifier(attn_out)  # Shape: [batch_size, seq_len, 1]
        
        return final_pred

def preprocess_data(data_file, country_map=None, season_map=None, year_map=None, use_ling_features=True, ling_stats=None):
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
    ling_features_all = [] if use_ling_features else None
    for game in tqdm(data, desc="Processing conversations"):
        messages = [msg if msg else '[EMPTY]' for msg in game['messages']]
        if len(messages) == 0:
            continue  # Skip empty conversations
        senders = [country_map[s] for s in game['speakers']]
        receivers = [country_map[r] for r in game['receivers']]
        seasons = [season_map[s] for s in game['seasons']]
        years = [year_map[s] for s in game['years']]
        game_scores = [float(gs) for gs in game['game_score']]
        game_score_deltas = [float(gsd) for gsd in game['game_score_delta']]
        sender_labels = [1 if l else 0 for l in game['sender_labels']]
        receiver_labels = [-1 if l == "NOANNOTATION" else (1 if l else 0) for l in game['receiver_labels']]
        
        # Compute linguistic features
        ling_features = []
        for msg in messages:
            if msg == '[EMPTY]' or not use_ling_features:
                ling_features.append([0.0] * 6)
            else:
                doc = nlp(msg)
                words = [token.text.lower() for token in doc if not token.is_punct]
                word_count = min(len(words), 100)
                lexical_diversity = len(set(words)) / (len(words) + 1e-8) if len(words) > 0 else 0
                lexical_diversity = min(lexical_diversity, 1.0)
                self_refs = min(sum(1 for token in doc if token.text.lower() in ['i', 'me', 'my', 'mine', 'myself']), 10)
                group_refs = min(sum(1 for token in doc if token.text.lower() in ['we', 'us', 'our', 'ours', 'ourselves']), 10)
                neg_emotion = sid.polarity_scores(msg)['neg']
                modifiers = min(sum(1 for token in doc if token.pos_ in ['ADJ', 'ADV']), 10)
                ling_features.append([
                    np.log1p(word_count),
                    lexical_diversity,
                    self_refs,
                    group_refs,
                    neg_emotion,
                    modifiers
                ])
            if data_file.endswith('train.jsonl'):
                ling_features_all.append(ling_features[-1])
        
        conversations.append({
            'messages': messages,
            'ling_features': torch.tensor(ling_features, dtype=torch.float32),
            'speakers': torch.tensor(senders, dtype=torch.long),
            'receivers': torch.tensor(receivers, dtype=torch.long),
            'seasons': torch.tensor(seasons, dtype=torch.long),
            'years': torch.tensor(years, dtype=torch.long),
            'game_scores': torch.tensor(game_scores, dtype=torch.float32),
            'game_score_deltas': torch.tensor(game_score_deltas, dtype=torch.float32),
            'sender_labels': torch.tensor(sender_labels, dtype=torch.float32),
            'receiver_labels': torch.tensor(receiver_labels, dtype=torch.float32)
        })
    
    # Compute linguistic feature statistics for train data
    if use_ling_features and data_file.endswith('train.jsonl'):
        ling_features_all = np.array(ling_features_all)  # Shape: (num_messages, 6)
        ling_stats = {
            'mean': np.mean(ling_features_all, axis=0),  # Shape: (6,)
            'std': np.std(ling_features_all, axis=0) + 1e-8  # Shape: (6,)
        }
    elif use_ling_features:
        # For val/test, ling_stats should be passed from train
        if ling_stats is None:
            raise ValueError("ling_stats must be provided for val/test when use_ling_features=True")
    
    # Normalize linguistic features
    if use_ling_features:
        for conv in conversations:
            ling_features = conv['ling_features'].numpy()
            ling_features = (ling_features - ling_stats['mean']) / ling_stats['std']
            conv['ling_features'] = torch.tensor(ling_features, dtype=torch.float32)
    
    logger.info(f"Processed {len(conversations)} conversations from {data_file}")
    return conversations, country_map, season_map, year_map, ling_stats

def oversample_lies(data, oversample_factor=3):
    lie_convs = [conv for conv in data if any(l == 1 for l in conv['sender_labels'])]
    oversampled = data + lie_convs * (oversample_factor - 1)
    logger.info(f"Oversampled: {len(data)} -> {len(oversampled)} conversations")
    return oversampled

def compute_metrics(preds, labels, mask=None):
    if mask is not None:
        preds = preds[mask.bool()]
        labels = labels[mask.bool()]
    preds = (preds > 0).float()
    labels = labels.float()
    
    precision, recall, fscore, _ = precision_recall_fscore_support(labels.cpu(), preds.cpu(), average=None, labels=[0, 1], zero_division=0)
    micro = precision_recall_fscore_support(labels.cpu(), preds.cpu(), average='micro', zero_division=0)[:3]
    macro = precision_recall_fscore_support(labels.cpu(), preds.cpu(), average='macro', zero_division=0)[:3]
    conf_matrix = confusion_matrix(labels.cpu(), preds.cpu(), labels=[0, 1])
    
    return {
        'True_precision': precision[1], 'False_precision': precision[0],
        'True_recall': recall[1], 'False_recall': recall[0],
        'True_fscore': fscore[1], 'False_fscore': fscore[0],
        'micro_precision': micro[0], 'micro_recall': micro[1], 'micro_fscore': micro[2],
        'macro_precision': macro[0], 'macro_recall': macro[1], 'macro_fscore': macro[2],
        'confusion_matrix': conf_matrix
    }

def train_and_evaluate(model, train_data, val_data, test_data, epochs=15, batch_size=4, lr=0.0001, patience=3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate warmup
    warmup_steps = int(0.1 * epochs * (len(train_data) / batch_size))
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / warmup_steps
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Compute initial pos_weight
    all_labels = [l.item() for conv in train_data for l in conv['sender_labels'] if l != -1]
    num_lies = sum(l == 1 for l in all_labels)
    num_truths = len(all_labels) - num_lies
    pos_weight_val = (num_truths / num_lies) * 1.5 if num_lies > 0 else 1.0
    logger.info(f"Class stats - Lies: {num_lies}, Truths: {num_truths}, initial pos_weight: {pos_weight_val:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device), reduction='none')
    
    # Initialize classifier bias
    model.classifier.bias.data.fill_(1.0)
    
    best_val_loss = float('inf')
    best_metrics = None
    best_model_state = None
    epochs_no_improve = 0
    best_pos_weight = pos_weight_val
    max_seq_len = model.max_seq_len  # Access model's max_seq_len
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_lies = 0
        batch_total = 0
        for i in tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch = train_data[i:i+batch_size]
            messages = [c['messages'] for c in batch]
            ling_features = [c['ling_features'] for c in batch]
            senders = pad_sequence([c['speakers'] for c in batch], batch_first=True, padding_value=model.country_embedding.padding_idx).to(device)
            receivers = pad_sequence([c['receivers'] for c in batch], batch_first=True, padding_value=model.country_embedding.padding_idx).to(device)
            seasons = pad_sequence([c['seasons'] for c in batch], batch_first=True, padding_value=model.season_embedding.padding_idx).to(device)
            years = pad_sequence([c['years'] for c in batch], batch_first=True, padding_value=model.year_embedding.padding_idx).to(device)
            game_scores = pad_sequence([c['game_scores'] for c in batch], batch_first=True, padding_value=0).to(device)
            game_score_deltas = pad_sequence([c['game_score_deltas'] for c in batch], batch_first=True, padding_value=0).to(device)
            sender_labels = pad_sequence([c['sender_labels'] for c in batch], batch_first=True, padding_value=-1).to(device)
            lengths = [len(c['messages']) for c in batch]
            
            # Truncate sender_labels to match model's max_seq_len
            if sender_labels.size(1) > max_seq_len:
                sender_labels = sender_labels[:, :max_seq_len]
            
            # Track batch stats
            mask = (sender_labels != -1)
            batch_lies += (sender_labels[mask] == 1).sum().item()
            batch_total += mask.sum().item()
            
            optimizer.zero_grad()
            final_pred = model(messages, ling_features, senders, receivers, seasons, years, 
                              game_scores, game_score_deltas, lengths)
            loss = criterion(final_pred.squeeze(-1), sender_labels)
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / (len(train_data) / batch_size)
        batch_lie_ratio = batch_lies / (batch_total + 1e-8)
        logger.info(f"Epoch {epoch+1}/{epochs} - Batch lie ratio: {batch_lie_ratio:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                messages = [c['messages'] for c in batch]
                ling_features = [c['ling_features'] for c in batch]
                senders = pad_sequence([c['speakers'] for c in batch], batch_first=True, padding_value=model.country_embedding.padding_idx).to(device)
                receivers = pad_sequence([c['receivers'] for c in batch], batch_first=True, padding_value=model.country_embedding.padding_idx).to(device)
                seasons = pad_sequence([c['seasons'] for c in batch], batch_first=True, padding_value=model.season_embedding.padding_idx).to(device)
                years = pad_sequence([c['years'] for c in batch], batch_first=True, padding_value=model.year_embedding.padding_idx).to(device)
                game_scores = pad_sequence([c['game_scores'] for c in batch], batch_first=True, padding_value=0).to(device)
                game_score_deltas = pad_sequence([c['game_score_deltas'] for c in batch], batch_first=True, padding_value=0).to(device)
                sender_labels = pad_sequence([c['sender_labels'] for c in batch], batch_first=True, padding_value=-1).to(device)
                lengths = [len(c['messages']) for c in batch]
                
                # Truncate sender_labels to match model's max_seq_len
                if sender_labels.size(1) > max_seq_len:
                    sender_labels = sender_labels[:, :max_seq_len]
                
                final_pred = model(messages, ling_features, senders, receivers, seasons, years, 
                                  game_scores, game_score_deltas, lengths)
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
        
        # Log lie predictions
        predicted_lies = (all_preds > 0).sum().item()
        avg_prob = torch.sigmoid(all_preds).mean().item()
        logger.info(f"Epoch {epoch+1}/{epochs} - Predicted lies: {predicted_lies}, Avg prob: {avg_prob:.4f}")
        logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        
        # Adjust pos_weight based on lie recall
        if metrics['True_recall'] < 0.3 and pos_weight_val < 10.0:
            pos_weight_val *= 1.2
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device), reduction='none')
            logger.info(f"Increased pos_weight to {pos_weight_val:.2f}")
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                    f"True F1: {metrics['True_fscore']:.4f}, Macro F1: {metrics['macro_fscore']:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_metrics = metrics
            best_model_state = model.state_dict().copy()
            best_pos_weight = pos_weight_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([best_pos_weight], device=device), reduction='none')
    logger.info(f"Best Val Loss: {best_val_loss:.4f}, Best pos_weight: {best_pos_weight:.2f}")
    
    # Test Evaluation
    model.eval()
    test_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            messages = [c['messages'] for c in batch]
            ling_features = [c['ling_features'] for c in batch]
            senders = pad_sequence([c['speakers'] for c in batch], batch_first=True, padding_value=model.country_embedding.padding_idx).to(device)
            receivers = pad_sequence([c['receivers'] for c in batch], batch_first=True, padding_value=model.country_embedding.padding_idx).to(device)
            seasons = pad_sequence([c['seasons'] for c in batch], batch_first=True, padding_value=model.season_embedding.padding_idx).to(device)
            years = pad_sequence([c['years'] for c in batch], batch_first=True, padding_value=model.year_embedding.padding_idx).to(device)
            game_scores = pad_sequence([c['game_scores'] for c in batch], batch_first=True, padding_value=0).to(device)
            game_score_deltas = pad_sequence([c['game_score_deltas'] for c in batch], batch_first=True, padding_value=0).to(device)
            sender_labels = pad_sequence([c['sender_labels'] for c in batch], batch_first=True, padding_value=-1).to(device)
            lengths = [len(c['messages']) for c in batch]
            
            # Truncate sender_labels to match model's max_seq_len
            if sender_labels.size(1) > max_seq_len:
                sender_labels = sender_labels[:, :max_seq_len]
            
            final_pred = model(messages, ling_features, senders, receivers, seasons, years, 
                              game_scores, game_score_deltas, lengths)
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
    
    # Log test lie predictions
    predicted_lies = (all_preds > 0).sum().item()
    avg_prob = torch.sigmoid(all_preds).mean().item()
    logger.info(f"Test - Predicted lies: {predicted_lies}, Avg prob: {avg_prob:.4f}")
    logger.info(f"Test Confusion Matrix:\n{test_metrics['confusion_matrix']}")
    
    return {
        'training_loss': avg_train_loss,
        'validation_loss': avg_val_loss,
        'best_validation_loss': best_val_loss,
        'test_loss': avg_test_loss,
        'best_validation_metrics': best_metrics,
        'test_metrics': test_metrics,
        'best_pos_weight': best_pos_weight
    }

def grid_search_hyperparams(train_data, val_data, test_data, num_countries, num_seasons, num_years, 
                           use_ling_features=True, epochs=15, patience=3):
    param_grid = {
        'lr': [0.00005, 0.0001, 0.0002],
        'batch_size': [4, 8]
    }
    best_macro_f1 = 0
    best_params = None
    best_results = None
    
    for lr, batch_size in itertools.product(param_grid['lr'], param_grid['batch_size']):
        logger.info(f"Testing lr={lr}, batch_size={batch_size}")
        model = EnhancedDeceptionDetector(
            num_countries=num_countries, 
            num_seasons=num_seasons, 
            num_years=num_years, 
            sentence_emb_dim=384, 
            ling_feature_dim=6 if use_ling_features else 0,
            country_emb_dim=16, 
            season_emb_dim=8, 
            year_emb_dim=8, 
            hidden_dim=64, 
            dropout=0.4
        ).to(device)
        
        results = train_and_evaluate(model, train_data, val_data, test_data, 
                                     epochs=epochs, batch_size=batch_size, lr=lr, patience=patience)
        
        macro_f1 = results['best_validation_metrics']['macro_fscore']
        logger.info(f"lr={lr}, batch_size={batch_size}, Validation Macro F1: {macro_f1:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_params = {'lr': lr, 'batch_size': batch_size}
            best_results = results
    
    return best_params, best_results

if __name__ == "__main__":
    # File paths
    train_file = "/kaggle/input/dataset-deception/train.jsonl"
    val_file = "/kaggle/input/dataset-deception/validation.jsonl"
    test_file = "/kaggle/input/dataset-deception/test.jsonl"
    
    # Preprocess train data first to get ling_stats
    use_ling_features = True
    train_data, country_map, season_map, year_map, ling_stats = preprocess_data(train_file, use_ling_features=use_ling_features)
    
    # Preprocess val and test with train's ling_stats
    val_data, _, _, _, _ = preprocess_data(val_file, country_map, season_map, year_map, use_ling_features=use_ling_features, ling_stats=ling_stats)
    test_data, _, _, _, _ = preprocess_data(test_file, country_map, season_map, year_map, use_ling_features=use_ling_features, ling_stats=ling_stats)
    
    # Oversample lie conversations
    train_data = oversample_lies(train_data, oversample_factor=3)
    
    # Grid search
    num_countries = len(country_map)
    num_seasons = len(season_map)
    num_years = len(year_map)
    best_params, results = grid_search_hyperparams(
        train_data, val_data, test_data, num_countries, num_seasons, num_years, 
        use_ling_features=use_ling_features, epochs=15, patience=3
    )
    
    logger.info(f"Best parameters: lr={best_params['lr']}, batch_size={best_params['batch_size']}")
    
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
    print(f"test_loss: {results['test_loss']:.4f}")  # Fixed typo in key
    
    print("\n🧪 Best Validation Metrics")
    for metric in ['True_precision', 'False_precision', 'True_recall', 'False_recall', 
                   'True_fscore', 'False_fscore', 'micro_precision', 'micro_recall', 
                   'micro_fscore', 'macro_precision', 'macro_recall', 'macro_fscore']:
        print(f"best_validation_{metric}: {results['best_validation_metrics'][metric]:.4f}")
    
    print("\n🔍 Test Confusion Matrix")
    print(results['test_metrics']['confusion_matrix'])
