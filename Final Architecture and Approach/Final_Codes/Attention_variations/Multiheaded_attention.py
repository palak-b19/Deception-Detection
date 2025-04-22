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
    
    # Project to hidden dimension
    combined = self.input_projection(combined)
    combined = self.dropout(combined)
    
    # LSTM processing
    packed_input = pack_padded_sequence(combined, lengths, batch_first=True, enforce_sorted=False)
    lstm_out_packed, _ = self.lstm(packed_input)
    lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
    
    # Create attention mask for padded sequences
    max_len = max(lengths)
    attn_mask = torch.ones(batch_size, max_len, device=device).bool()
    for i, length in enumerate(lengths):
        attn_mask[i, :length] = False  # Valid positions are False, padded are True
    
    # Apply multi-headed attention (no transpose needed since batch_first=True)
    attn_output, attn_weights = self.multihead_attn(
        query=lstm_out,
        key=lstm_out,
        value=lstm_out,
        key_padding_mask=attn_mask
    )
    
    # Pool the attention output (e.g., mean pooling over sequence length, excluding padded positions)
    context = torch.zeros(batch_size, attn_output.size(2), device=device)
    for i in range(batch_size):
        valid_output = attn_output[i, :lengths[i]]  # Select non-padded positions
        context[i] = valid_output.mean(dim=0)       # Mean pooling
    
    # Normalize and classify
    context = self.layer_norm(context)
    final_pred = self.classifier(context).unsqueeze(1)
    
    # Expand predictions to match sequence length
    return final_pred.expand(-1, lstm_out.size(1), -1)
