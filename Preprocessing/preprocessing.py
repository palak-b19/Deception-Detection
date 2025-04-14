import json
import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
# import dgl
from sklearn.preprocessing import LabelEncoder

# Paths to dataset files (adjust as needed)
DATA_DIR = "/kaggle/input/dataset-deception"
MOVES_DIR = "/kaggle/input/dataset-deception"
LEXICON_PATH = "utils/2015_Diplomacy_lexicon.json"

def load_diplomacy_data(jsonl_path, moves_dir):
    """Load and preprocess train.jsonl and moves data."""
    # Load JSONLines file
    dialogues = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            dialogues.append(json.loads(line))
    
    # Flatten dialogues to message-level DataFrame
    messages = []
    for dlg in dialogues:
        for i in range(len(dlg['messages'])):
            messages.append({
                'game_id': dlg['game_id'],
                'message': dlg['messages'][i],
                'speaker': dlg['speakers'][i],
                'receiver': dlg['receivers'][i],
                'sender_label': dlg['sender_labels'][i],
                'receiver_label': dlg['receiver_labels'][i],
                'game_score': int(dlg['game_score'][i]),
                'score_delta': int(dlg['game_score_delta'][i]),
                'season': dlg['seasons'][i],
                'year': int(dlg['years'][i]),
                'abs_msg_idx': dlg['absolute_message_index'][i],
                'rel_msg_idx': dlg['relative_message_index'][i]
            })
    
    df = pd.DataFrame(messages)
    
    # Load moves data
    moves_data = {}
    for game_id in df['game_id'].unique():
        moves_data[game_id] = {}
        for season in ['Spring', 'Fall', 'Winter']:
            for year in range(1901, 1911):  # Adjust based on dataset
                move_file = os.path.join(moves_dir, f"DiplomacyGame{game_id}_{year}_{season.lower()}.json")
                if os.path.exists(move_file):
                    with open(move_file, 'r') as f:
                        moves_data[game_id][f"{season}_{year}"] = json.load(f)
    
    # Encode labels
    df['actual_lie'] = df['sender_label'].astype(int)  # True=1, False=0
    df['suspected_lie'] = df['receiver_label'].map({'true': 1, 'false': 0, 'NOANNOTATION': -1})
    
    return df, moves_data

# Example usage
train_df, moves_data = load_diplomacy_data(os.path.join(DATA_DIR, "train.jsonl"), MOVES_DIR)
print(train_df.head())

import json
import re
from collections import Counter
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def robust_load_jsonl(file_path):
    """Load JSON Lines with error handling."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping invalid line in {file_path}")
                continue
    return pd.DataFrame(data)

def flatten_dialogues(df):
    """Flatten nested dialogue lists with validation."""
    messages = []
    for idx, row in df.iterrows():
        n = len(row['messages'])
        # Validate list lengths
        keys = ['sender_labels', 'receiver_labels', 'speakers', 'receivers', 
                'game_score', 'game_score_delta', 'seasons', 'years', 
                'absolute_message_index', 'relative_message_index']
        if not all(len(row[k]) == n for k in keys):
            print(f"Skipping dialogue {row['game_id']} at index {idx}: length mismatch")
            continue
        for i in range(n):
            messages.append({
                'game_id': row['game_id'],
                'message': row['messages'][i],
                'speaker': row['speakers'][i],
                'receiver': row['receivers'][i],
                'actual_lie': 1 if row['sender_labels'][i] else 0,
                'suspected_lie': -1 if row['receiver_labels'][i] == 'NOANNOTATION' else (1 if row['receiver_labels'][i] == 'true' else 0),
                'game_score': int(row['game_score'][i]),
                'game_score_delta': int(row['game_score_delta'][i]),
                'season': row['seasons'][i],
                'year': int(row['years'][i]),
                'abs_msg_idx': row['absolute_message_index'][i],
                'rel_msg_idx': row['relative_message_index'][i]
            })
    df_flat = pd.DataFrame(messages)
    # Normalize numeric features
    scaler = MinMaxScaler()
    df_flat[['game_score', 'game_score_delta', 'year']] = scaler.fit_transform(
        df_flat[['game_score', 'game_score_delta', 'year']]
    )
    return df_flat

def load_lexicon(lexicon_path):
    """Load Diplomacy lexicon."""
    with open(lexicon_path, 'r') as f:
        lexicon = json.load(f)
    return set(lexicon)

def build_custom_tokenizer(df, lexicon_path, min_freq=2, max_vocab=10000):
    """Build tokenizer with improved handling of Diplomacy terms."""
    lexicon = load_lexicon(lexicon_path)
    
    def tokenizer(text):
        # Preserve contractions and multi-word lexicon terms
        text = text.lower()
        tokens = []
        # Check for lexicon phrases first (e.g., "black sea")
        for term in lexicon:
            if term.lower() in text:
                tokens.append(term.lower())
                text = text.replace(term.lower(), ' ')
        # Tokenize remaining words
        tokens.extend(re.findall(r'\b\w+\'?\w*\b', text))
        return [t for t in tokens if t]  # Remove empty
    
    # Count tokens
    token_counts = Counter()
    for text in df['message']:
        token_counts.update(tokenizer(text))
    
    # Build vocab
    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    for token, count in token_counts.items():
        if count >= min_freq and idx < max_vocab:
            vocab[token] = idx
            idx += 1
    # Add lexicon terms with lower min_freq
    for term in lexicon:
        if term.lower() not in vocab and idx < max_vocab:
            vocab[term.lower()] = idx
            idx += 1
    
    def encode(text):
        tokens = tokenizer(text)
        return [vocab.get(t, vocab['<unk>']) for t in tokens]
    
    return vocab, encode, tokenizer

# Load and flatten data
jsonl_path = '/kaggle/input/dataset-deception/train.jsonl'
lexicon_path = '/kaggle/input/dataset-deception/2015_Diplomacy_lexicon.json'
try:
    train_df = pd.read_json(jsonl_path, lines=True, encoding='utf-8')
except ValueError:
    train_df = robust_load_jsonl(jsonl_path)
train_df = flatten_dialogues(train_df)

# Build tokenizer
vocab, encode, tokenizer = build_custom_tokenizer(train_df, lexicon_path, min_freq=1)
print(f"Vocab size: {len(vocab)}")


def load_all_data(data_dir, moves_dir):
    """Load train, val, test, and moves data."""
    splits = {}
    for split in ['train', 'validation', 'test']:
        path = os.path.join(data_dir, f"{split}.jsonl")
        if os.path.exists(path):
            try:
                df = pd.read_json(path, lines=True, encoding='utf-8')
            except ValueError:
                df = robust_load_jsonl(path)
            splits[split] = flatten_dialogues(df)
    
    # Load moves data
    moves_data = {}
    for game_id in set(splits['train']['game_id']).union(splits.get('validation', pd.DataFrame())['game_id'], splits.get('test', pd.DataFrame())['game_id']):
        moves_data[game_id] = {}
        for season in ['Spring', 'Fall', 'Winter']:
            for year in range(1901, 1911):
                move_file = os.path.join(moves_dir, f"DiplomacyGame{game_id}_{year}_{season.lower()}.json")
                if os.path.exists(move_file):
                    with open(move_file, 'r') as f:
                        moves_data[game_id][f"{season}_{year}"] = json.load(f)
    
    return splits, moves_data

# Load data
data_dir = '/kaggle/input/dataset-deception'
moves_dir = '/kaggle/input/dataset-deception/moves'
splits, moves_data = load_all_data(data_dir, moves_dir)
train_df, val_df, test_df = splits['train'], splits.get('validation', pd.DataFrame()), splits.get('test', pd.DataFrame())

def compute_move_features(row, moves_data, game_locations):
    """Compute move-aware features."""
    game_id = row['game_id']
    season_year = f"{row['season']}_{row['year']}"
    speaker = row['speaker']
    message = row['message'].lower()
    
    move_info = moves_data.get(game_id, {}).get(season_year, {})
    moves = move_info.get('moves', {})
    units = move_info.get('units', {})
    
    # Contradiction
    contradiction = 0
    if 'support' in message and any(loc.lower() in message for loc in game_locations):
        mentioned_loc = next((loc for loc in game_locations if loc.lower() in message), None)
        if mentioned_loc and speaker in moves and 'support' not in moves[speaker].get(mentioned_loc.lower(), '').lower():
            contradiction = 1
    
    # Unit Proximity
    proximity = 0
    mentioned_locs = [loc for loc in game_locations if loc.lower() in message]
    if mentioned_locs:
        speaker_units = units.get(speaker, {})
        for loc in mentioned_locs:
            if loc in speaker_units:
                proximity = max(proximity, 1.0)
            elif loc in game_locations:  # Simplified adjacency
                proximity = max(proximity, 0.5)
    
    # Alliance Shift
    alliance_shift = 0
    prev_season = f"{'Spring' if row['season'] == 'Fall' else 'Fall'}_{row['year']-1 if row['season'] == 'Spring' else row['year']}"
    prev_score = moves_data.get(game_id, {}).get(prev_season, {}).get('game_score', {}).get(speaker, row['game_score'])
    alliance_shift = row['game_score'] - prev_score
    
    return np.array([contradiction, proximity, alliance_shift])

# Simplified game locations
game_locations = {'Munich', 'Paris', 'Berlin', 'Tyrolia', 'Black Sea', 'Smyrna'}

# Apply to splits
for split_df in [train_df, val_df, test_df]:
    if not split_df.empty:
        split_df['move_features'] = split_df.apply(
            lambda row: compute_move_features(row, moves_data, game_locations), axis=1
        )

def verify_splits(train_df, val_df, test_df):
    """Ensure non-overlapping game IDs."""
    train_games = set(train_df['game_id'])
    val_games = set(val_df['game_id']) if not val_df.empty else set()
    test_games = set(test_df['game_id']) if not test_df.empty else set()
    print(f"Train games: {train_games}, Val games: {val_games}, Test games: {test_games}")
    assert train_games.isdisjoint(val_games) and train_games.isdisjoint(test_games), "Game ID overlap detected"
    return train_df, val_df, test_df

train_df, val_df, test_df = verify_splits(train_df, val_df, test_df)
print(f"Encoded: {encode('I’ll support your move to Munich')}")
