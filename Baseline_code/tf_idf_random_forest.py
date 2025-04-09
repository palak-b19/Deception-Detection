
import jsonlines
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
import numpy as np
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import sys
import warnings
warnings.filterwarnings("ignore")

# Check if token is a number
def is_number(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False

# Spacy tokenizer (same as n-gram)
def spacy_tokenizer(text):
    return [tok.text if not is_number(tok.text) else '_NUM_' for tok in nlp(text)]

# Aggregate dialogs into a single list (same as n-gram)
def aggregate(dataset):
    messages = []
    rec = []
    send = []
    power = []
    for dialogs in dataset:
        messages.extend(dialogs['messages'])
        rec.extend(dialogs['receiver_labels'])
        send.extend(dialogs['sender_labels'])
        power.extend(dialogs['game_score_delta'])
    merged = []
    for i, item in enumerate(messages):
        merged.append({'message': item, 'sender_annotation': send[i], 'receiver_annotation': rec[i], 'score_delta': int(power[i])})
    return merged

# Convert to binary features (power and labels)
def convert_to_binary(dataset):
    binary_data = []
    for message in dataset:
        if message['receiver_annotation'] not in [True, False]:
            if TASK == "SENDER":
                pass
            elif TASK == "RECEIVER":
                continue
        binary = []
        if POWER == "y":
            binary.append(1 if message['score_delta'] > 4 else 0)
            binary.append(1 if message['score_delta'] < -4 else 0)
        annotation = 'sender_annotation' if TASK == "SENDER" else 'receiver_annotation'
        binary.append(0 if message[annotation] == False else 1)
        binary_data.append(binary)
    return binary_data

# Split X and y
def split_xy(data):
    X, y = [], []
    for line in data:
        x = line[:len(line)-1]
        single_y = line[len(line)-1]
        X.append(x)
        y.append(single_y)
    return X, y

# TF-IDF + Random Forest
def tfidf_rf(train, test):
    # Convert train data with TF-IDF
    vectorizer = TfidfVectorizer(
        tokenizer=spacy_tokenizer,
        stop_words=STOP_WORDS,
        strip_accents='unicode',
        ngram_range=(1, 2),  # Unigrams and bigrams (adjustable)
        max_features=50000   # Limit features to avoid memory issues
    )
    if TASK == "SENDER":
        corpus = [message['message'].lower() for message in aggregate(train)]
    elif TASK == "RECEIVER":
        corpus = [message['message'].lower() for message in aggregate(train) if message['receiver_annotation'] != "NOANNOTATION"]
    X = vectorizer.fit_transform(corpus)

    # Convert test data with same vocabulary
    new_vec = TfidfVectorizer(
        tokenizer=spacy_tokenizer,
        vocabulary=vectorizer.vocabulary_,
        stop_words=STOP_WORDS,
        strip_accents='unicode',
        ngram_range=(1, 2),
        max_features=50000
    )
    if TASK == "SENDER":
        test_corpus = [message['message'].lower() for message in aggregate(test)]
    elif TASK == "RECEIVER":
        test_corpus = [message['message'].lower() for message in aggregate(test) if message['receiver_annotation'] != "NOANNOTATION"]
    y = new_vec.fit_transform(test_corpus)

    # Get binary features
    train_binary = convert_to_binary(aggregate(train))
    test_binary = convert_to_binary(aggregate(test))
    train_X, train_y = split_xy(train_binary)
    test_X, test_y = split_xy(test_binary)

    # Convert power features to sparse matrices
    train_X_sparse = csr_matrix(train_X)
    test_X_sparse = csr_matrix(test_X)

    # Append power features using sparse hstack
    X_combined = hstack([X, train_X_sparse])
    y_combined = hstack([y, test_X_sparse])

    # Train Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,    # Number of trees
        max_depth=20,        # Limit depth to control memory/computation
        class_weight='balanced',
        n_jobs=-1,           # Use all available cores
        random_state=1994    # Consistent with your seeds
    )
    rf_model.fit(X_combined, train_y)
    predictions = rf_model.predict(y_combined)

    print(classification_report(test_y, predictions, digits=3))

if __name__ == '__main__':
    # Parse arguments
    if len(sys.argv) == 2 or len(sys.argv) == 3:
        POWER = "y"
        if sys.argv[1] == 's':
            TASK = "SENDER"
        elif sys.argv[1] == 'r':
            TASK = "RECEIVER"
        else:
            print("Specify 's' for sender or 'r' for receiver")
            exit()
        if len(sys.argv) == 3:
            if sys.argv[2] in ['y', 'n']:
                POWER = sys.argv[2]
            else:
                print("Specify 'y' for power or 'n' for no power, e.g.: python tfidf_rf.py s n")
                exit()
    else:
        print("Specify 's' for sender or 'r' for receiver, e.g.: python tfidf_rf.py s")
        exit()

    # Load data
    data_path = 'data/'
    with jsonlines.open(data_path + 'train.jsonl', 'r') as reader:
        train = list(reader)
    with jsonlines.open(data_path + 'test.jsonl', 'r') as reader:
        test = list(reader)

    # Initialize spaCy
    nlp = English()

    tfidf_rf(train, test)
