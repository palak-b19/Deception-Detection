import os
import jsonlines
import time
import json
from openai import OpenAI
from sklearn.metrics import f1_score, classification_report, accuracy_score
import backoff
from requests.exceptions import HTTPError

# Set Groq API key (use Kaggle Secrets or environment variable)
os.environ["GROQ_API_KEY"] = "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"  # Replace with your key

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

# Checkpoint file to save progress
CHECKPOINT_FILE = "/kaggle/working/checkpoint.json"

def save_checkpoint(predictions, labels, last_dialogue_idx):
    """Save predictions, labels, and last processed dialogue index to disk."""
    checkpoint = {
        "predictions": predictions,
        "labels": labels,
        "last_dialogue_idx": last_dialogue_idx
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)

def load_checkpoint():
    """Load checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint = json.load(f)
        return (
            checkpoint["predictions"],
            checkpoint["labels"],
            checkpoint["last_dialogue_idx"]
        )
    return [], [], -1

@backoff.on_exception(
    backoff.expo,
    HTTPError,
    max_tries=5,
    giveup=lambda e: not str(e).startswith("429")
)
def generate_batch_prediction(client, game_states, dialogues, statements):
    """Generate predictions for multiple messages in a single API call."""
    prompt = "Given the following game states, dialogues, and statements from a Diplomacy game, predict whether each statement is a lie or truth:\n\n"
    for idx, (game_state, dialogue, statement) in enumerate(zip(game_states, dialogues, statements)):
        prompt += f"Message {idx + 1}:\n"
        prompt += f"Game State: {game_state}\n"
        prompt += f"Dialogue: {dialogue}\n"
        prompt += f"Statement: {statement}\n"
        prompt += "Please provide your prediction and rationale for this message.\n\n"
    prompt += "Format the response as:\nMessage <number>:\nPrediction: [lie or truth]\nRationale: [explanation]\n"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000  # Adjust based on response length
    )
    content = response.choices[0].message.content
    
    # Initialize with default values to ensure we have predictions for all inputs
    predictions = ["unknown"] * len(game_states)
    rationales = ["No rationale provided"] * len(game_states)
    
    # Parse the response
    lines = content.split("\n")
    current_message_idx = None
    current_message_num = None
    current_rationale = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for message header
        if line.startswith("Message "):
            # Save previous message data if exists
            if current_message_idx is not None and current_message_idx < len(predictions):
                rationales[current_message_idx] = " ".join(current_rationale) if current_rationale else "No rationale provided"
            
            # Parse new message number
            try:
                msg_parts = line.split()
                if len(msg_parts) > 1:
                    current_message_num = int(msg_parts[1].rstrip(':'))
                    current_message_idx = current_message_num - 1  # Convert to 0-indexed
                    current_rationale = []
            except (ValueError, IndexError):
                current_message_idx = None
                
        # Check for prediction
        elif line.startswith("Prediction:") and current_message_idx is not None and current_message_idx < len(predictions):
            prediction_value = line.split(":", 1)[1].strip().lower()
            if prediction_value in ["lie", "truth"]:
                predictions[current_message_idx] = prediction_value
                
        # Check for rationale
        elif line.startswith("Rationale:") and current_message_idx is not None:
            rationale_text = line.split(":", 1)[1].strip()
            current_rationale = [rationale_text]
        elif current_message_idx is not None and current_rationale:
            current_rationale.append(line)
    
    # Save the last message's rationale if exists
    if current_message_idx is not None and current_message_idx < len(predictions):
        rationales[current_message_idx] = " ".join(current_rationale) if current_rationale else "No rationale provided"
    
    return predictions, rationales

def process_dialogue(client, dialogue_data, max_messages=3, batch_size=3):
    """Process a single dialogue with batching."""
    speakers = dialogue_data["speakers"]
    receivers = dialogue_data["receivers"]
    messages = dialogue_data["messages"]
    sender_labels = dialogue_data["sender_labels"]
    seasons = dialogue_data["seasons"]
    years = dialogue_data["years"]
    game_score = dialogue_data["game_score"]
    score_delta = dialogue_data["game_score_delta"]

    num_messages = min(len(messages), max_messages)
    revised_predictions = []
    actual_labels = []

    # Process messages in batches
    for start_idx in range(0, num_messages, batch_size):
        end_idx = min(start_idx + batch_size, num_messages)
        game_states = []
        dialogues = []
        statements = []
        batch_labels = []

        # Prepare batch
        for i in range(start_idx, end_idx):
            # Ensure all fields exist in the data
            if i >= len(speakers) or i >= len(receivers) or i >= len(messages) or i >= len(sender_labels) or i >= len(seasons) or i >= len(years) or i >= len(game_score) or i >= len(score_delta):
                print(f"Warning: Skipping message at index {i} due to missing data")
                continue
                
            dialogue = "\n".join([f"{speakers[j]} to {receivers[j]}: {messages[j]}" for j in range(i)]) if i > 0 else ""
            statement = messages[i]
            game_state = f"Season: {seasons[i]}, Year: {years[i]}, Sender ({speakers[i]}) has {game_score[i]} supply centers, score delta with receiver ({receivers[i]}) is {score_delta[i]}."
            game_states.append(game_state)
            dialogues.append(dialogue)
            statements.append(statement)
            
            # Map sender_labels
            if isinstance(sender_labels[i], bool):
                label = "truth" if sender_labels[i] else "lie"
            elif isinstance(sender_labels[i], str):
                label = "truth" if sender_labels[i].lower() == "true" else "lie"
            else:
                label = "unknown"
            batch_labels.append(label)

        # Skip empty batches
        if not game_states:
            continue
            
        try:
            # Generate batch predictions
            batch_predictions, batch_rationales = generate_batch_prediction(client, game_states, dialogues, statements)
            time.sleep(1)  # Respect per-minute rate limit

            # Collect results
            for pred, label in zip(batch_predictions, batch_labels):
                revised_predictions.append(pred)
                actual_labels.append(label)

            # Debugging output
            for i, (pred, rationale, act_label) in enumerate(zip(batch_predictions, batch_rationales, batch_labels)):
                msg_idx = start_idx + i
                print(f"\nMessage {msg_idx}:")
                print(f"Game State: {game_states[i]}")
                print(f"Dialogue: {dialogues[i]}")
                print(f"Statement: {statements[i]}")
                print(f"Prediction: {pred}")
                print(f"Rationale: {rationale}")
                print(f"Actual Label: {act_label}")

        except HTTPError as e:
            if str(e).startswith("429"):
                print(f"Rate limit hit, waiting...")
                time.sleep(60)  # Wait longer for daily limit
            else:
                print(f"Error processing batch: {e}")
                continue
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue

    return revised_predictions, actual_labels

def evaluate(all_revised_predictions, all_actual_labels):
    """Evaluate predictions."""
    label_map = {'truth': 0, 'lie': 1}
    valid_pairs = [(pred, label) for pred, label in zip(all_revised_predictions, all_actual_labels)
                   if pred in label_map and label in label_map]
    
    if not valid_pairs:
        print("No valid predictions/labels to evaluate.")
        return
    
    all_predictions_num = [label_map[pred] for pred, _ in valid_pairs]
    all_labels_num = [label_map[label] for _, label in valid_pairs]
    
    lie_f1 = f1_score(all_labels_num, all_predictions_num, pos_label=1)
    truth_f1 = f1_score(all_labels_num, all_predictions_num, pos_label=0)
    macro_f1 = f1_score(all_labels_num, all_predictions_num, average='macro')
    micro_f1 = accuracy_score(all_labels_num, all_predictions_num)
    
    print("\nPerformance Metrics:")
    print(f"Lie F1: {lie_f1:.4f}")
    print(f"Truth F1: {truth_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1 (Accuracy): {micro_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels_num, all_predictions_num, target_names=['truth', 'lie']))

if __name__ == "__main__":
    # Load dataset
    dataset_path = "/kaggle/input/deception-detection/validation.jsonl"  # Adjust path
    max_dialogues = 20  # Process ~700 dialogues to get ~2,000 messages (avg 3 messages/dialogue)
    max_messages_per_dialogue = 10
    batch_size = 5  # Number of messages per API call

    # Load checkpoint
    all_revised_predictions, all_actual_labels, last_dialogue_idx = load_checkpoint()

    with jsonlines.open(dataset_path) as reader:
        dialogues = list(reader)
        for idx, obj in enumerate(dialogues):
            if idx <= last_dialogue_idx:
                continue
            if idx >= max_dialogues:
                break
            dialogue_data = obj
            
            # Check if dialogue_data has all required fields
            required_fields = ["speakers", "receivers", "messages", "sender_labels", "seasons", "years", "game_score", "game_score_delta"]
            if not all(field in dialogue_data for field in required_fields):
                print(f"Dialogue {idx} missing required fields. Skipping.")
                continue
                
            # Check if any of the lists in dialogue_data are empty
            if not all(len(dialogue_data[field]) > 0 for field in required_fields):
                print(f"Dialogue {idx} has empty fields. Skipping.")
                continue
                
            try:
                preds, labels = process_dialogue(
                    client,
                    dialogue_data,
                    max_messages=max_messages_per_dialogue,
                    batch_size=batch_size
                )
                all_revised_predictions.extend(preds)
                all_actual_labels.extend(labels)
                save_checkpoint(all_revised_predictions, all_actual_labels, idx)
                print(f"Processed dialogue {idx + 1}/{max_dialogues}")
            except Exception as e:
                print(f"Error processing dialogue {idx}: {e}")
                # Save checkpoint for previously processed dialogues
                save_checkpoint(all_revised_predictions, all_actual_labels, idx-1)

    evaluate(all_revised_predictions, all_actual_labels)
