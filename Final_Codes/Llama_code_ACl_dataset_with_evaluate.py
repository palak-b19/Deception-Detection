import os
import jsonlines
from openai import OpenAI
from sklearn.metrics import f1_score, classification_report, accuracy_score

# Set your Groq API key as an environment variable
os.environ["GROQ_API_KEY"] = "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"

# Initialize the OpenAI client with Groq's base URL and API key
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

def generate_initial_prediction(client, game_state, dialogue, statement):
    """Generate initial prediction using the Groq model."""
    prompt = f"""
    Given the following game state and dialogue from a Diplomacy game, predict whether the following statement is a lie or not:

    Game State: {game_state}
    Dialogue: {dialogue}
    Statement: {statement}

    Please provide your prediction and rationale in the following format:
    Prediction: [lie or truth]
    Rationale: [your explanation]
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content
    lines = content.split("\n")
    prediction = None
    rationale = "No rationale provided"
    for line in lines:
        if line.startswith("Prediction:"):
            prediction = line.split(":")[1].strip().lower()
            if prediction not in ["lie", "truth"]:
                prediction = "unknown"
        elif line.startswith("Rationale:"):
            rationale = line.split(":")[1].strip()
    return prediction, rationale

def generate_feedback(client, prediction, rationale):
    """Generate feedback on the initial prediction using the Groq model."""
    prompt = f"""
    Here is a prediction about whether a statement in a Diplomacy game is a lie:

    Prediction: {prediction}
    Rationale: {rationale}

    Please provide feedback on this prediction in the following format:
    Feedback: [your feedback]
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content
    feedback = content.split(": ")[1].strip() if ": " in content else content
    return feedback

def refine_prediction(client, initial_prediction, initial_rationale, feedback):
    """Refine the prediction using the Groq model."""
    prompt = f"""
    Here is an initial prediction about whether a statement in a Diplomacy game is a lie, along with feedback on that prediction:

    Initial Prediction: {initial_prediction}
    Rationale: {initial_rationale}
    Feedback: {feedback}

    Please revise the prediction based on the feedback provided, and provide the revised prediction and rationale in the following format:
    Revised Prediction: [lie or truth]
    Revised Rationale: [your explanation]
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content
    lines = content.split("\n")
    revised_prediction = None
    revised_rationale = "No rationale provided"
    for line in lines:
        if line.startswith("Revised Prediction:"):
            revised_prediction = line.split(":")[1].strip().lower()
            if revised_prediction not in ["lie", "truth"]:
                revised_prediction = "unknown"
        elif line.startswith("Revised Rationale:"):
            revised_rationale = line.split(":")[1].strip()
    return revised_prediction, revised_rationale

def process_dialogue(client, dialogue_data, max_messages=None):
    """Process a single dialogue from the dataset and collect predictions/labels."""
    speakers = dialogue_data["speakers"]
    receivers = dialogue_data["receivers"]
    messages = dialogue_data["messages"]
    sender_labels = dialogue_data["sender_labels"]
    seasons = dialogue_data["seasons"]
    years = dialogue_data["years"]
    game_score = dialogue_data["game_score"]
    score_delta = dialogue_data["game_score_delta"]

    num_messages = min(len(messages), max_messages) if max_messages else len(messages)
    
    # Initialize lists to collect predictions and labels
    revised_predictions = []
    actual_labels = []

    for i in range(num_messages):
        # Dialogue is all previous messages (empty if first message)
        dialogue = "\n".join([f"{speakers[j]} to {receivers[j]}: {messages[j]}" for j in range(i)]) if i > 0 else ""
        statement = messages[i]
        game_state = f"Season: {seasons[i]}, Year: {years[i]}, Sender ({speakers[i]}) has {game_score[i]} supply centers, score delta with receiver ({receivers[i]}) is {score_delta[i]}."

        # Step 1: Generate initial prediction
        initial_prediction, initial_rationale = generate_initial_prediction(client, game_state, dialogue, statement)
        
        # Step 2: Generate feedback
        feedback = generate_feedback(client, initial_prediction, initial_rationale)
        
        # Step 3: Refine prediction
        revised_prediction, revised_rationale = refine_prediction(client, initial_prediction, initial_rationale, feedback)

        # Map sender_labels to "truth" or "lie"
        if isinstance(sender_labels[i], bool):
            label = "truth" if sender_labels[i] else "lie"
        elif isinstance(sender_labels[i], str):
            label = "truth" if sender_labels[i].lower() == "true" else "lie"
        else:
            label = "unknown"

        # Collect predictions and labels
        revised_predictions.append(revised_prediction)
        actual_labels.append(label)

        # Print results (optional, kept for debugging)
        print(f"\nMessage {i} in dialogue:")
        print(f"Game State: {game_state}")
        print(f"Dialogue: {dialogue}")
        print(f"Statement: {statement}")
        print('*'*50)
        print(" Predictions ")
        print(f"Initial Prediction: {initial_prediction}")
        print(f"Initial Rationale: {initial_rationale}")
        print(f"Feedback: {feedback}")
        print(f"Revised Prediction: {revised_prediction}")
        print(f"Actual Label: {label}")
        print(f"Revised Rationale: {revised_rationale}")
        print('*'*50)

    return revised_predictions, actual_labels

def evaluate(all_revised_predictions, all_actual_labels):
    # Define the label mapping
    label_map = {'truth': 0, 'lie': 1}
    
    # Filter out invalid entries ('unknown')
    valid_pairs = [(pred, label) for pred, label in zip(all_revised_predictions, all_actual_labels)
                   if pred in label_map and label in label_map]
    
    if not valid_pairs:
        print("No valid predictions/labels to evaluate.")
        return
    
    # Map strings to numerical values
    all_predictions_num = [label_map[pred] for pred, _ in valid_pairs]
    all_labels_num = [label_map[label] for _, label in valid_pairs]
    
    # Compute metrics
    lie_f1 = f1_score(all_labels_num, all_predictions_num, pos_label=1)
    truth_f1 = f1_score(all_labels_num, all_predictions_num, pos_label=0)
    macro_f1 = f1_score(all_labels_num, all_predictions_num, average='macro')
    micro_f1 = accuracy_score(all_labels_num, all_predictions_num)  # Micro F1 = accuracy in binary case
    
    # Print results
    print("\nPerformance Metrics:")
    print(f"Lie F1: {lie_f1:.4f}")
    print(f"Truth F1: {truth_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1 (Accuracy): {micro_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels_num, all_predictions_num, target_names=['truth', 'lie']))

if __name__ == "__main__":
    # Load the dataset (e.g., train.jsonl)
    dataset_path = "val.jsonl"  # Adjust path as needed
    with jsonlines.open(dataset_path) as reader:
        # Collect predictions and labels from all dialogues
        all_revised_predictions = []
        all_actual_labels = []

        for obj in reader:
            dialogue_data = obj
            preds, labels = process_dialogue(client, dialogue_data, max_messages=3)  # Limit to 3 messages for testing
            all_revised_predictions.extend(preds)
            all_actual_labels.extend(labels)
            # Remove break to process all dialogues; kept for testing
            break
        evaluate(all_revised_predictions,all_actual_labels)

       
