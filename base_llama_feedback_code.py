from openai import OpenAI
import os

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
        model="llama-3.3-70b-versatile",  # Example Groq model; adjust as needed
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content
    lines = content.split("\n")
    prediction = lines[0].split(": ")[1].strip()
    rationale = lines[1].split(": ")[1].strip()
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
    feedback = content.split(": ")[1].strip()
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
    revised_prediction = lines[0].split(": ")[1].strip()
    revised_rationale = lines[1].split(": ")[1].strip()
    return revised_prediction, revised_rationale

if __name__ == "__main__":
    # Simulated Diplomacy game data
    game_state = "Player A controls France, Player B controls Germany. France moves to Munich, Germany promises support but moves elsewhere."
    dialogue = "Player B says, 'I will support your move to Munich.'"
    statement = "I will support your move to Munich."

    # Step 1: Generate initial prediction
    initial_prediction, initial_rationale = generate_initial_prediction(client, game_state, dialogue, statement)
    print("Initial Prediction:", initial_prediction)
    print("Initial Rationale:", initial_rationale)
    print()

    # Step 2: Generate feedback
    feedback = generate_feedback(client, initial_prediction, initial_rationale)
    print("Feedback:", feedback)
    print()

    # Step 3: Refine prediction
    revised_prediction, revised_rationale = refine_prediction(client, initial_prediction, initial_rationale, feedback)
    print("Revised Prediction:", revised_prediction)
    print("Revised Rationale:", revised_rationale)