from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
import random
import os
import json
import psutil


# Check Python version
print(sys.executable)  # Should show the path within the 'venv' directory
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Get the process memory usage
process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 ** 2} MB")



# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Set the device (CUDA for GPU, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Flask setup
app = Flask(__name__)

# Function to load data from JSON files
def load_json_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Load fatherhood responses and predefined conversations from files
fatherhood_responses = load_json_data("fatherhood_responses.json")
conversation = load_json_data("conversation.json")

# Clear GPU memory
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@app.route("/")
def index():
    return jsonify({"message": "Welcome to the API!"}), 200

@app.route("/get", methods=["POST"])
def chat():
    data = request.get_json()  
    msg = data.get("msg")  

    if not msg:
        return jsonify({"error": "No 'msg' field provided in the request."}), 400

    # Print the incoming message for debugging
    print(f"Received message: {msg}")
    
    token_count = len(tokenizer.encode(msg))

    # Get the appropriate response based on the input message
    response = get_chat_response(msg)
    return jsonify({"response": response, "token_count": token_count, "msg": msg}), 200

def get_chat_response(text):
    # Check if the input matches a predefined conversation entry
    if text in conversation:
        return conversation[text]

    # Check if the input matches any fatherhood-related keywords
    matched_keywords = [
        keyword for keyword in fatherhood_responses.keys() if keyword in text.lower()
    ]

    # If there are matched keywords, randomly select one
    if matched_keywords:
        selected_keyword = random.choice(matched_keywords)
        selected_response = random.choice(fatherhood_responses[selected_keyword])
        return selected_response
    
    # If no keyword matches, use the AI model to generate a response
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
    new_user_input_ids = new_user_input_ids.to(device)
    
    # Generate response using model
    try:
        chat_history_ids = model.generate(new_user_input_ids, max_length=150, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error during model generation: {e}")
        response = "Sorry, something went wrong."
    
    # Clear memory after the generation
    clear_memory()

    return response


if __name__ == "__main__":
    # Get the port from the environment variable or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
