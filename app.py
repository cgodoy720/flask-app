from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
import random
import os
import json
import psutil

# Check Python version and set environment variables
print(sys.executable)  # Should show the path within the 'venv' directory
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Get the process memory usage for initial debugging
process = psutil.Process(os.getpid())
print(f"Initial memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Set the device (CUDA for GPU, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Use mixed precision if available (float16 for reduced memory usage)
if device.type == 'cuda':
    model.half()
else:
    # Apply quantization for CPU deployment to reduce memory
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Flask setup
app = Flask(__name__)

# Function to load data from JSON files
def load_json_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Load predefined responses from JSON files
fatherhood_responses = load_json_data("fatherhood_responses.json")
conversation = load_json_data("conversation.json")

# Clear GPU memory function
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

    print(f"Received message: {msg}")  # For debugging
    token_count = len(tokenizer.encode(msg))

    response = get_chat_response(msg)
    return jsonify({"response": response, "token_count": token_count, "msg": msg}), 200

def get_chat_response(text):
    # Check for predefined responses
    if text in conversation:
        return conversation[text]

    # Check for keywords in fatherhood responses
    matched_keywords = [keyword for keyword in fatherhood_responses.keys() if keyword in text.lower()]
    if matched_keywords:
        selected_keyword = random.choice(matched_keywords)
        return random.choice(fatherhood_responses[selected_keyword])

    # Generate response using the model
    with torch.no_grad():  # Ensure no gradient calculation
        input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt').to(device)

        try:
            response_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error during model generation: {e}")
            response = "Sorry, something went wrong."

        clear_memory()
        return response

if __name__ == "__main__":
    # Get the port from environment variable or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
