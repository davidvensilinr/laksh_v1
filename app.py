from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Load model and tokenizer
model_path = "laksh_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Add pad_token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"response": "Please say something first 😊"})

    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_new_tokens=40,              # 🔹 Shorter response
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process to extract reply after user input
    reply_only = reply[len(user_input):].strip().split("\n")[0]
    reply_only = reply_only[:200]  # Truncate long rambling if needed

    return jsonify({"response": reply_only})

import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

