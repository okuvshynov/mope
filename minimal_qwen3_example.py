#!/usr/bin/env python3
"""
Minimal example for running Qwen3-Coder-30B with mlx-lm
"""

from mlx_lm import generate, load
import mlx.core as mx

# Model checkpoint
checkpoint = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"

# Load model and tokenizer
print(f"Loading model from {checkpoint}...")
model, tokenizer = load(path_or_hf_repo=checkpoint)

# Print model architecture
print("\nModel structure:")
print(f"Model type: {model.model_type}")
print(f"Number of layers: {model.model.num_hidden_layers}")
print(f"Number of experts: {model.args.num_experts}")
print(f"Experts per token: {model.args.num_experts_per_tok}")
print(f"Hidden size: {model.args.hidden_size}")

# Simple test prompt
prompt = "Write a Python function to calculate fibonacci numbers."
conversation = [{"role": "user", "content": prompt}]

# Transform to chat template
prompt = tokenizer.apply_chat_template(
    conversation=conversation, add_generation_prompt=True
)

# Generate response
print("\nGenerating response...")
response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=100,
    verbose=True,
)

print("\nResponse:", response)