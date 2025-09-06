#!/usr/bin/env python3
"""
Analyze Qwen3-Coder-30B model architecture and MoE components
"""

from mlx_lm import load
import mlx.core as mx
import numpy as np

# Load model
checkpoint = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
print(f"Loading model from {checkpoint}...")
model, tokenizer = load(path_or_hf_repo=checkpoint)

print("\n=== Model Architecture Analysis ===")
print(f"Model type: {model.model_type}")
print(f"Number of layers: {model.model.num_hidden_layers}")
print(f"Hidden size: {model.args.hidden_size}")
print(f"Intermediate size: {model.args.intermediate_size}")
print(f"MoE intermediate size: {model.args.moe_intermediate_size}")
print(f"Number of experts: {model.args.num_experts}")
print(f"Experts per token: {model.args.num_experts_per_tok}")
print(f"Decoder sparse step: {model.args.decoder_sparse_step}")
print(f"MLP only layers: {model.args.mlp_only_layers}")
print(f"Norm topk prob: {model.args.norm_topk_prob}")

# Analyze layer types
print("\n=== Layer Analysis ===")
moe_layers = []
mlp_layers = []
for i in range(model.model.num_hidden_layers):
    layer = model.model.layers[i]
    if hasattr(layer.mlp, 'num_experts'):  # MoE layer
        moe_layers.append(i)
    else:  # Regular MLP layer
        mlp_layers.append(i)

print(f"MoE layers ({len(moe_layers)}): {moe_layers[:10]}..." if len(moe_layers) > 10 else f"MoE layers ({len(moe_layers)}): {moe_layers}")
print(f"MLP layers ({len(mlp_layers)}): {mlp_layers[:10]}..." if len(mlp_layers) > 10 else f"MLP layers ({len(mlp_layers)}): {mlp_layers}")

# Analyze a specific MoE layer
if moe_layers:
    moe_layer_idx = moe_layers[0]
    moe_layer = model.model.layers[moe_layer_idx].mlp
    print(f"\n=== MoE Layer {moe_layer_idx} Details ===")
    print(f"Number of experts: {moe_layer.num_experts}")
    print(f"Top-K experts: {moe_layer.top_k}")
    print(f"Normalize top-k probabilities: {moe_layer.norm_topk_prob}")
    
    # Check expert weights
    if hasattr(moe_layer.switch_mlp, 'gate_proj'):
        gate_proj = moe_layer.switch_mlp.gate_proj
        print(f"\nGate projection shape: {gate_proj.weight.shape}")
        print(f"  - Experts: {gate_proj.num_experts}")
        print(f"  - Input dims: {gate_proj.input_dims}")
        print(f"  - Output dims: {gate_proj.output_dims}")
    
    # Check routing gate
    if hasattr(moe_layer, 'gate'):
        print(f"\nRouting gate shape: {moe_layer.gate.weight.shape}")

# Test a simple forward pass with dummy input
print("\n=== Testing Forward Pass ===")
test_input = mx.array([[1, 2, 3, 4, 5]])  # Simple test tokens
print(f"Input shape: {test_input.shape}")

# Run through embedding
embeddings = model.model.embed_tokens(test_input)
print(f"Embeddings shape: {embeddings.shape}")

# Test routing through first MoE layer
if moe_layers:
    moe_layer_idx = moe_layers[0]
    moe_layer = model.model.layers[moe_layer_idx].mlp
    
    # Simulate routing
    test_tensor = mx.random.normal((1, 5, model.args.hidden_size))
    gates = moe_layer.gate(test_tensor)
    gates_softmax = mx.softmax(gates, axis=-1)
    
    print(f"\nRouting gates shape: {gates.shape}")
    print(f"Gates softmax shape: {gates_softmax.shape}")
    
    # Get top-k experts
    k = moe_layer.top_k
    topk_values = mx.topk(gates_softmax, k=k, axis=-1)
    print(f"Top-{k} expert indices shape: {topk_values[1].shape}")
    print(f"Top-{k} expert scores shape: {topk_values[0].shape}")

print("\n=== Memory Usage ===")
print(f"Model parameter count: ~30B (quantized to 4-bit)")
print(f"Estimated memory per expert: {30000 / 128:.1f}M parameters")
print(f"Active experts per token: {model.args.num_experts_per_tok}")
print(f"Memory efficiency gain: {128 / model.args.num_experts_per_tok:.1f}x theoretical")