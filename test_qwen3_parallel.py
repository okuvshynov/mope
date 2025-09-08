#!/usr/bin/env python3
"""
Test ParallelSwitchLinear with real Qwen3 MoE model.
"""

import time
from typing import Dict

from mlx_lm.utils import load
from mlx_lm.generate import generate
from mlx_lm.models.switch_layers import SwitchGLU

# Import our parallel implementation
from parallel_switch_linear import ParallelSwitchLinear, patch_model_for_expert_parallelism


def count_moe_layers(model):
    """Count the number of MoE layers in the model."""
    moe_layers = []
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'switch_mlp'):
            if isinstance(layer.mlp.switch_mlp, SwitchGLU):
                moe_layers.append(layer_idx)
    return moe_layers


def create_expert_distribution(num_experts: int, num_nodes: int = 2, 
                              strategy: str = "balanced") -> Dict[int, int]:
    """
    Create expert to node mapping.
    
    Args:
        num_experts: Total number of experts
        num_nodes: Number of nodes to distribute across
        strategy: Distribution strategy ("balanced", "first_heavy", "custom")
    
    Returns:
        Dict mapping expert_id -> node_id
    """
    if strategy == "balanced":
        # Evenly distribute experts across nodes
        experts_per_node = num_experts // num_nodes
        mapping = {}
        for expert_id in range(num_experts):
            node_id = expert_id // experts_per_node
            if node_id >= num_nodes:  # Handle remainder
                node_id = num_nodes - 1
            mapping[expert_id] = node_id
            
    elif strategy == "first_heavy":
        # Put more experts on first node (simulating imbalanced load)
        mapping = {}
        first_node_experts = (num_experts * 2) // 3
        for expert_id in range(num_experts):
            if expert_id < first_node_experts:
                mapping[expert_id] = 0
            else:
                mapping[expert_id] = 1
                
    else:  # custom
        # Example custom distribution
        mapping = {i: i % num_nodes for i in range(num_experts)}
    
    return mapping


def test_patched_model_inference(model_path: str = "mlx-community/Qwen3-30B-A3B-Instruct-2507-bf16",
                                num_nodes: int = 2,
                                distribution_strategy: str = "balanced"):
    """
    Test the patched model with inference.
    """
    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)
    
    # Check if it's a MoE model
    moe_layers = count_moe_layers(model)
    if not moe_layers:
        print("This model doesn't have MoE layers. Testing with standard model...")
        return
    
    print(f"Found {len(moe_layers)} MoE layers at positions: {moe_layers}")
    
    # Get number of experts from first MoE layer
    first_moe_layer = model.model.layers[moe_layers[0]]
    num_experts = first_moe_layer.mlp.switch_mlp.gate_proj.num_experts
    print(f"Number of experts per layer: {num_experts}")
    
    # Create expert distribution for each MoE layer
    expert_to_node_mapping = {}
    for layer_idx in moe_layers:
        expert_to_node_mapping[layer_idx] = create_expert_distribution(
            num_experts, num_nodes, distribution_strategy
        )
    
    # Print distribution plan
    print(f"\nExpert distribution plan ({distribution_strategy} strategy, {num_nodes} nodes):")
    for layer_idx in moe_layers[:3]:  # Show first 3 layers as example
        print(f"Layer {layer_idx}:")
        nodes_experts = {}
        for expert_id, node_id in expert_to_node_mapping[layer_idx].items():
            if node_id not in nodes_experts:
                nodes_experts[node_id] = []
            nodes_experts[node_id].append(expert_id)
        for node_id, experts in sorted(nodes_experts.items()):
            print(f"  Node {node_id}: {len(experts)} experts")
    
    # Test inference BEFORE patching
    print("\n" + "="*60)
    print("Testing inference BEFORE patching...")
    prompt = "The capital of France is"
    
    start_time = time.time()
    response_before = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=20, verbose=False
    )
    time_before = time.time() - start_time
    print(f"Response: {response_before}")
    print(f"Time: {time_before:.2f}s")
    
    # Apply patching
    print("\n" + "="*60)
    print("Applying expert parallelism patches...")
    model = patch_model_for_expert_parallelism(model, expert_to_node_mapping)
    
    # Verify patching
    patched_count = 0
    for layer_idx in moe_layers:
        layer = model.model.layers[layer_idx]
        if hasattr(layer.mlp, 'switch_mlp'):
            switch_mlp = layer.mlp.switch_mlp
            if isinstance(switch_mlp.gate_proj, ParallelSwitchLinear):
                patched_count += 1
    
    print(f"Successfully patched {patched_count}/{len(moe_layers)} MoE layers")
    
    # Test inference AFTER patching
    print("\n" + "="*60)
    print("Testing inference AFTER patching...")
    
    start_time = time.time()
    response_after = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=20, verbose=False
    )
    time_after = time.time() - start_time
    print(f"Response: {response_after}")
    print(f"Time: {time_after:.2f}s")
    
    # Compare results
    print("\n" + "="*60)
    print("Comparison:")
    print(f"Outputs match: {response_before == response_after}")
    print(f"Time before: {time_before:.2f}s")
    print(f"Time after:  {time_after:.2f}s")
    print(f"Overhead: {(time_after - time_before) / time_before * 100:.1f}%")
    

    # Test inference AFTER patching
    print("\n" + "="*60)
    print("Testing inference AFTER patching and warmup...")
    
    start_time = time.time()
    response_after = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=20, verbose=False
    )
    time_after = time.time() - start_time
    print(f"Response: {response_after}")
    print(f"Time: {time_after:.2f}s")

    # Memory analysis
    print("\n" + "="*60)
    print("Memory distribution analysis:")
    
    # Calculate memory per node
    total_expert_params = 0
    params_per_node = {i: 0 for i in range(num_nodes)}
    
    for layer_idx in moe_layers:
        layer = model.model.layers[layer_idx]
        if hasattr(layer.mlp, 'switch_mlp'):
            switch_mlp = layer.mlp.switch_mlp
            
            # Count parameters in each projection
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(switch_mlp, proj_name)
                if isinstance(proj, ParallelSwitchLinear):
                    for node_id, node in proj.nodes.items():
                        # Each node has num_local_experts * output_dims * input_dims parameters
                        node_params = node.weight.size
                        params_per_node[node_id] += node_params
                        total_expert_params += node_params
    
    print(f"Total expert parameters: {total_expert_params / 1e6:.1f}M")
    for node_id, params in params_per_node.items():
        percentage = (params / total_expert_params) * 100 if total_expert_params > 0 else 0
        print(f"  Node {node_id}: {params / 1e6:.1f}M parameters ({percentage:.1f}%)")
    
    return model


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Qwen3 with expert parallelism")
    parser.add_argument("--model", type=str, 
                       default="mlx-community/Qwen3-30B-A3B-Instruct-2507-bf16",
                       help="Model path or HuggingFace model ID")
    parser.add_argument("--nodes", type=int, default=2,
                       help="Number of nodes to simulate")
    parser.add_argument("--strategy", type=str, default="balanced",
                       choices=["balanced", "first_heavy", "custom"],
                       help="Expert distribution strategy")
    
    args = parser.parse_args()
    
    # Run the test
    test_patched_model_inference(
        model_path=args.model,
        num_nodes=args.nodes,
        distribution_strategy=args.strategy
    )


if __name__ == "__main__":
    main()