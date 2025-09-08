import argparse
import math
from typing import Dict, Optional
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load

from mlx_lm.models.switch_layers import SwitchLinear, SwitchGLU

def split_switch_linear(orig_switch_linear : SwitchLinear, experts):
    input_dims = orig_switch_linear.input_dims
    output_dims = orig_switch_linear.output_dims
    num_experts = orig_switch_linear.num_experts

    switch_linear = SwitchLinear(input_dims, output_dims, num_experts, bias=False)
    switch_linear.weight = orig_switch_linear.weight[experts, :, :]

    if "bias" in orig_switch_linear:
        switch_linear.bias = orig_switch_linear.bias[experts, :]

    return switch_linear


# only for single token gen (seq_len=1)
class ParallelSwitchGLU(nn.Module):
    def __init__(self, orig_switch_glu: SwitchGLU, n_nodes: int):
        super().__init__()

        print(orig_switch_glu.gate_proj.weight.shape)
        print(orig_switch_glu.up_proj.weight.shape)
        print(orig_switch_glu.down_proj.weight.shape)

        n_experts = orig_switch_glu.gate_proj.weight.shape[0]
        n_experts_per_node = n_experts // n_nodes

        input_dims = orig_switch_glu.gate_proj.input_dims
        hidden_dims = orig_switch_glu.gate_proj.output_dims
        has_bias = "bias" in orig_switch_glu.gate_proj

        self.nodes = [SwitchGLU(input_dims, hidden_dims=hidden_dims, num_experts=n_experts_per_node, activation=orig_switch_glu.activation, bias=has_bias) for _ in range(n_nodes)]
        self.n_experts_per_node = n_experts // n_nodes
        self.n_experts = n_experts
        self.input_dims = input_dims

        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            if not hasattr(orig_switch_glu, proj_name):
                print(f'projection not found: {proj_name}')
                continue
            orig_proj = getattr(orig_switch_glu, proj_name)
            for n_node, node in enumerate(self.nodes):
                experts = [i for i in range(n_node * n_experts_per_node, n_node * n_experts_per_node + n_experts_per_node)]
                proj = split_switch_linear(orig_switch_linear=orig_proj, experts=experts)
                print(proj.weight.shape)
                setattr(node, proj_name, proj)

    def __call__(self, x, indices: mx.array) -> mx.array:
        batch_size, seq_len = x.shape[:2]
        
        indices = mx.flatten(indices)

        print(f'indices = {indices}')

        result = mx.zeros((indices.size, self.input_dims))
        
        curr = 0
        for node_idx, node in enumerate(self.nodes):
            # Determine which experts belong to this node
            node_expert_start = node_idx * self.n_experts_per_node
            node_expert_end = (node_idx + 1) * self.n_experts_per_node

            filtered = [idx - node_expert_start for idx in indices.tolist() if idx >= node_expert_start and idx < node_expert_end]
            local_size = len(filtered)

            local_indices = mx.array(filtered)
            
            print(f'local_indices = {local_indices}')

            local_indices = local_indices[None][None]
            
            # Process through the node's SwitchGLU
            node_output = node(x, local_indices)

            result[curr:curr+local_size] = node_output[0, 0, :, :]

            curr += local_size
        
        return result[None][None]

# assuming disaggregation, this is only for seq_len=1
def compare_switch_glus(switch_glu_a, switch_glu_b):
    input_dims = switch_glu_a.gate_proj.input_dims
    num_experts = switch_glu_a.gate_proj.weight.shape[0]
    num_selected_experts = min(4, num_experts)

    batch_size = 1
    seq_len = 1

    indices = mx.random.randint(0, num_experts, (batch_size, seq_len, num_selected_experts))

    scale = math.sqrt(1 / input_dims)

    x = mx.random.uniform(
        low=-scale,
        high=scale,
        shape=(batch_size, seq_len, input_dims),
    )

    y_a = switch_glu_a(x, indices)
    print(f'y_a.shape = {y_a.shape}')
    print('=' * 80)
    
    y_b = switch_glu_b(x, indices)
    print(f'y_b.shape = {y_b.shape}')

    print(mx.array_equal(y_a, y_b))

    exit(0)


def patch_switch_glus(model, n_nodes: int):
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and isinstance(layer.mlp, type(layer.mlp)):
            # Check if it has switch_mlp attribute (indicates MoE layer)
            if hasattr(layer.mlp, 'switch_mlp') and isinstance(layer.mlp.switch_mlp, SwitchGLU):
                print(f"Patching layer {layer_idx} SwitchGLU module...")
                
                orig_switch_glu = layer.mlp.switch_mlp
                switch_glu = ParallelSwitchGLU(orig_switch_glu, n_nodes=n_nodes)

                compare_switch_glus(orig_switch_glu, switch_glu)
                
                layer.mlp.switch_mlp = switch_glu

    return model

def main():
    """Main test function."""

    
    parser = argparse.ArgumentParser(description="Test Qwen3 with expert parallelism")
    parser.add_argument("--model", type=str, 
                       default="mlx-community/Qwen3-30B-A3B-Instruct-2507-bf16",
                       help="Model path or HuggingFace model ID")
    parser.add_argument("--nodes", type=int, default=2,
                       help="Number of nodes to simulate")
    
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    
    # Run the test
    patch_switch_glus(
        model=model,
        n_nodes=args.nodes
    )


if __name__ == "__main__":
    main()