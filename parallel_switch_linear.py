#!/usr/bin/env python3
"""
Parallel Switch Linear implementation for expert parallelism.
Gradually distributes SwitchLinear experts across nodes.
"""

import math
from typing import Dict, Optional
import mlx.core as mx
import mlx.nn as nn


class ParallelSwitchLinearNode(nn.Module):
    """
    A node that holds a subset of experts from a SwitchLinear layer.
    Initially runs locally, but designed for future distributed execution.
    """
    
    def __init__(
        self,
        weight: mx.array,  # Shape: [num_local_experts, output_dims, input_dims]
        bias: Optional[mx.array] = None,  # Shape: [num_local_experts, output_dims]
        expert_offset: int = 0,  # Global index offset for this node's experts
        node_id: int = 0,
    ):
        super().__init__()
        self.weight = weight
        if bias is not None:
            self.bias = bias
        self.expert_offset = expert_offset
        self.node_id = node_id
        
    @property
    def num_local_experts(self):
        return self.weight.shape[0]
    
    @property
    def output_dims(self):
        return self.weight.shape[1]
    
    @property
    def input_dims(self):
        return self.weight.shape[2]
    
    def __call__(self, x: mx.array, local_indices: mx.array, sorted_indices: bool = False) -> mx.array:
        """
        Process tokens with local expert indices.
        
        Args:
            x: Input tokens [batch, seq_len, dims] or expanded shape
            local_indices: Local expert indices (0-based for this node)
            sorted_indices: Whether indices are pre-sorted for efficiency
        
        Returns:
            Processed tokens
        """
        # Use gather_mm for efficient batched matrix multiplication
        x = mx.gather_mm(
            x,
            self.weight.swapaxes(-1, -2),
            lhs_indices=None,
            rhs_indices=local_indices,
            sorted_indices=sorted_indices,
        )
        
        if "bias" in self:
            x = x + mx.expand_dims(self.bias[local_indices], -2)
            
        return x


class ParallelSwitchLinear(nn.Module):
    """
    Distributed version of SwitchLinear that splits experts across nodes.
    Handles routing and index translation.
    """
    
    def __init__(
        self,
        input_dims: int,
        output_dims: int, 
        num_experts: int,
        expert_to_node: Optional[Dict[int, int]] = None,
        bias: bool = True,
    ):
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_experts = num_experts
        
        # Default: all experts on node 0 (local execution)
        if expert_to_node is None:
            expert_to_node = {i: 0 for i in range(num_experts)}
        self.expert_to_node = expert_to_node
        
        # Group experts by node
        self.node_to_experts = {}
        for expert_id, node_id in expert_to_node.items():
            if node_id not in self.node_to_experts:
                self.node_to_experts[node_id] = []
            self.node_to_experts[node_id].append(expert_id)
        
        # Sort expert lists for consistent ordering
        for node_id in self.node_to_experts:
            self.node_to_experts[node_id].sort()
        
        # Create nodes with their subset of experts
        self.nodes = {}
        scale = math.sqrt(1 / input_dims)
        
        for node_id, expert_ids in self.node_to_experts.items():
            num_local_experts = len(expert_ids)
            
            # Initialize weights for this node's experts
            node_weight = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_local_experts, output_dims, input_dims),
            )
            
            node_bias = None
            if bias:
                node_bias = mx.zeros((num_local_experts, output_dims))
            
            # Track the offset for index translation
            expert_offset = min(expert_ids)
            
            self.nodes[node_id] = ParallelSwitchLinearNode(
                weight=node_weight,
                bias=node_bias,
                expert_offset=expert_offset,
                node_id=node_id,
            )
    
    @classmethod
    def from_switch_linear(
        cls,
        switch_linear: "SwitchLinear",  # Type from switch_layers.py
        expert_to_node: Optional[Dict[int, int]] = None,
    ) -> "ParallelSwitchLinear":
        """
        Create a ParallelSwitchLinear from an existing SwitchLinear,
        splitting the weights according to expert_to_node mapping.
        """
        num_experts = switch_linear.num_experts
        input_dims = switch_linear.input_dims
        output_dims = switch_linear.output_dims
        has_bias = "bias" in switch_linear
        
        # Default: all experts on node 0
        if expert_to_node is None:
            expert_to_node = {i: 0 for i in range(num_experts)}
        
        # Create instance without initializing random weights
        parallel = cls.__new__(cls)
        nn.Module.__init__(parallel)
        
        parallel.input_dims = input_dims
        parallel.output_dims = output_dims
        parallel.num_experts = num_experts
        parallel.expert_to_node = expert_to_node
        
        # Group experts by node
        parallel.node_to_experts = {}
        for expert_id, node_id in expert_to_node.items():
            if node_id not in parallel.node_to_experts:
                parallel.node_to_experts[node_id] = []
            parallel.node_to_experts[node_id].append(expert_id)
        
        for node_id in parallel.node_to_experts:
            parallel.node_to_experts[node_id].sort()
        
        # Split weights and create nodes
        parallel.nodes = {}
        
        for node_id, expert_ids in parallel.node_to_experts.items():
            # Extract weights for this node's experts
            node_weight = switch_linear.weight[expert_ids]
            
            node_bias = None
            if has_bias:
                node_bias = switch_linear.bias[expert_ids]
            
            # Create local-to-global index mapping
            expert_offset = min(expert_ids)
            
            parallel.nodes[node_id] = ParallelSwitchLinearNode(
                weight=node_weight,
                bias=node_bias,
                expert_offset=expert_offset,
                node_id=node_id,
            )
        
        return parallel
    
    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        """
        Route inputs to appropriate nodes based on expert indices.
        
        Args:
            x: Input tensor with shape [..., num_experts, ..., input_dims]
            indices: Global expert indices
            sorted_indices: Whether to pre-sort for efficiency
            
        Returns:
            Processed outputs with shape [..., num_experts, ..., output_dims]
        """
        # Handle the case where x has expert dimension
        # x shape: [batch, seq, num_experts, 1, input_dims]
        # indices shape: [batch, seq, num_experts]
        
        # For each expert selection, process through appropriate node
        output_shape = list(x.shape)
        output_shape[-1] = self.output_dims
        output = mx.zeros(output_shape)
        
        # Process each node's tokens
        for node_id, expert_ids in self.node_to_experts.items():
            # Create mask for this node's experts
            node_mask = mx.array(mx.zeros(indices.shape) > 1)  # Create bool array
            for expert_id in expert_ids:
                node_mask = node_mask | (indices == expert_id)
            
            if mx.any(node_mask):
                # Convert global to local indices where mask is true
                local_indices = mx.zeros_like(indices)
                for local_idx, global_idx in enumerate(expert_ids):
                    local_indices = mx.where(
                        indices == global_idx,
                        local_idx,
                        local_indices
                    )
                
                # Process through the node
                # The node will only process where mask is true, but we pass all data
                # In distributed setting, we'd only send/receive masked portions
                node_output = self.nodes[node_id](x, local_indices, sorted_indices=sorted_indices)
                
                # Expand mask to match output shape
                # node_mask shape: [batch, seq, num_experts]
                # node_output shape: [batch, seq, num_experts, 1, output_dims]
                expanded_mask = node_mask
                # Add dimensions to match node_output
                if len(x.shape) == 5:  # [batch, seq, num_experts, 1, input_dims]
                    expanded_mask = mx.expand_dims(expanded_mask, 3)  # Add the '1' dimension
                    expanded_mask = mx.expand_dims(expanded_mask, 4)  # Add the output_dims dimension
                
                # Update output where this node processed
                output = mx.where(expanded_mask, node_output, output)
        
        return output


def test_parallel_switch_linear():
    """
    Unit test comparing SwitchLinear with ParallelSwitchLinear.
    """
    # Import SwitchLinear from mlx_lm
    from mlx_lm.models.switch_layers import SwitchLinear
    
    print("Testing ParallelSwitchLinear equivalence...")
    
    # Test parameters
    batch_size = 2
    seq_len = 4
    input_dims = 64
    output_dims = 128
    num_experts = 8
    
    # Create random inputs
    mx.random.seed(42)
    x = mx.random.uniform(-1, 1, (batch_size, seq_len, input_dims))
    
    # Create random expert assignments (top-k=2 for each token)
    indices = mx.random.randint(0, num_experts, (batch_size, seq_len, 2))
    
    # Test 1: All experts on single node (should be identical)
    print("\nTest 1: All experts on node 0")
    
    # Create original SwitchLinear
    switch_linear = SwitchLinear(input_dims, output_dims, num_experts, bias=True)
    
    # Create ParallelSwitchLinear from it
    parallel_linear = ParallelSwitchLinear.from_switch_linear(switch_linear)
    
    # Expand inputs for expert dimension
    x_expanded = mx.expand_dims(x, (-2, -3))
    
    # Forward pass through both
    output_orig = switch_linear(x_expanded, indices)
    output_parallel = parallel_linear(x_expanded, indices, sorted_indices=False)
    
    # Check equivalence
    diff = mx.abs(output_orig - output_parallel).max()
    print(f"Max difference (single node): {diff.item():.2e}")
    assert diff < 1e-5, f"Outputs don't match! Max diff: {diff.item()}"
    
    # Test 2: Split experts across 2 nodes
    print("\nTest 2: Experts split across 2 nodes")
    
    expert_to_node = {
        0: 0, 1: 0, 2: 0, 3: 0,  # First 4 experts on node 0
        4: 1, 5: 1, 6: 1, 7: 1,  # Last 4 experts on node 1
    }
    
    parallel_linear_2nodes = ParallelSwitchLinear.from_switch_linear(
        switch_linear, expert_to_node
    )
    
    output_parallel_2nodes = parallel_linear_2nodes(x_expanded, indices, sorted_indices=False)
    
    # Check equivalence with original
    diff = mx.abs(output_orig - output_parallel_2nodes).max()
    print(f"Max difference (2 nodes): {diff.item():.2e}")
    assert diff < 1e-5, f"Outputs don't match! Max diff: {diff.item()}"
    
    # Test 3: Irregular split
    print("\nTest 3: Irregular expert distribution")
    
    expert_to_node_irregular = {
        0: 0, 1: 0, 2: 0,  # 3 experts on node 0
        3: 1, 4: 1,        # 2 experts on node 1
        5: 2, 6: 2, 7: 2,  # 3 experts on node 2
    }
    
    parallel_linear_irregular = ParallelSwitchLinear.from_switch_linear(
        switch_linear, expert_to_node_irregular
    )
    
    output_parallel_irregular = parallel_linear_irregular(x_expanded, indices, sorted_indices=False)
    
    # Check equivalence with original
    diff = mx.abs(output_orig - output_parallel_irregular).max()
    print(f"Max difference (irregular): {diff.item():.2e}")
    assert diff < 1e-5, f"Outputs don't match! Max diff: {diff.item()}"
    
    print("\n✅ All tests passed!")
    
    # Print distribution info
    print("\nExpert distribution (irregular case):")
    for node_id, experts in parallel_linear_irregular.node_to_experts.items():
        print(f"  Node {node_id}: experts {experts}")


def patch_model_for_expert_parallelism(
    model,
    expert_to_node_mapping: Optional[Dict[int, Dict[int, int]]] = None
):
    """
    Patch a model's SwitchGLU modules to use ParallelSwitchLinear.
    
    Args:
        model: The model to patch
        expert_to_node_mapping: Optional dict mapping layer_idx -> expert_id -> node_id
                               If None, all experts stay on node 0 (local)
    
    Returns:
        The patched model
    """
    from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear
    
    # Default: all experts on node 0
    if expert_to_node_mapping is None:
        expert_to_node_mapping = {}
    
    # Find and patch all SwitchGLU modules
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and isinstance(layer.mlp, type(layer.mlp)):
            # Check if it has switch_mlp attribute (indicates MoE layer)
            if hasattr(layer.mlp, 'switch_mlp') and isinstance(layer.mlp.switch_mlp, SwitchGLU):
                print(f"Patching layer {layer_idx} SwitchGLU module...")
                
                switch_glu = layer.mlp.switch_mlp
                
                # Get expert mapping for this layer
                if layer_idx in expert_to_node_mapping:
                    expert_mapping = expert_to_node_mapping[layer_idx]
                else:
                    # Default: all experts on node 0
                    num_experts = switch_glu.gate_proj.num_experts
                    expert_mapping = {i: 0 for i in range(num_experts)}
                
                # Replace each SwitchLinear with ParallelSwitchLinear
                for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                    if hasattr(switch_glu, proj_name):
                        orig_proj = getattr(switch_glu, proj_name)
                        if isinstance(orig_proj, SwitchLinear):
                            parallel_proj = ParallelSwitchLinear.from_switch_linear(
                                orig_proj, expert_mapping
                            )
                            setattr(switch_glu, proj_name, parallel_proj)    
    return model


if __name__ == "__main__":
    test_parallel_switch_linear()