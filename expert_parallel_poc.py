#!/usr/bin/env python3
"""
Proof of Concept: Expert Parallelism for Qwen3-Coder-30B MoE Inference

This POC demonstrates how to split experts across multiple devices/processes
for parallel inference. In a real distributed setting, this would use MPI or
similar for communication.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class ExpertGroup:
    """Represents a group of experts that will be processed together"""
    expert_ids: List[int]
    device_id: int  # In MLX, this would map to different compute units


class ParallelMoELayer(nn.Module):
    """
    Expert-parallel MoE layer that distributes experts across devices.
    This is a simplified version for demonstration.
    """
    
    def __init__(self, 
                 hidden_size: int,
                 intermediate_size: int,
                 num_experts: int,
                 num_experts_per_tok: int,
                 num_devices: int = 4):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_devices = num_devices
        
        # Router/gate network (stays on all devices)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Partition experts across devices
        experts_per_device = num_experts // num_devices
        self.expert_groups = []
        
        for device_id in range(num_devices):
            start_expert = device_id * experts_per_device
            end_expert = start_expert + experts_per_device
            if device_id == num_devices - 1:
                end_expert = num_experts  # Handle remainder
            
            expert_ids = list(range(start_expert, end_expert))
            self.expert_groups.append(ExpertGroup(expert_ids, device_id))
        
        # Create expert weights (in real implementation, these would be distributed)
        self.expert_weights = self._create_expert_weights()
        
        print(f"Initialized ParallelMoELayer:")
        print(f"  - Total experts: {num_experts}")
        print(f"  - Experts per token: {num_experts_per_tok}")
        print(f"  - Number of devices: {num_devices}")
        print(f"  - Expert distribution: {[len(g.expert_ids) for g in self.expert_groups]} experts per device")
    
    def _create_expert_weights(self):
        """Create expert weight matrices"""
        weights = {}
        for i in range(self.num_experts):
            # Simplified expert: just gate, up, down projections
            weights[f"expert_{i}_gate"] = mx.random.normal(
                (self.hidden_size, self.intermediate_size)) * 0.02
            weights[f"expert_{i}_up"] = mx.random.normal(
                (self.hidden_size, self.intermediate_size)) * 0.02
            weights[f"expert_{i}_down"] = mx.random.normal(
                (self.intermediate_size, self.hidden_size)) * 0.02
        return weights
    
    def _route_tokens(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Route tokens to experts using top-k selection
        Returns: (expert_indices, expert_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute routing scores
        router_logits = self.gate(x)  # [batch, seq_len, num_experts]
        router_probs = mx.softmax(router_logits, axis=-1)
        
        # Manually implement top-k to preserve batch dimension
        topk_indices_list = []
        topk_values_list = []
        
        for b in range(batch_size):
            for s in range(seq_len):
                probs = router_probs[b, s]  # [num_experts]
                # MLX topk returns only indices, values computed separately
                indices = mx.argpartition(-probs, kth=self.num_experts_per_tok-1)[:self.num_experts_per_tok]
                values = probs[indices]
                topk_indices_list.append(indices)
                topk_values_list.append(values)
        
        # Stack and reshape to [batch, seq_len, k]
        topk_indices = mx.stack(topk_indices_list).reshape(batch_size, seq_len, self.num_experts_per_tok)
        topk_values = mx.stack(topk_values_list).reshape(batch_size, seq_len, self.num_experts_per_tok)
        
        # Normalize the top-k probabilities
        topk_values = topk_values / mx.sum(topk_values, axis=-1, keepdims=True)
        
        return topk_indices, topk_values
    
    def _expert_forward(self, x: mx.array, expert_id: int) -> mx.array:
        """Forward pass through a single expert"""
        gate = self.expert_weights[f"expert_{expert_id}_gate"]
        up = self.expert_weights[f"expert_{expert_id}_up"]
        down = self.expert_weights[f"expert_{expert_id}_down"]
        
        # Ensure x has the right shape for matrix multiplication
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])  # Flatten to 2D
        
        # SwiGLU activation
        gate_out = x_flat @ gate
        up_out = x_flat @ up
        activated = nn.silu(gate_out) * up_out
        output = activated @ down
        
        # Restore original shape
        return output.reshape(original_shape)
    
    
    def _parallel_expert_compute(self, x: mx.array, expert_indices: mx.array, 
                                expert_weights: mx.array) -> mx.array:
        """
        Simulate parallel computation of experts across devices.
        In a real implementation, this would involve actual parallelization.
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Convert to numpy for simpler processing in this POC
        x_np = np.array(x)
        expert_indices_np = np.array(expert_indices)
        expert_weights_np = np.array(expert_weights)
        
        output_np = np.zeros_like(x_np)
        
        # Track which tokens need which experts
        token_expert_map = {}
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(self.num_experts_per_tok):
                    expert_id = int(expert_indices_np[b, s, k])
                    weight = float(expert_weights_np[b, s, k])
                    
                    if expert_id not in token_expert_map:
                        token_expert_map[expert_id] = []
                    token_expert_map[expert_id].append((b, s, k, weight))
        
        # Simulate parallel processing per device
        device_computations = {d: [] for d in range(self.num_devices)}
        
        for expert_group in self.expert_groups:
            for expert_id in expert_group.expert_ids:
                if expert_id in token_expert_map:
                    device_computations[expert_group.device_id].append(expert_id)
        
        # Process experts (in real implementation, this would be parallel)
        for device_id, expert_ids in device_computations.items():
            if expert_ids:
                print(f"  Device {device_id} processing {len(expert_ids)} experts: {expert_ids[:5]}...")
                
                for expert_id in expert_ids:
                    for b, s, k, weight in token_expert_map[expert_id]:
                        token_input = x[b:b+1, s:s+1]  # [1, 1, hidden_size]
                        expert_out = self._expert_forward(token_input, expert_id)
                        # Add expert contribution to the output
                        expert_out_np = np.array(expert_out)
                        output_np[b, s] += expert_out_np[0, 0] * weight
        
        return mx.array(output_np)
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through the parallel MoE layer"""
        # Route tokens to experts
        expert_indices, expert_weights = self._route_tokens(x)
        
        print(f"    Expert indices shape: {expert_indices.shape}")
        print(f"    Expert weights shape: {expert_weights.shape}")
        
        # Compute expert outputs in parallel
        output = self._parallel_expert_compute(x, expert_indices, expert_weights)
        
        return output


class ExpertParallelModel(nn.Module):
    """
    Simplified model demonstrating expert parallelism
    """
    
    def __init__(self, vocab_size: int = 32000, 
                 hidden_size: int = 2048,
                 num_layers: int = 4,
                 num_experts: int = 128,
                 num_experts_per_tok: int = 8,
                 intermediate_size: int = 768,
                 num_devices: int = 4):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer layers with MoE
        self.layers = []
        for i in range(num_layers):
            # Simplified: just MoE layers, no attention for this POC
            moe_layer = ParallelMoELayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                num_devices=num_devices
            )
            self.layers.append(moe_layer)
        
        # Output projection
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def __call__(self, input_ids: mx.array) -> mx.array:
        # Embed tokens
        x = self.embed_tokens(input_ids)
        
        # Pass through MoE layers
        for i, layer in enumerate(self.layers):
            print(f"\nProcessing layer {i}...")
            x = layer(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits


def benchmark_expert_parallelism():
    """Benchmark and demonstrate expert parallelism"""
    
    print("=" * 60)
    print("Expert Parallelism Proof of Concept")
    print("=" * 60)
    
    # Model configuration (scaled down for demonstration)
    config = {
        "vocab_size": 32000,
        "hidden_size": 512,  # Scaled down from 2048
        "num_layers": 2,      # Scaled down from 48
        "num_experts": 32,    # Scaled down from 128
        "num_experts_per_tok": 4,  # Scaled down from 8
        "intermediate_size": 256,  # Scaled down from 768
        "num_devices": 4      # Simulate 4 devices
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create model
    print("\nInitializing model...")
    model = ExpertParallelModel(**config)
    
    # Create sample input
    batch_size = 2
    seq_len = 10
    input_ids = mx.random.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Run inference
    print("\n" + "=" * 40)
    print("Running inference with expert parallelism...")
    print("=" * 40)
    
    start_time = time.time()
    output = model(input_ids)
    mx.eval(output)  # Force computation
    end_time = time.time()
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Inference time: {end_time - start_time:.3f} seconds")
    
    # Calculate theoretical speedup
    total_experts = config["num_experts"]
    active_experts = config["num_experts_per_tok"]
    num_devices = config["num_devices"]
    
    print("\n" + "=" * 40)
    print("Theoretical Performance Analysis:")
    print("=" * 40)
    print(f"Total experts: {total_experts}")
    print(f"Active experts per token: {active_experts}")
    print(f"Number of devices: {num_devices}")
    print(f"Experts per device: {total_experts // num_devices}")
    print(f"Max active experts per device: ~{active_experts // num_devices + 1}")
    print(f"Load balancing efficiency: {active_experts / num_devices:.1f} experts/device average")
    print(f"Memory reduction per device: {num_devices}x")
    print(f"Compute efficiency gain: ~{total_experts / active_experts:.1f}x")


def demonstrate_routing_analysis():
    """Analyze expert routing patterns"""
    
    print("\n" + "=" * 60)
    print("Expert Routing Analysis")
    print("=" * 60)
    
    # Create a simple MoE layer
    moe = ParallelMoELayer(
        hidden_size=256,
        intermediate_size=128,
        num_experts=16,
        num_experts_per_tok=4,
        num_devices=2
    )
    
    # Generate sample input
    batch_size = 1
    seq_len = 5
    hidden_size = 256
    x = mx.random.normal((batch_size, seq_len, hidden_size))
    
    # Get routing decisions
    expert_indices, expert_weights = moe._route_tokens(x)
    
    print(f"\nInput tokens: {seq_len}")
    print(f"Expert selections per token:")
    
    for token_idx in range(seq_len):
        experts = expert_indices[0, token_idx].tolist()
        weights = expert_weights[0, token_idx].tolist()
        print(f"  Token {token_idx}: experts {experts}, weights {[f'{w:.3f}' for w in weights]}")
    
    # Analyze load distribution
    expert_counts = {}
    for token_idx in range(seq_len):
        for expert_id in expert_indices[0, token_idx].tolist():
            expert_counts[expert_id] = expert_counts.get(expert_id, 0) + 1
    
    print(f"\nExpert load distribution:")
    for expert_id in sorted(expert_counts.keys()):
        print(f"  Expert {expert_id}: {expert_counts[expert_id]} tokens")
    
    # Device load analysis
    device_loads = {d: 0 for d in range(moe.num_devices)}
    for expert_group in moe.expert_groups:
        for expert_id in expert_group.expert_ids:
            if expert_id in expert_counts:
                device_loads[expert_group.device_id] += expert_counts[expert_id]
    
    print(f"\nDevice load distribution:")
    for device_id, load in device_loads.items():
        print(f"  Device {device_id}: {load} token-expert pairs")


if __name__ == "__main__":
    # Run the benchmarks and demonstrations
    benchmark_expert_parallelism()
    demonstrate_routing_analysis()
    
    print("\n" + "=" * 60)
    print("Expert Parallelism POC Complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("1. Experts can be distributed across devices to reduce memory per device")
    print("2. Only active experts (8 out of 128) need computation per token")
    print("3. Routing decisions determine load balancing across devices")
    print("4. Communication overhead is the main challenge in distributed setting")
    print("5. Load balancing is critical for efficient parallelization")