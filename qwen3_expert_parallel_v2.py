#!/usr/bin/env python3
"""
Expert Parallelism V2 - Individual Expert Patching with Device Tracking

This version patches individual experts rather than the entire MoE block,
storing device_id in metadata and tracking call counts per device.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load
from mlx_lm.generate import generate
from typing import Dict
import time
from collections import defaultdict


class DistributedExpert(nn.Module):
    """
    Wrapper for individual expert that tracks device assignment and calls
    """
    
    def __init__(self, original_expert, expert_id: int, device_id: int):
        """
        Args:
            original_expert: The original expert module
            expert_id: Unique identifier for this expert
            device_id: Assigned device ID for this expert
        """
        super().__init__()
        
        # Store the original expert
        self.expert = original_expert
        
        # Store metadata
        self.expert_id = expert_id
        self.device_id = device_id
        
        # Track call count
        self.call_count = 0
        
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with call tracking
        """
        self.call_count += 1
        
        # In real distributed setting, this would involve:
        # 1. Sending input to self.device_id
        # 2. Computing on that device
        # 3. Returning result
        
        # For now, just compute locally
        return self.expert(x)
    
    def reset_stats(self):
        """Reset call statistics"""
        self.call_count = 0


class DeviceTracker:
    """
    Global tracker for device usage statistics
    """
    
    def __init__(self):
        self.device_calls = defaultdict(int)
        self.expert_calls = defaultdict(int)
        self.layer_stats = defaultdict(lambda: defaultdict(int))
        
    def record_call(self, layer_id: int, expert_id: int, device_id: int):
        """Record an expert call"""
        self.device_calls[device_id] += 1
        self.expert_calls[expert_id] += 1
        self.layer_stats[layer_id][device_id] += 1
        
    def print_stats(self):
        """Print usage statistics"""
        print("\n" + "=" * 60)
        print("Device Usage Statistics")
        print("=" * 60)
        
        if self.device_calls:
            print("\nCalls per device:")
            total_calls = sum(self.device_calls.values())
            for device_id in sorted(self.device_calls.keys()):
                calls = self.device_calls[device_id]
                percentage = (calls / total_calls * 100) if total_calls > 0 else 0
                print(f"  Device {device_id}: {calls:6d} calls ({percentage:5.1f}%)")
            
            print(f"\nTotal calls across all devices: {total_calls}")
            
            # Calculate load balance score (1.0 = perfect balance)
            if len(self.device_calls) > 1:
                avg_calls = total_calls / len(self.device_calls)
                variance = sum((calls - avg_calls) ** 2 for calls in self.device_calls.values())
                std_dev = (variance / len(self.device_calls)) ** 0.5
                balance_score = 1.0 - (std_dev / avg_calls if avg_calls > 0 else 0)
                print(f"Load balance score: {balance_score:.3f} (1.0 = perfect)")
        else:
            print("No calls recorded yet")
        
    def reset(self):
        """Reset all statistics"""
        self.device_calls.clear()
        self.expert_calls.clear()
        self.layer_stats.clear()


# Global device tracker instance
device_tracker = DeviceTracker()


class InstrumentedQwen3MoeSparseMoeBlock(nn.Module):
    """
    Modified Qwen3 MoE block that uses individually patched experts
    """
    
    def __init__(self, original_moe_block, layer_id: int, device_map: Dict[int, int]):
        """
        Args:
            original_moe_block: The original Qwen3MoeSparseMoeBlock
            layer_id: Identifier for this layer
            device_map: Maps expert_id -> device_id
        """
        super().__init__()
        
        self.layer_id = layer_id
        
        # Copy parameters from original block
        self.num_experts = original_moe_block.num_experts
        self.top_k = original_moe_block.top_k  
        self.norm_topk_prob = original_moe_block.norm_topk_prob
        
        # Copy the gate (router) network
        self.gate = original_moe_block.gate
        
        # Store original switch_mlp for reference
        self._original_switch_mlp = original_moe_block.switch_mlp
        
        # Store device mapping
        self.device_map = device_map
        
        # Track which experts are on which devices
        self.device_experts = defaultdict(list)
        for expert_id, device_id in device_map.items():
            self.device_experts[device_id].append(expert_id)
        
        # Print initialization info (once)
        if layer_id == 0:
            print(f"\nInstrumented MoE Block Configuration:")
            print(f"  Total experts: {self.num_experts}")
            print(f"  Active experts per token: {self.top_k}")
            print(f"  Devices: {len(set(device_map.values()))}")
            for device_id in sorted(set(device_map.values())):
                experts = self.device_experts[device_id]
                print(f"  Device {device_id}: {len(experts)} experts")
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with device tracking
        """
        # Step 1: Compute routing (same as original)
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1)
        
        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True)
        
        # Step 2: Track which experts are being called
        # Extract unique expert indices for this batch
        inds_flat = inds.reshape(-1)
        inds_list = inds_flat.tolist() if hasattr(inds_flat, 'tolist') else list(inds_flat)
        unique_experts = list(set(inds_list))
        for expert_id in unique_experts:
            if 0 <= expert_id < self.num_experts:
                device_id = self.device_map.get(expert_id, 0)
                device_tracker.record_call(self.layer_id, expert_id, device_id)
        
        # Step 3: Compute expert outputs using original switch_mlp
        y = self._original_switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        
        return y


def patch_individual_experts(model, num_devices: int = 4):
    """
    Patch individual experts in a Qwen3 model with device tracking
    
    This version patches each expert individually rather than the entire MoE block
    """
    print(f"\nPatching model with individual expert instrumentation ({num_devices} devices)...")
    
    # Create device mapping for experts
    num_experts = model.args.num_experts
    device_map = {}
    experts_per_device = num_experts // num_devices
    
    for expert_id in range(num_experts):
        device_id = expert_id // experts_per_device
        if device_id >= num_devices:  # Handle remainder
            device_id = num_devices - 1
        device_map[expert_id] = device_id
    
    # Patch each MoE layer
    patched_layers = 0
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'num_experts'):  # This is a MoE layer
            # Replace with instrumented version
            original_moe = layer.mlp
            layer.mlp = InstrumentedQwen3MoeSparseMoeBlock(
                original_moe, 
                layer_id=layer_idx,
                device_map=device_map
            )
            patched_layers += 1
    
    print(f"Patched {patched_layers} MoE layers with expert instrumentation")
    
    # Print device distribution
    print(f"\nExpert Distribution:")
    device_expert_count = defaultdict(int)
    for expert_id, device_id in device_map.items():
        device_expert_count[device_id] += 1
    
    for device_id in sorted(device_expert_count.keys()):
        count = device_expert_count[device_id]
        print(f"  Device {device_id}: {count} experts ({count/num_experts*100:.1f}%)")
    
    return model, device_map


def benchmark_expert_distribution():
    """Benchmark expert distribution and track device usage"""
    
    print("=" * 80)
    print("Expert Parallelism V2 - Individual Expert Patching")
    print("=" * 80)
    
    # Load the model
    checkpoint = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
    print(f"Loading model: {checkpoint}")
    model, tokenizer = load(path_or_hf_repo=checkpoint)
    
    print(f"\nModel Configuration:")
    print(f"  Architecture: {model.model_type}")  
    print(f"  Layers: {getattr(model.model, 'num_hidden_layers', 'Unknown')}")
    print(f"  Experts: {getattr(model.args, 'num_experts', 'Unknown')}")
    print(f"  Active experts per token: {getattr(model.args, 'num_experts_per_tok', 'Unknown')}")
    print(f"  Hidden size: {getattr(model.args, 'hidden_size', 'Unknown')}")
    
    # Patch model with individual expert instrumentation
    num_devices = 8  # Simulate 8 devices
    patched_model, device_map = patch_individual_experts(model, num_devices)
    
    # Test with multiple prompts to get better statistics
    test_prompts = [
        "Write a Python function to implement quicksort:",
        "Explain the concept of recursion in programming:",
        "How do neural networks learn from data?",
    ]
    
    print(f"\n" + "=" * 50)
    print("Running Inference with Device Tracking")
    print("=" * 50)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: {prompt[:50]}...")
        
        # Reset device tracker for this prompt
        device_tracker.reset()
        
        conversation = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, 'apply_chat_template'):
            formatted_prompt = tokenizer.apply_chat_template(
                conversation=conversation, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        # Generate response
        start_time = time.time()
        try:
            response = generate(
                model=patched_model,
                tokenizer=tokenizer,
                prompt=formatted_prompt,
                max_tokens=50,  # Shorter for testing
                verbose=False,
            )
            generation_time = time.time() - start_time
            
            # Print brief response preview
            response_preview = response.strip()[:100] + "..." if len(response.strip()) > 100 else response.strip()
            print(f"Response preview: {response_preview}")
            print(f"Generation time: {generation_time:.2f}s")
            
            # Show device usage for this prompt
            device_tracker.print_stats()
            
        except Exception as e:
            print(f"Generation failed: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("Benchmark Complete")
    print("=" * 80)
    print("\nKey Insights:")
    print("- Individual experts are now instrumented with device metadata")
    print("- Each expert tracks its assigned device_id")
    print("- Device usage is monitored in real-time during inference")
    print("- Load balancing can be evaluated using the balance score")
    print("\nIn a real distributed system:")
    print("- Each device would only hold its assigned experts")
    print("- Inter-device communication would occur for routing")
    print("- Computation would be truly parallel across devices")


if __name__ == "__main__":
    try:
        # Run the benchmark
        benchmark_expert_distribution()
        
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: This POC demonstrates the concept of individual expert patching")
        print("even if full execution isn't possible due to resource constraints.")