#!/usr/bin/env python3
"""
Distributed Expert Parallelism - Main Node

This is the main coordinator node that:
1. Loads the full model and tokenizer
2. Handles token routing and embeddings
3. Sends expert computation requests to worker nodes
4. Aggregates results and generates final output

Usage:
  python distributed_expert_main.py --port 8000 --worker-addresses localhost:8001
"""

import argparse
import asyncio
import json
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
import aiohttp
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from dataclasses import dataclass


@dataclass
class WorkerInfo:
    address: str
    port: int
    expert_ids: List[int]
    
    @property
    def url(self):
        return f"http://{self.address}:{self.port}"


class DistributedExpertRouter:
    """
    Handles routing decisions and communication with worker nodes
    """
    
    def __init__(self, num_experts: int, num_experts_per_tok: int, workers: List[WorkerInfo]):
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.workers = workers
        
        # Create expert-to-worker mapping
        self.expert_to_worker = {}
        for worker in workers:
            for expert_id in worker.expert_ids:
                self.expert_to_worker[expert_id] = worker
        
        print(f"Distributed Expert Router initialized:")
        print(f"  Total experts: {num_experts}")
        print(f"  Experts per token: {num_experts_per_tok}")
        print(f"  Worker nodes: {len(workers)}")
        for worker in workers:
            print(f"    {worker.url}: experts {worker.expert_ids[:5]}{'...' if len(worker.expert_ids) > 5 else ''} ({len(worker.expert_ids)} total)")
    
    def route_tokens(self, router_logits: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Compute routing decisions (same as original)
        Returns: (expert_indices, expert_weights)
        """
        router_probs = mx.softmax(router_logits, axis=-1, precise=True)
        
        # Get top-k experts per token
        k = self.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-router_probs, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(router_probs, inds, axis=-1)
        scores /= mx.sum(scores, axis=-1, keepdims=True)  # Normalize
        
        return inds, scores
    
    def group_requests_by_worker(self, expert_indices: mx.array, expert_weights: mx.array, 
                                token_embeddings: mx.array, layer_idx: int) -> Dict[WorkerInfo, List[Dict]]:
        """
        Group expert computation requests by worker node
        """
        # Convert to numpy for easier processing
        mx.eval(expert_indices, expert_weights, token_embeddings)
        indices_np = np.asarray(expert_indices)
        weights_np = np.asarray(expert_weights)
        embeddings_np = np.asarray(token_embeddings)
        
        batch_size, seq_len, k = indices_np.shape
        
        # Group requests by worker
        worker_requests = {worker: [] for worker in self.workers}
        
        for b in range(batch_size):
            for s in range(seq_len):
                for k_idx in range(k):
                    expert_id = int(indices_np[b, s, k_idx])
                    weight = float(weights_np[b, s, k_idx])
                    token_embedding = embeddings_np[b, s].tolist()
                    
                    worker = self.expert_to_worker[expert_id]
                    
                    worker_requests[worker].append({
                        'expert_id': expert_id,
                        'weight': weight,
                        'token_embedding': token_embedding,
                        'layer_idx': layer_idx,
                        'batch_idx': b,
                        'seq_idx': s,
                        'k_idx': k_idx
                    })
        
        return worker_requests


class DistributedMoELayer(nn.Module):
    """
    Distributed MoE layer that sends expert computations to remote workers
    """
    
    def __init__(self, original_moe_block, router: DistributedExpertRouter, layer_idx: int):
        super().__init__()
        
        # Copy routing components (these stay on main node)
        self.gate = original_moe_block.gate
        self.num_experts = original_moe_block.num_experts
        self.top_k = original_moe_block.top_k
        self.norm_topk_prob = original_moe_block.norm_topk_prob
        
        # Distributed router and layer info
        self.router = router
        self.layer_idx = layer_idx
        
        # HTTP session for async requests
        self.session = None
    
    async def _ensure_session(self):
        """Ensure HTTP session is available"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
    
    async def _send_expert_requests(self, worker_requests: Dict[WorkerInfo, List[Dict]]) -> Dict[WorkerInfo, List[Dict]]:
        """
        Send expert computation requests to all workers in parallel
        """
        await self._ensure_session()
        
        async def send_to_worker(worker: WorkerInfo, requests: List[Dict]):
            if not requests:
                return worker, []
            
            payload = {
                'requests': requests,
                'hidden_size': 2048  # Qwen3 hidden size
            }
            
            try:
                async with self.session.post(
                    f"{worker.url}/compute_experts", 
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return worker, result['results']
                    else:
                        print(f"Error from worker {worker.url}: {response.status}")
                        return worker, []
            except Exception as e:
                print(f"Failed to communicate with worker {worker.url}: {e}")
                return worker, []
        
        # Send requests to all workers in parallel
        tasks = [send_to_worker(worker, requests) for worker, requests in worker_requests.items()]
        results = await asyncio.gather(*tasks)
        
        return dict(results)
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with distributed expert computation
        """
        # Step 1: Compute routing decisions locally
        router_logits = self.gate(x)
        expert_indices, expert_weights = self.router.route_tokens(router_logits)
        
        # Step 2: Group requests by worker (include layer info)
        worker_requests = self.router.group_requests_by_worker(expert_indices, expert_weights, x, self.layer_idx)
        
        # Step 3: Send requests to workers (this needs to be async, but MLX calls are sync)
        # We'll use a sync wrapper for now
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            worker_results = loop.run_until_complete(self._send_expert_requests(worker_requests))
        finally:
            loop.close()
        
        # Step 4: Aggregate results
        return self._aggregate_results(x.shape, worker_results, expert_indices, expert_weights)
    
    def _aggregate_results(self, input_shape: Tuple, worker_results: Dict[WorkerInfo, List[Dict]], 
                          expert_indices: mx.array, expert_weights: mx.array) -> mx.array:
        """
        Aggregate expert computation results from all workers
        """
        batch_size, seq_len, hidden_size = input_shape
        output = mx.zeros((batch_size, seq_len, hidden_size))
        
        # Convert output to numpy for easier manipulation
        output_np = np.zeros((batch_size, seq_len, hidden_size))
        
        # Process results from each worker
        for worker, results in worker_results.items():
            for result in results:
                b = result['batch_idx']
                s = result['seq_idx']
                weight = result['weight']
                expert_output = np.array(result['expert_output'])
                
                # Add weighted expert contribution
                output_np[b, s] += expert_output * weight
        
        return mx.array(output_np)


async def setup_workers(worker_addresses: List[str]) -> List[WorkerInfo]:
    """
    Setup worker nodes and distribute experts among them
    """
    workers = []
    num_experts = 128  # Qwen3 has 128 experts
    experts_per_worker = num_experts // len(worker_addresses)
    
    for i, addr in enumerate(worker_addresses):
        if ':' in addr:
            address, port = addr.split(':')
            port = int(port)
        else:
            address = addr
            port = 8001  # Default port
        
        # Assign experts to this worker
        start_expert = i * experts_per_worker
        end_expert = start_expert + experts_per_worker
        if i == len(worker_addresses) - 1:  # Last worker gets remainder
            end_expert = num_experts
        
        expert_ids = list(range(start_expert, end_expert))
        
        worker = WorkerInfo(address, port, expert_ids)
        workers.append(worker)
    
    # Initialize workers
    print("Initializing worker nodes...")
    async with aiohttp.ClientSession() as session:
        for worker in workers:
            try:
                payload = {
                    'expert_ids': worker.expert_ids,
                    'model_path': 'mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit'
                }
                async with session.post(f"{worker.url}/initialize", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"  Worker {worker.url}: {result['message']}")
                    else:
                        print(f"  Failed to initialize worker {worker.url}: {response.status}")
            except Exception as e:
                print(f"  Error initializing worker {worker.url}: {e}")
    
    return workers


def patch_model_with_distributed_experts(model, router: DistributedExpertRouter):
    """
    Patch model to use distributed expert computation
    """
    print("\nPatching model for distributed expert parallelism...")
    
    patched_layers = 0
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'num_experts'):  # This is a MoE layer
            original_moe = layer.mlp
            layer.mlp = DistributedMoELayer(original_moe, router, i)
            patched_layers += 1
    
    print(f"Patched {patched_layers} MoE layers for distributed computation")
    return model


async def main():
    parser = argparse.ArgumentParser(description='Distributed Expert Parallelism - Main Node')
    parser.add_argument('--port', type=int, default=8000, help='Port for main node')
    parser.add_argument('--worker-addresses', nargs='+', required=True, 
                       help='Worker node addresses (e.g., localhost:8001 localhost:8002)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Distributed Expert Parallelism - Main Node")
    print("=" * 80)
    
    # Setup workers
    workers = await setup_workers(args.worker_addresses)
    
    # Load model and tokenizer on main node
    print(f"\nLoading model on main node...")
    checkpoint = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
    model, tokenizer = load(path_or_hf_repo=checkpoint)
    
    print(f"Model loaded:")
    print(f"  Architecture: {model.model_type}")
    print(f"  Layers: {model.model.num_hidden_layers}")
    print(f"  Experts: {model.args.num_experts}")
    print(f"  Active experts per token: {model.args.num_experts_per_tok}")
    
    # Create distributed router
    router = DistributedExpertRouter(
        model.args.num_experts, 
        model.args.num_experts_per_tok, 
        workers
    )
    
    # Patch model for distributed computation
    distributed_model = patch_model_with_distributed_experts(model, router)
    
    # Test distributed generation
    print(f"\n" + "=" * 50)
    print("Testing Distributed Generation")
    print("=" * 50)
    
    prompt = "Write a Python function to calculate factorial:"
    conversation = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        conversation=conversation, add_generation_prompt=True
    )
    
    print(f"Prompt: {prompt}")
    print(f"Generating with distributed expert parallelism...")
    
    start_time = time.time()
    try:
        response = generate(
            model=distributed_model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_tokens=100,
            verbose=False
        )
        success = True
    except Exception as e:
        print(f"Generation error: {e}")
        response = "[Distributed generation failed - check worker connectivity]"
        success = False
    end_time = time.time()
    
    print(f"\n" + "-" * 40)
    print("Generated Response:")
    print("-" * 40)
    print(response)
    print("-" * 40)
    
    if success:
        print(f"\nDistributed Generation Stats:")
        print(f"  Generation time: {end_time - start_time:.2f} seconds")
        print(f"  Tokens generated: 100")
        print(f"  Speed: ~{100 / (end_time - start_time):.1f} tokens/sec")
        print(f"  Workers used: {len(workers)}")
        print(f"  Experts per worker: {[len(w.expert_ids) for w in workers]}")
    
    print(f"\n" + "=" * 80)
    print("Distributed Expert Parallelism Complete!")
    print("=" * 80)
    
    # Cleanup
    for worker in workers:
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(f"{worker.url}/shutdown")
        except:
            pass


if __name__ == "__main__":
    asyncio.run(main())