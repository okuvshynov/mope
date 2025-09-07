#!/usr/bin/env python3
"""
Distributed Expert Parallelism - Worker Node

This is a worker node that:
1. Loads only the expert weights it's responsible for
2. Receives expert computation requests via HTTP API
3. Computes expert outputs and returns results
4. Handles multiple concurrent requests efficiently

Usage:
  python distributed_expert_worker.py --port 8001 --gpu-id 0
"""

import argparse
import asyncio
import json
import time
import numpy as np
from typing import List, Dict, Optional
from aiohttp import web, ClientSession
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpertWorkerNode:
    """
    Worker node that handles expert computations for a subset of experts
    """
    
    def __init__(self, port: int, gpu_id: int = 0):
        self.port = port
        self.gpu_id = gpu_id
        self.expert_ids = []
        self.model = None
        self.expert_weights = {}
        self.initialized = False
        
        # Performance tracking
        self.request_count = 0
        self.total_compute_time = 0.0
        self.total_requests_processed = 0
        
        logger.info(f"Worker node initialized on port {port}")
    
    async def initialize_experts(self, expert_ids: List[int], model_path: str):
        """
        Initialize this worker with specific experts
        """
        logger.info(f"Loading experts {expert_ids[:5]}{'...' if len(expert_ids) > 5 else ''} ({len(expert_ids)} total)")
        
        try:
            # Load the full model (we'll extract only needed expert weights)
            self.model, _ = load(path_or_hf_repo=model_path)
            self.expert_ids = expert_ids
            
            # Extract expert weights for our assigned experts
            self._extract_expert_weights()
            
            self.initialized = True
            logger.info(f"Successfully initialized {len(expert_ids)} experts")
            
            return {"status": "success", "message": f"Loaded {len(expert_ids)} experts"}
            
        except Exception as e:
            logger.error(f"Failed to initialize experts: {e}")
            return {"status": "error", "message": str(e)}
    
    def _extract_expert_weights(self):
        """
        Extract and cache weights for assigned experts from all MoE layers
        """
        logger.info("Extracting expert weights...")
        
        self.expert_weights = {}
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer.mlp, 'num_experts'):  # MoE layer
                # Get the switch MLP that contains expert weights
                switch_mlp = layer.mlp.switch_mlp
                
                # Extract weights for our experts
                layer_experts = {}
                for expert_id in self.expert_ids:
                    # Get expert weights from the switch MLP
                    # In MLX, switch layers store weights as [num_experts, output_dim, input_dim]
                    expert_weights = {
                        'gate_proj': switch_mlp.gate_proj.weight[expert_id],
                        'up_proj': switch_mlp.up_proj.weight[expert_id], 
                        'down_proj': switch_mlp.down_proj.weight[expert_id]
                    }
                    layer_experts[expert_id] = expert_weights
                
                self.expert_weights[layer_idx] = layer_experts
        
        logger.info(f"Extracted weights for {len(self.expert_ids)} experts across {len(self.expert_weights)} layers")
    
    def compute_expert(self, expert_id: int, layer_idx: int, token_embedding: List[float]) -> List[float]:
        """
        Compute expert forward pass for a single token
        """
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
        
        if expert_id not in self.expert_ids:
            raise ValueError(f"Expert {expert_id} not assigned to this worker")
        
        if layer_idx not in self.expert_weights:
            raise ValueError(f"Layer {layer_idx} not found")
        
        # Get expert weights for this layer
        expert_weights = self.expert_weights[layer_idx][expert_id]
        
        # Convert input to MLX array
        x = mx.array(token_embedding).reshape(1, -1)  # [1, hidden_size]
        
        # Expert computation: SwiGLU activation
        # gate_proj and up_proj have shape [output_dim, input_dim]
        # down_proj has shape [input_dim, output_dim]  
        gate_out = x @ expert_weights['gate_proj'].T  # [1, intermediate_size]
        up_out = x @ expert_weights['up_proj'].T     # [1, intermediate_size]
        
        # SwiGLU: silu(gate) * up
        activated = nn.silu(gate_out) * up_out       # [1, intermediate_size]
        
        # Project back to hidden dimension
        output = activated @ expert_weights['down_proj'].T  # [1, hidden_size]
        
        # Convert back to list
        mx.eval(output)
        return np.asarray(output).flatten().tolist()
    
    async def handle_compute_experts(self, request):
        """
        HTTP endpoint to handle expert computation requests
        """
        try:
            data = await request.json()
            requests = data['requests']
            
            start_time = time.time()
            results = []
            
            for req in requests:
                expert_id = req['expert_id']
                token_embedding = req['token_embedding']
                layer_idx = req.get('layer_idx', 0)  # Default to first layer for now
                
                # Compute expert output
                expert_output = self.compute_expert(expert_id, layer_idx, token_embedding)
                
                # Add request metadata to result
                result = {
                    'expert_id': expert_id,
                    'expert_output': expert_output,
                    'batch_idx': req['batch_idx'],
                    'seq_idx': req['seq_idx'],
                    'k_idx': req['k_idx'],
                    'weight': req['weight']
                }
                results.append(result)
            
            compute_time = time.time() - start_time
            self.total_compute_time += compute_time
            self.total_requests_processed += len(requests)
            self.request_count += 1
            
            logger.info(f"Processed {len(requests)} expert computations in {compute_time:.3f}s")
            
            return web.json_response({
                'status': 'success',
                'results': results,
                'compute_time': compute_time,
                'num_requests': len(requests)
            })
            
        except Exception as e:
            logger.error(f"Error in compute_experts: {e}")
            return web.json_response({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    async def handle_initialize(self, request):
        """
        HTTP endpoint to initialize worker with expert assignments
        """
        try:
            data = await request.json()
            expert_ids = data['expert_ids']
            model_path = data['model_path']
            
            result = await self.initialize_experts(expert_ids, model_path)
            
            if result['status'] == 'success':
                return web.json_response(result)
            else:
                return web.json_response(result, status=500)
                
        except Exception as e:
            logger.error(f"Error in initialize: {e}")
            return web.json_response({
                'status': 'error',
                'message': str(e)
            }, status=500)
    
    async def handle_status(self, request):
        """
        HTTP endpoint to get worker status and statistics
        """
        avg_compute_time = (self.total_compute_time / max(self.request_count, 1)) * 1000
        avg_requests_per_batch = self.total_requests_processed / max(self.request_count, 1)
        
        status = {
            'initialized': self.initialized,
            'expert_ids': self.expert_ids,
            'num_experts': len(self.expert_ids),
            'stats': {
                'total_batches': self.request_count,
                'total_requests': self.total_requests_processed,
                'total_compute_time': self.total_compute_time,
                'avg_compute_time_ms': avg_compute_time,
                'avg_requests_per_batch': avg_requests_per_batch
            }
        }
        
        return web.json_response(status)
    
    async def handle_shutdown(self, request):
        """
        HTTP endpoint to shutdown worker gracefully
        """
        logger.info("Shutdown requested")
        return web.json_response({'status': 'shutting_down'})
    
    def create_app(self):
        """
        Create aiohttp web application
        """
        app = web.Application()
        
        # Add routes
        app.router.add_post('/initialize', self.handle_initialize)
        app.router.add_post('/compute_experts', self.handle_compute_experts)
        app.router.add_get('/status', self.handle_status)
        app.router.add_post('/shutdown', self.handle_shutdown)
        
        # Health check
        app.router.add_get('/health', lambda req: web.json_response({'status': 'healthy'}))
        
        return app
    
    async def run(self):
        """
        Start the worker HTTP server
        """
        app = self.create_app()
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        logger.info(f"Worker node running on http://0.0.0.0:{self.port}")
        logger.info("Endpoints:")
        logger.info("  POST /initialize - Initialize with expert assignments")
        logger.info("  POST /compute_experts - Compute expert outputs")
        logger.info("  GET /status - Get worker status and stats")
        logger.info("  GET /health - Health check")
        logger.info("  POST /shutdown - Shutdown worker")
        
        # Keep the server running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down worker...")
        finally:
            await runner.cleanup()


async def main():
    parser = argparse.ArgumentParser(description='Distributed Expert Parallelism - Worker Node')
    parser.add_argument('--port', type=int, default=8001, help='Port for worker HTTP server')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use (currently unused in MLX)')
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Distributed Expert Worker - Port {args.port}")
    print("=" * 60)
    
    worker = ExpertWorkerNode(args.port, args.gpu_id)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())