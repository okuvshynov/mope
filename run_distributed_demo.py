#!/usr/bin/env python3
"""
Distributed Expert Parallelism Demo Launcher

This script helps launch both worker and main nodes for local testing.
It starts worker processes in the background and then runs the main node.

Usage:
  python run_distributed_demo.py
"""

import subprocess
import time
import sys
import signal
import requests
from typing import List


class DistributedDemo:
    def __init__(self):
        self.worker_processes = []
        self.worker_ports = [8001, 8002]
        self.main_port = 8000
    
    def start_workers(self):
        """Start worker processes"""
        print("Starting worker nodes...")
        
        for port in self.worker_ports:
            print(f"  Starting worker on port {port}...")
            
            # Start worker process
            process = subprocess.Popen([
                sys.executable, 'distributed_expert_worker.py',
                '--port', str(port)
            ], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
            )
            
            self.worker_processes.append(process)
            
            # Give worker time to start
            time.sleep(2)
            
            # Check if worker is healthy
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=5)
                if response.status_code == 200:
                    print(f"    ✅ Worker on port {port} is healthy")
                else:
                    print(f"    ❌ Worker on port {port} health check failed")
            except Exception as e:
                print(f"    ❌ Worker on port {port} not responding: {e}")
    
    def run_main_node(self):
        """Run the main coordination node"""
        print(f"\nStarting main node on port {self.main_port}...")
        
        worker_addresses = [f"localhost:{port}" for port in self.worker_ports]
        
        try:
            # Run main node
            process = subprocess.run([
                sys.executable, 'distributed_expert_main.py',
                '--port', str(self.main_port),
                '--worker-addresses'] + worker_addresses,
                text=True,
                check=True
            )
            
        except subprocess.CalledProcessError as e:
            print(f"Main node failed with exit code {e.returncode}")
        except KeyboardInterrupt:
            print("\nMain node interrupted by user")
    
    def cleanup_workers(self):
        """Stop all worker processes"""
        print("\nStopping worker nodes...")
        
        # Try graceful shutdown first
        for port in self.worker_ports:
            try:
                requests.post(f"http://localhost:{port}/shutdown", timeout=2)
                print(f"  Sent shutdown signal to worker on port {port}")
            except:
                pass
        
        time.sleep(1)
        
        # Force terminate any remaining processes
        for i, process in enumerate(self.worker_processes):
            if process.poll() is None:  # Still running
                print(f"  Terminating worker process {self.worker_ports[i]}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
    
    def run(self):
        """Run the complete distributed demo"""
        print("=" * 80)
        print("Distributed Expert Parallelism Demo")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  Worker ports: {self.worker_ports}")
        print(f"  Main port: {self.main_port}")
        print(f"  Expert distribution: 64 experts per worker (128 total)")
        print()
        
        try:
            # Start workers
            self.start_workers()
            
            # Give workers more time to fully initialize
            print("Waiting for workers to be ready...")
            time.sleep(3)
            
            # Run main node
            self.run_main_node()
            
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        finally:
            # Cleanup
            self.cleanup_workers()
            print("Demo complete!")


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\nReceived interrupt signal, shutting down...")
    sys.exit(0)


if __name__ == "__main__":
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check dependencies
    try:
        import aiohttp
        import requests
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Please install with: pip install aiohttp requests")
        sys.exit(1)
    
    # Run demo
    demo = DistributedDemo()
    demo.run()