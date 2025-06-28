#!/usr/bin/env python3
"""
Start MLflow server for experiment tracking
"""

import subprocess
import time
import requests
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_mlflow_server(host="localhost", port=5000, timeout=5):
    """Check if MLflow server is already running"""
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=timeout)
        if response.status_code == 200:
            logger.info(f"✅ MLflow server is already running at http://{host}:{port}")
            return True
    except requests.exceptions.RequestException:
        pass
    
    logger.info(f"❌ MLflow server is not running at http://{host}:{port}")
    return False

def start_mlflow_server(backend_store_uri=None, default_artifact_root=None, host="0.0.0.0", port=5000):
    """Start MLflow server"""
    
    # Set default paths
    if backend_store_uri is None:
        mlflow_dir = Path("/tmp/mlflow")
        mlflow_dir.mkdir(exist_ok=True)
        backend_store_uri = f"sqlite:///{mlflow_dir}/mlflow.db"
    
    if default_artifact_root is None:
        artifacts_dir = Path("/tmp/mlflow/artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        default_artifact_root = str(artifacts_dir)
    
    logger.info(f"Starting MLflow server...")
    logger.info(f"Backend store: {backend_store_uri}")
    logger.info(f"Artifacts root: {default_artifact_root}")
    logger.info(f"Host: {host}, Port: {port}")
    
    # MLflow server command
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", backend_store_uri,
        "--default-artifact-root", default_artifact_root,
        "--host", host,
        "--port", str(port)
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Start server in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"MLflow server started with PID: {process.pid}")
        
        # Wait for server to start
        max_wait = 30  # seconds
        wait_interval = 2
        
        for i in range(0, max_wait, wait_interval):
            if check_mlflow_server(host, port):
                logger.info(f"✅ MLflow server is ready after {i} seconds")
                return process
            
            logger.info(f"Waiting for MLflow server to start... ({i}/{max_wait}s)")
            time.sleep(wait_interval)
        
        # Check if process is still running
        if process.poll() is None:
            logger.warning(f"MLflow server process is running but not responding after {max_wait}s")
            return process
        else:
            # Process died
            stdout, stderr = process.communicate()
            logger.error(f"MLflow server failed to start:")
            logger.error(f"STDOUT: {stdout}")
            logger.error(f"STDERR: {stderr}")
            return None
        
    except FileNotFoundError:
        logger.error("MLflow not found. Install with: pip install mlflow")
        return None
    except Exception as e:
        logger.error(f"Error starting MLflow server: {str(e)}")
        return None

def stop_mlflow_server(process):
    """Stop MLflow server"""
    if process and process.poll() is None:
        logger.info("Stopping MLflow server...")
        process.terminate()
        
        # Wait for graceful shutdown
        try:
            process.wait(timeout=10)
            logger.info("✅ MLflow server stopped gracefully")
        except subprocess.TimeoutExpired:
            logger.warning("MLflow server didn't stop gracefully, killing...")
            process.kill()
            process.wait()
            logger.info("✅ MLflow server killed")

def main():
    """Main function to start MLflow server"""
    
    # Check if already running
    if check_mlflow_server():
        logger.info("MLflow server is already running, no need to start")
        return
    
    # Start server
    process = start_mlflow_server()
    
    if process:
        logger.info("MLflow server started successfully!")
        logger.info("Access the MLflow UI at: http://localhost:5000")
        logger.info("Press Ctrl+C to stop the server")
        
        try:
            # Keep the server running
            process.wait()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
            stop_mlflow_server(process)
    else:
        logger.error("Failed to start MLflow server")

if __name__ == "__main__":
    main()