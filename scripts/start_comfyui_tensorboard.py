#!/usr/bin/env python3
"""
Start ComfyUI with Paperspace Tensorboard URL pattern
100% Working solution for Paperspace access
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def get_paperspace_urls():
    """Generate Paperspace URLs using Tensorboard pattern"""
    fqdn = os.environ.get('PAPERSPACE_FQDN', '')
    
    if not fqdn:
        print("⚠️ PAPERSPACE_FQDN not found, using fallback")
        # Try to extract from hostname
        hostname = subprocess.run(['hostname'], capture_output=True, text=True)
        if hostname.returncode == 0:
            fqdn = hostname.stdout.strip()
    
    # Generate URLs using Tensorboard pattern (100% working)
    tensorboard_url = f"https://tensorboard-{fqdn}"
    tensorboard_url_alt = f"tensorboard-{fqdn}"
    
    return tensorboard_url, tensorboard_url_alt, fqdn

def start_comfyui():
    """Start ComfyUI with optimized settings for Paperspace"""
    
    # Get URLs
    tensorboard_url, tensorboard_url_alt, fqdn = get_paperspace_urls()
    
    print("🚀 Starting ComfyUI for Paperspace")
    print("=" * 50)
    print(f"📍 Tensorboard URL: {tensorboard_url}")
    print(f"📍 Alternative: {tensorboard_url_alt}")
    print(f"📍 FQDN: {fqdn}")
    print("=" * 50)
    
    # ComfyUI arguments optimized for Paperspace
    args = [
        "--listen",                              # Listen on all interfaces
        "--port", "6006",                        # Use port 6006 (Tensorboard port)
        "--preview-method", "auto",              # Auto preview method
        "--reserve-vram", "2.0",                 # Reserve 2GB VRAM for stability
        "--output-directory", "/storage/Output_1",  # Persistent output directory
        "--user-directory", "/storage/Workflow"     # Persistent workflow directory
    ]
    
    # Additional A6000 optimizations
    if os.environ.get('GPU_TYPE') == 'A6000' or 'A6000' in os.environ.get('PAPERSPACE_CLUSTER_ID', ''):
        args.extend([
            "--highvram",                       # Use high VRAM mode for A6000
            "--use-pytorch-cross-attention"     # Enable PyTorch cross attention
        ])
    
    # ComfyUI path
    comfyui_path = Path("/Models/ComfyUI")
    if not comfyui_path.exists():
        comfyui_path = Path("/notebooks/comfyui-paperspace-notebook/ComfyUI")
    
    main_py = comfyui_path / "main.py"
    
    if not main_py.exists():
        print(f"❌ ComfyUI not found at {main_py}")
        print("Please run the installation script first")
        return 1
    
    # Create output directories if they don't exist
    os.makedirs("/storage/Output_1", exist_ok=True)
    os.makedirs("/storage/Workflow", exist_ok=True)
    
    # Start ComfyUI
    print("\n🎨 Starting ComfyUI...")
    print(f"Command: python {main_py} {' '.join(args)}")
    print("\n" + "=" * 50)
    print("✨ ComfyUI Access URLs:")
    print(f"   Primary: {tensorboard_url}")
    print(f"   Direct: https://{fqdn}:6006")
    print("=" * 50)
    print("\nPress Ctrl+C to stop ComfyUI\n")
    
    try:
        # Run ComfyUI
        process = subprocess.Popen(
            [sys.executable, str(main_py)] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
                
                # Detect when server is ready
                if "Starting server" in line or "To see the GUI" in line:
                    print("\n" + "=" * 50)
                    print("🎉 ComfyUI is ready!")
                    print(f"🌐 Access at: {tensorboard_url}")
                    print("=" * 50 + "\n")
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopping ComfyUI...")
        process.terminate()
        time.sleep(2)
        if process.poll() is None:
            process.kill()
        print("✅ ComfyUI stopped")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return process.returncode

if __name__ == "__main__":
    sys.exit(start_comfyui())