#!/usr/bin/env python3
"""
Start ComfyUI with Paperspace Tensorboard URL
Exact implementation as requested by user
"""

import os
import subprocess
import sys

# Generate Paperspace URLs using Tensorboard pattern
localurl = "https://tensorboard-" + os.environ.get('PAPERSPACE_FQDN', '')
localurl_1 = "tensorboard-" + os.environ.get('PAPERSPACE_FQDN', '')
print(localurl)

# ComfyUI arguments for Paperspace
Args = "--listen --port 6006 --preview-method auto --reserve-vram 2.0 --output-directory /storage/Output_1 --user-directory /storage/Workflow"

# Create output directories
os.makedirs("/storage/Output_1", exist_ok=True)
os.makedirs("/storage/Workflow", exist_ok=True)

print("=" * 50)
print("🚀 Starting ComfyUI for Paperspace")
print(f"📍 Access URL: {localurl}")
print(f"📍 Alternative: {localurl_1}")
print("=" * 50)

# Start ComfyUI
comfyui_path = "/Models/ComfyUI/main.py"
if not os.path.exists(comfyui_path):
    # Fallback to standard location
    comfyui_path = "/notebooks/comfyui-paperspace-notebook/ComfyUI/main.py"

if os.path.exists(comfyui_path):
    print(f"Starting: python {comfyui_path} {Args}")
    subprocess.run(f"python {comfyui_path} {Args}", shell=True)
else:
    print(f"❌ ComfyUI not found at {comfyui_path}")
    print("Please install ComfyUI first")