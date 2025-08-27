#!/bin/bash
# Test script to verify JupyterLab root access configuration

set -e

echo "=========================================="
echo "🧪 Testing ComfyUI Paperspace Setup"
echo "=========================================="

# Check if we're running as root or can become root
echo "==> Current user: $(whoami)"
echo "==> User ID: $(id -u)"

# Test Python availability
echo "==> Testing Python versions..."
python3 --version 2>/dev/null || echo "python3 not found"
python3.10 --version 2>/dev/null || echo "python3.10 not found"
python --version 2>/dev/null || echo "python not found"

# Test if we can install JupyterLab
echo "==> Testing pip install (dry run)..."
pip install --dry-run jupyterlab 2>/dev/null || echo "pip install test failed"

# Test if we can create directories in /storage
echo "==> Testing /storage access..."
if [ -w "/storage" ] 2>/dev/null; then
    echo "/storage is writable"
    mkdir -p /storage/test-comfyui 2>/dev/null || echo "Cannot create test directory in /storage"
else
    echo "/storage not accessible - using local storage"
    mkdir -p ./storage/test-comfyui
fi

# Test Git availability
echo "==> Testing Git..."
git --version 2>/dev/null || echo "git not found"

# Test network connectivity (for model downloads)
echo "==> Testing network connectivity..."
curl -s --head https://huggingface.co >/dev/null && echo "HuggingFace accessible" || echo "HuggingFace not accessible"

echo "=========================================="
echo "✅ Test completed"
echo "=========================================="