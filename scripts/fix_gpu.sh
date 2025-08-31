#!/bin/bash
# fix_gpu.sh - Fix CUDA GPU detection issues in Paperspace

echo "🔧 GPU Detection & Fix Script"
echo "=============================="

# Function to check GPU availability
check_gpu() {
    echo "📊 Checking GPU status..."
    
    # Check if nvidia-smi is available
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ nvidia-smi found"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
            echo "❌ nvidia-smi failed to query GPU"
            return 1
        }
    else
        echo "❌ nvidia-smi not found"
        return 1
    fi
    
    # Check CUDA availability
    if [ -d "/usr/local/cuda" ]; then
        echo "✅ CUDA installation found at /usr/local/cuda"
        /usr/local/cuda/bin/nvcc --version 2>/dev/null || echo "⚠️ nvcc not functional"
    else
        echo "❌ CUDA not found at /usr/local/cuda"
    fi
    
    # Check PyTorch CUDA
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('❌ CUDA not available in PyTorch')
" 2>/dev/null || echo "❌ Failed to check PyTorch CUDA"
}

# Function to fix common GPU issues
fix_gpu_issues() {
    echo ""
    echo "🔧 Attempting to fix GPU issues..."
    
    # 1. Export CUDA paths
    echo "1️⃣ Setting CUDA environment variables..."
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export CUDA_HOME=/usr/local/cuda
    
    # Save to bashrc for persistence
    if ! grep -q "CUDA_HOME" ~/.bashrc; then
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
        echo "✅ CUDA paths added to ~/.bashrc"
    fi
    
    # 2. Reset GPU
    echo "2️⃣ Resetting GPU state..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --gpu-reset 2>/dev/null || echo "⚠️ Could not reset GPU (may require sudo)"
    fi
    
    # 3. Check for driver issues
    echo "3️⃣ Checking NVIDIA drivers..."
    if ! lsmod | grep -q nvidia; then
        echo "❌ NVIDIA kernel modules not loaded"
        echo "   Try: sudo modprobe nvidia"
    else
        echo "✅ NVIDIA kernel modules loaded"
    fi
    
    # 4. Check for conflicting processes
    echo "4️⃣ Checking for GPU-using processes..."
    nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>/dev/null || echo "   No processes using GPU"
    
    # 5. Paperspace-specific fixes
    echo "5️⃣ Applying Paperspace-specific fixes..."
    
    # Check if we're in a Paperspace environment
    if [ ! -z "$PAPERSPACE_CLUSTER_ID" ] || [ -f "/notebooks/.paperspace" ]; then
        echo "✅ Paperspace environment detected"
        
        # Wait for GPU to be available (Paperspace sometimes takes time)
        echo "   Waiting for GPU initialization (up to 30 seconds)..."
        for i in {1..30}; do
            if nvidia-smi &> /dev/null; then
                echo "   ✅ GPU available after $i seconds"
                break
            fi
            sleep 1
        done
        
        # Check Paperspace GPU allocation
        if [ ! -z "$PAPERSPACE_METRIC_WORKLOAD_ID" ]; then
            echo "   Workload ID: $PAPERSPACE_METRIC_WORKLOAD_ID"
        fi
    else
        echo "⚠️ Not in Paperspace environment"
    fi
}

# Function to install CUDA if missing
install_cuda_if_needed() {
    if [ ! -d "/usr/local/cuda" ]; then
        echo ""
        echo "🚨 CUDA not installed. Installing CUDA 12.4..."
        
        # This would need sudo in most cases
        if command -v apt-get &> /dev/null; then
            echo "Would need to run:"
            echo "  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
            echo "  sudo dpkg -i cuda-keyring_1.1-1_all.deb"
            echo "  sudo apt-get update"
            echo "  sudo apt-get -y install cuda-toolkit-12-4"
        else
            echo "❌ Package manager not found"
        fi
    fi
}

# Function to create CPU fallback config
create_cpu_fallback() {
    echo ""
    echo "💡 Creating CPU fallback configuration..."
    
    cat > /tmp/comfyui_cpu_mode.py << 'EOF'
#!/usr/bin/env python3
"""
CPU Mode launcher for ComfyUI when GPU is not available
"""
import os
import sys

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['FORCE_CPU'] = '1'

# Add ComfyUI to path
sys.path.insert(0, '/comfyui-paperspace-notebook/ComfyUI')

print("🖥️  Starting ComfyUI in CPU mode...")
print("⚠️  Performance will be significantly slower than GPU mode")

# Import and run ComfyUI
try:
    import main
    main.main()
except ImportError as e:
    print(f"❌ Failed to import ComfyUI: {e}")
    sys.exit(1)
EOF
    
    chmod +x /tmp/comfyui_cpu_mode.py
    echo "✅ CPU fallback created at /tmp/comfyui_cpu_mode.py"
    echo "   Run with: python3 /tmp/comfyui_cpu_mode.py"
}

# Function to test GPU with ComfyUI
test_comfyui_gpu() {
    echo ""
    echo "🧪 Testing GPU with ComfyUI..."
    
    python3 -c "
import sys
sys.path.insert(0, '/comfyui-paperspace-notebook/ComfyUI')

try:
    import torch
    import comfy.model_management as mm
    
    device = mm.get_torch_device()
    print(f'✅ ComfyUI device: {device}')
    
    if device.type == 'cuda':
        print(f'✅ GPU Memory: {mm.get_total_memory(device) / (1024**3):.1f} GB')
        print('✅ ComfyUI can use GPU!')
    else:
        print('⚠️ ComfyUI will run in CPU mode')
        
except Exception as e:
    print(f'❌ Error testing ComfyUI GPU: {e}')
"
}

# Main execution
echo "🚀 Starting GPU detection and fix process..."
echo ""

# Step 1: Check current GPU status
check_gpu

# Step 2: If GPU not detected, try to fix
if ! nvidia-smi &> /dev/null; then
    fix_gpu_issues
    
    # Re-check after fixes
    echo ""
    echo "📊 Re-checking GPU status after fixes..."
    check_gpu
fi

# Step 3: Test with ComfyUI
test_comfyui_gpu

# Step 4: Create CPU fallback
create_cpu_fallback

echo ""
echo "=============================="
echo "📋 Summary:"
echo ""

if nvidia-smi &> /dev/null && python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✅ GPU is available and working!"
    echo "   You can start ComfyUI normally"
else
    echo "❌ GPU not available"
    echo ""
    echo "🔧 Troubleshooting options:"
    echo "1. Wait a few minutes for Paperspace to allocate GPU"
    echo "2. Restart the Paperspace machine"
    echo "3. Check your Paperspace GPU allocation"
    echo "4. Use CPU mode: python3 /tmp/comfyui_cpu_mode.py"
    echo ""
    echo "💡 For Paperspace Free Tier:"
    echo "   - GPUs may not always be available"
    echo "   - Try different times of day"
    echo "   - Consider upgrading for guaranteed GPU access"
fi

echo "=============================="