#!/bin/bash
# start_comfyui_safe.sh - Safe ComfyUI starter with GPU fallback

echo "🚀 Safe ComfyUI Launcher"
echo "========================"

# Set working directory
cd /comfyui-paperspace-notebook

# Source environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Function to check GPU
check_gpu_available() {
    python3 -c "
import torch
import sys
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    sys.exit(0)
else:
    sys.exit(1)
" 2>/dev/null
    return $?
}

# Function to start ComfyUI with GPU
start_with_gpu() {
    echo "✅ Starting ComfyUI with GPU..."
    
    # Set optimal GPU settings for A6000
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export CUDA_LAUNCH_BLOCKING=0
    export TORCH_CUDA_ARCH_LIST="8.6"  # A6000 architecture
    
    cd ComfyUI
    python3 main.py \
        --listen 0.0.0.0 \
        --port 8188 \
        --enable-cors-header '*' \
        --use-pytorch-cross-attention \
        --force-fp16 \
        --preview-method auto \
        --use-quad-cross-attention
}

# Function to start ComfyUI with CPU
start_with_cpu() {
    echo "⚠️ Starting ComfyUI in CPU mode (slow performance)..."
    
    # Force CPU mode
    export CUDA_VISIBLE_DEVICES=""
    export FORCE_CPU=1
    
    cd ComfyUI
    python3 main.py \
        --listen 0.0.0.0 \
        --port 8188 \
        --enable-cors-header '*' \
        --cpu \
        --normalvram \
        --preview-method auto
}

# Function to wait for GPU
wait_for_gpu() {
    echo "⏳ Waiting for GPU to become available (max 60 seconds)..."
    
    for i in {1..60}; do
        if check_gpu_available; then
            echo "✅ GPU detected after $i seconds!"
            return 0
        fi
        
        # Every 10 seconds, show status
        if [ $((i % 10)) -eq 0 ]; then
            echo "   Still waiting... ($i/60)"
            
            # Try to reset GPU state
            if command -v nvidia-smi &> /dev/null; then
                nvidia-smi --gpu-reset 2>/dev/null || true
            fi
        fi
        
        sleep 1
    done
    
    echo "❌ GPU not available after 60 seconds"
    return 1
}

# Main logic
echo ""
echo "🔍 Checking GPU availability..."

# First quick check
if check_gpu_available; then
    echo "✅ GPU is available!"
    start_with_gpu
else
    echo "⚠️ GPU not immediately available"
    
    # Check if we're in Paperspace
    if [ ! -z "$PAPERSPACE_CLUSTER_ID" ] || [ -f "/notebooks/.paperspace" ]; then
        echo "📍 Paperspace environment detected"
        
        # Wait for GPU in Paperspace
        if wait_for_gpu; then
            start_with_gpu
        else
            # Ask user what to do
            echo ""
            echo "🤔 GPU not available. Options:"
            echo "1) Start in CPU mode (slow but works)"
            echo "2) Exit and try again later"
            echo "3) Try Docker mode (if available)"
            
            read -p "Choose option (1/2/3): " -n 1 -r choice
            echo ""
            
            case $choice in
                1)
                    start_with_cpu
                    ;;
                2)
                    echo "👋 Exiting. Try again later when GPU is available."
                    exit 0
                    ;;
                3)
                    echo "🐳 Starting Docker mode..."
                    if [ -f "scripts/docker_quick_start.sh" ]; then
                        ./scripts/docker_quick_start.sh start
                    else
                        echo "❌ Docker setup not found"
                        exit 1
                    fi
                    ;;
                *)
                    echo "❌ Invalid choice"
                    exit 1
                    ;;
            esac
        fi
    else
        # Not in Paperspace, probably local development
        echo "📍 Local environment detected"
        if command -v nvidia-smi &> /dev/null; then
            # Has NVIDIA tools but no GPU available
            echo "❌ NVIDIA drivers installed but no GPU detected"
            echo "   Check your GPU installation or drivers"
        fi
        
        # Start in CPU mode for local testing
        start_with_cpu
    fi
fi