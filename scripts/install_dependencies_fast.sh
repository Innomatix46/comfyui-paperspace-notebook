#!/bin/bash
# install_dependencies_fast.sh - Schnellere Version mit Parallelisierung und Timeouts

echo "⚡ FAST Dependencies Installation"
echo "================================="

# Set faster pip options
export PIP_DEFAULT_TIMEOUT=60
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1

# Function to install package with retry
install_package() {
    local package=$1
    local max_attempts=2
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "  Installing $package (attempt $attempt/$max_attempts)..."
        
        # Try installation with timeout
        timeout 45 pip install --no-cache-dir --no-deps "$package" >/dev/null 2>&1
        
        if [ $? -eq 0 ]; then
            echo "  ✅ $package installed"
            return 0
        else
            echo "  ⚠️ Attempt $attempt failed for $package"
            attempt=$((attempt + 1))
        fi
    done
    
    echo "  ❌ Failed to install $package after $max_attempts attempts"
    return 1
}

# Export function for parallel execution
export -f install_package

# Activate venv if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Kill hanging pip processes
pkill -f pip 2>/dev/null

# Core packages needed for ComfyUI
echo ""
echo "📦 Installing core packages..."

# Group 1: Essential packages (install with dependencies)
ESSENTIAL_PACKAGES=(
    "torch>=2.1.0"
    "torchvision"
    "numpy==1.26.4"
)

echo "⚡ Installing essential packages..."
for pkg in "${ESSENTIAL_PACKAGES[@]}"; do
    pip install --no-cache-dir "$pkg" 2>/dev/null || echo "  ⚠️ $pkg failed"
done

# Group 2: Quick packages (no dependencies to speed up)
QUICK_PACKAGES=(
    "torchsde"
    "torchaudio"
    "scipy"
    "einops"
    "av"
    "safetensors"
    "opencv-python-headless"
    "Pillow"
    "tqdm"
    "psutil"
    "requests"
    "websocket-client"
)

echo ""
echo "⚡ Installing quick packages in parallel..."

# Install in parallel using background jobs
for package in "${QUICK_PACKAGES[@]}"; do
    (
        timeout 30 pip install --no-cache-dir --no-deps "$package" >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "  ✅ $package"
        else
            echo "  ⚠️ $package failed"
        fi
    ) &
    
    # Limit parallel jobs to 4
    while [ $(jobs -r | wc -l) -ge 4 ]; do
        sleep 0.5
    done
done

# Wait for all background jobs
wait

# Group 3: Heavy packages (install sequentially)
HEAVY_PACKAGES=(
    "transformers"
    "accelerate"
    "diffusers"
)

echo ""
echo "📦 Installing heavy packages..."
for pkg in "${HEAVY_PACKAGES[@]}"; do
    echo "  Installing $pkg..."
    timeout 60 pip install --no-cache-dir --no-deps "$pkg" >/dev/null 2>&1 || echo "  ⚠️ $pkg failed"
done

# Final verification
echo ""
echo "🔍 Quick verification..."
python -c "
import sys
try:
    import torch
    import torchsde
    import torchaudio
    import scipy
    import einops
    import av
    print('✅ Core packages verified!')
    sys.exit(0)
except ImportError as e:
    print(f'❌ Missing package: {e}')
    sys.exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Installation successful!"
    echo "=========================="
    echo "ComfyUI is ready to start"
else
    echo ""
    echo "⚠️ Some packages missing"
    echo "========================"
    echo "Run: ./scripts/quick_fix_dependencies.sh"
fi