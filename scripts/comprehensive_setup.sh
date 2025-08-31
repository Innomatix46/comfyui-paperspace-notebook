#!/bin/bash
# comprehensive_setup.sh - Complete ComfyUI setup with all packages and optimizations

set -euo pipefail

echo "🚀 COMPREHENSIVE COMFYUI SETUP"
echo "=============================="
echo "This installs ALL required and optional packages for maximum compatibility"
echo

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    echo "==> Activating virtual environment..."
    source venv/bin/activate
fi

# Function to safely install packages
safe_install() {
    local package="$1"
    echo "Installing $package..."
    pip install -U "$package" || pip install --ignore-installed -U "$package" || echo "⚠️ $package installation had issues"
}

# 1. CORE SYSTEM PACKAGES
echo "==> Installing core system packages..."
safe_install "setuptools"
safe_install "pip"
safe_install "wheel"
safe_install "ninja"

# 2. PYTORCH ECOSYSTEM (Critical - specific order)
echo "==> Setting up PyTorch ecosystem..."
# Uninstall old torch to avoid conflicts
pip uninstall torch torchvision torchaudio xformers -y 2>/dev/null || true

# Install PyTorch 2.6 with CUDA 12.4
echo "Installing PyTorch 2.6 with CUDA 12.4..."
pip install --pre torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install xformers and related
pip install -U xformers --index-url https://download.pytorch.org/whl/cu124
pip install -U typing_extensions torchsde triton --index-url https://download.pytorch.org/whl/cu124

# 3. ML/AI PACKAGES
echo "==> Installing ML/AI packages..."
safe_install "transformers==4.49.0"
safe_install "bitsandbytes"
safe_install "accelerate"
safe_install "optimum"
safe_install "sageattention"

# 4. AUDIO/VIDEO PROCESSING
echo "==> Installing audio/video packages..."
safe_install "audio-separator"
safe_install "pedalboard"
safe_install "av"
safe_install "ffmpeg-python"
safe_install "python-ffmpeg-video-streaming"

# Install ffmpeg system package
echo "Installing ffmpeg system package..."
apt-get update && apt-get install -y ffmpeg || echo "⚠️ ffmpeg system package needs sudo"

# 5. IMAGE PROCESSING & COMPUTER VISION
echo "==> Installing image processing packages..."
safe_install "opencv-python-headless"
safe_install "matplotlib"
safe_install "scikit-image"
safe_install "Pillow"
safe_install "blend-modes"
safe_install "albucore==0.0.16"
safe_install "mmcv"
safe_install "albumentations"

# 6. SPECIALIZED PACKAGES
echo "==> Installing specialized packages..."
safe_install "LangSegment"
safe_install "onnx"
safe_install "onnx2torch"
safe_install "scepter"
safe_install "blinker"

# 7. NUMPY (Special handling for version)
echo "==> Setting up numpy..."
pip uninstall numpy -y 2>/dev/null || true
pip install numpy==1.26.4  # Compatible version for most packages

# 8. DEVELOPMENT & NOTEBOOK PACKAGES
echo "==> Installing development packages..."
safe_install "ipykernel"
safe_install "paperspace"
safe_install "jupyterlab"

# 9. FLASH ATTENTION (Complex installation)
echo "==> Installing Flash Attention..."
install_flash_attention() {
    local PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
    
    if [ "$PYTHON_VERSION" = "310" ]; then
        echo "Python 3.10 detected - using pre-built wheel..."
        wget -P /tmp -N https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
        pip install --no-dependencies --force-reinstall --upgrade /tmp/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    elif [ "$PYTHON_VERSION" = "311" ]; then
        echo "Python 3.11 detected - using pre-built wheel..."
        wget -P /tmp -N https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
        pip install --no-dependencies --force-reinstall --upgrade /tmp/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
    elif [ "$PYTHON_VERSION" = "312" ]; then
        echo "Python 3.12 detected - using pre-built wheel..."
        wget -P /tmp -N https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.0/flash_attn-2.6.3+cu124torch2.8-cp312-cp312-linux_x86_64.whl
        pip install /tmp/flash_attn-2.6.3+cu124torch2.8-cp312-cp312-linux_x86_64.whl
    else
        echo "Building Flash Attention from source..."
        pip install -U flash-attn --no-build-isolation
    fi
}

# Try to install Flash Attention
if command -v nvcc &> /dev/null; then
    install_flash_attention || echo "⚠️ Flash Attention installation failed"
else
    echo "⚠️ CUDA not detected, skipping Flash Attention"
fi

# 10. ADDITIONAL COMFYUI DEPENDENCIES
echo "==> Installing additional ComfyUI dependencies..."
safe_install "scipy"
safe_install "einops"
safe_install "tqdm"
safe_install "requests"
safe_install "safetensors"
safe_install "psutil"
safe_install "kornia"

# 11. VERIFY CRITICAL INSTALLATIONS
echo "==> Verifying installations..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} with CUDA {torch.cuda.is_available()}')"
python -c "import xformers; print(f'✅ xformers {xformers.__version__}')" || echo "⚠️ xformers not available"
python -c "import transformers; print(f'✅ transformers {transformers.__version__}')"
python -c "import cv2; print('✅ OpenCV installed')" || echo "⚠️ OpenCV not available"
python -c "import av; print('✅ PyAV installed')" || echo "⚠️ PyAV not available"

echo
echo "✅ Comprehensive setup completed!"
echo "Some packages may have warnings - this is normal for experimental packages."
echo "ComfyUI should now have maximum compatibility with all nodes and features."