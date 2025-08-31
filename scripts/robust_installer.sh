#!/bin/bash
# robust_installer.sh - Highly robust ComfyUI installation with extensive error handling

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging setup
LOG_DIR="/tmp/comfyui_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/install_$(date +%Y%m%d_%H%M%S).log"
ERROR_LOG="$LOG_DIR/errors_$(date +%Y%m%d_%H%M%S).log"

# Redirect stderr to error log
exec 2>"$ERROR_LOG"
exec 1> >(tee -a "$LOG_FILE")

echo -e "${GREEN}==> Robust ComfyUI Installation Starting...${NC}"
echo "==> Logs: $LOG_FILE"
echo "==> Errors: $ERROR_LOG"

# System check function
check_system() {
    echo -e "${YELLOW}==> Checking system requirements...${NC}"
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        echo "✅ Python $PYTHON_VERSION detected"
        
        if [[ "$PYTHON_VERSION" < "3.10" ]]; then
            echo -e "${RED}❌ Python 3.10+ required${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ Python not found${NC}"
        exit 1
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)
        echo "✅ GPU detected: $GPU_INFO"
    else
        echo -e "${YELLOW}⚠️ No NVIDIA GPU detected - CPU mode only${NC}"
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 10 ]; then
        echo -e "${RED}❌ Insufficient disk space (${AVAILABLE_SPACE}GB available, 10GB required)${NC}"
        exit 1
    else
        echo "✅ Disk space: ${AVAILABLE_SPACE}GB available"
    fi
}

# Network check with fallback
check_network() {
    echo "==> Checking network connectivity..."
    
    URLS=(
        "https://github.com"
        "https://pypi.org"
        "https://download.pytorch.org"
    )
    
    for url in "${URLS[@]}"; do
        if curl -s --head --max-time 5 "$url" > /dev/null; then
            echo "✅ $url reachable"
        else
            echo -e "${YELLOW}⚠️ $url not reachable${NC}"
        fi
    done
}

# Robust pip install with multiple fallbacks
robust_pip_install() {
    local package="$1"
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        echo "==> Installing $package (attempt $((retry+1))/$max_retries)..."
        
        # Try with cache
        if pip install --timeout 60 "$package" 2>/dev/null; then
            echo "✅ $package installed successfully"
            return 0
        fi
        
        # Try without cache
        if pip install --no-cache-dir --timeout 60 "$package" 2>/dev/null; then
            echo "✅ $package installed successfully (no cache)"
            return 0
        fi
        
        # Try with different index
        if pip install --index-url https://pypi.python.org/simple/ "$package" 2>/dev/null; then
            echo "✅ $package installed successfully (alt index)"
            return 0
        fi
        
        retry=$((retry + 1))
        [ $retry -lt $max_retries ] && sleep 5
    done
    
    echo -e "${YELLOW}⚠️ Failed to install $package after $max_retries attempts${NC}"
    return 1
}

# Install with progress tracking
install_with_progress() {
    local total_steps=10
    local current_step=0
    
    progress() {
        current_step=$((current_step + 1))
        local percent=$((current_step * 100 / total_steps))
        echo -e "${GREEN}[${current_step}/${total_steps}] Progress: ${percent}%${NC}"
    }
    
    # Step 1: System check
    progress
    check_system
    
    # Step 2: Network check
    progress
    check_network
    
    # Step 3: Create virtual environment
    progress
    echo "==> Creating virtual environment..."
    python3 -m venv venv || {
        echo -e "${RED}❌ Failed to create virtual environment${NC}"
        exit 1
    }
    source venv/bin/activate
    
    # Step 4: Upgrade pip
    progress
    echo "==> Upgrading pip..."
    python -m pip install --upgrade pip setuptools wheel
    
    # Step 5: Install PyTorch
    progress
    echo "==> Installing PyTorch..."
    if command -v nvidia-smi &> /dev/null; then
        robust_pip_install "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
    else
        robust_pip_install "torch torchvision torchaudio"
    fi
    
    # Step 6: Install xformers
    progress
    echo "==> Installing xformers..."
    robust_pip_install "xformers" || echo "⚠️ xformers optional, continuing..."
    
    # Step 7: Install ML packages
    progress
    echo "==> Installing ML packages..."
    for pkg in "numpy" "pillow" "scipy" "tqdm" "requests" "safetensors" "einops" "torchsde"; do
        robust_pip_install "$pkg"
    done
    
    # Step 8: Clone ComfyUI
    progress
    echo "==> Cloning ComfyUI..."
    if [ ! -d "ComfyUI" ]; then
        git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git || {
            echo "==> Trying full clone..."
            git clone https://github.com/comfyanonymous/ComfyUI.git
        }
    fi
    
    # Step 9: Install ComfyUI requirements
    progress
    if [ -f "ComfyUI/requirements.txt" ]; then
        echo "==> Installing ComfyUI requirements..."
        pip install -r ComfyUI/requirements.txt || echo "⚠️ Some requirements failed"
    fi
    
    # Step 10: Verify installation
    progress
    echo "==> Verifying installation..."
    python -c "
import torch
import numpy
import PIL
print('✅ Core packages verified')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
" || echo -e "${YELLOW}⚠️ Some packages missing${NC}"
}

# Cleanup on exit
cleanup() {
    echo -e "${YELLOW}==> Cleaning up...${NC}"
    # Compress logs if too large
    if [ -f "$LOG_FILE" ] && [ $(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null) -gt 1048576 ]; then
        gzip "$LOG_FILE"
        echo "==> Log compressed: ${LOG_FILE}.gz"
    fi
}

trap cleanup EXIT

# Main execution
main() {
    echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     Robust ComfyUI Installer v2.0     ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
    
    install_with_progress
    
    echo -e "${GREEN}✅ Installation completed!${NC}"
    echo "==> Logs saved to: $LOG_FILE"
    echo "==> Errors saved to: $ERROR_LOG"
    
    # Show next steps
    echo -e "${GREEN}==> Next steps:${NC}"
    echo "1. Activate environment: source venv/bin/activate"
    echo "2. Start ComfyUI: cd ComfyUI && python main.py --listen 0.0.0.0 --port 6006"
    echo "3. Access at: https://tensorboard-[PAPERSPACE_FQDN]/"
}

# Run main function
main "$@"