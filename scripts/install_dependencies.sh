#!/bin/bash
# install_dependencies.sh - Install ComfyUI and dependencies with robust error handling
# This script handles the initial setup of ComfyUI and all required dependencies

# Error handling
set -euo pipefail
trap 'echo "❌ Error occurred at line $LINENO. Exit code: $?"' ERR

# Logging
LOG_FILE="/tmp/comfyui_install_$(date +%Y%m%d_%H%M%S).log"
exec 2>&1 | tee -a "$LOG_FILE"

# Retry function
retry_command() {
    local max_attempts=3
    local delay=5
    local attempt=1
    local command="$@"
    
    while [ $attempt -le $max_attempts ]; do
        echo "==> Attempt $attempt/$max_attempts: $command"
        if eval "$command"; then
            echo "✅ Success"
            return 0
        fi
        
        if [ $attempt -lt $max_attempts ]; then
            echo "⚠️ Failed, retrying in ${delay}s..."
            sleep $delay
            delay=$((delay * 2))
        fi
        attempt=$((attempt + 1))
    done
    
    echo "❌ Failed after $max_attempts attempts"
    return 1
}

# Dependency check function
check_dependency() {
    local package=$1
    python -c "import $package" 2>/dev/null && echo "✅ $package installed" || echo "❌ $package missing"
}

install_dependencies() {
    echo "==> Starting robust dependency installation..."
    echo "==> Log file: $LOG_FILE"
    
    # Clone ComfyUI repository if not present
    echo "==> Checking for ComfyUI repository..."
    if [ ! -d "ComfyUI" ]; then
        echo "==> Cloning ComfyUI repository..."
        retry_command "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git" || {
            echo "==> Trying alternative clone method..."
            retry_command "git clone https://github.com/comfyanonymous/ComfyUI.git"
        }
        echo "==> ComfyUI repository cloned successfully"
    else
        echo "==> ComfyUI repository already exists, updating..."
        cd ComfyUI && git pull --rebase || echo "⚠️ Could not update ComfyUI" && cd ..
    fi
    
    # Create Python virtual environment if not present
    echo "==> Checking for Python virtual environment..."
    if [ ! -d "venv" ]; then
        echo "==> Creating Python virtual environment (3.12 preferred)..."
        # Try Python versions in order of preference: 3.12 > 3.11 > 3.10
        if command -v python3.12 &> /dev/null; then
            echo "==> Using Python 3.12 (recommended)"
            python3.12 -m venv venv
        elif command -v python3.11 &> /dev/null; then
            echo "==> Using Python 3.11 (good performance)"
            python3.11 -m venv venv
        elif command -v python3.10 &> /dev/null; then
            echo "==> Using Python 3.10 (fallback)"
            python3.10 -m venv venv
        else
            echo "==> Using default Python 3"
            python3 -m venv venv
        fi
        echo "==> Virtual environment created successfully"
    else
        echo "==> Virtual environment already exists, skipping creation"
    fi
    
    # Activate virtual environment
    echo "==> Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip to latest version
    echo "==> Upgrading pip to latest version..."
    pip install --upgrade pip
    
    # Install Python packages from requirements
    echo "==> Installing Python packages from requirements..."
    if [ -f "configs/python_requirements.txt" ]; then
        # Install build dependencies first
        echo "==> Installing build dependencies..."
        pip install --upgrade pip setuptools wheel ninja packaging
        
        # Detect Python version and environment
        PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        echo "==> Detected Python $PYTHON_VERSION"
        
        # Install PyTorch packages first with CUDA index (use -U for latest compatible versions)
        echo "==> Installing PyTorch packages with CUDA 12.4 support..."
        retry_command "pip install -U --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchvision" || {
            echo "==> Trying with specific versions as fallback..."
            retry_command "pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124"
        }
        
        # Verify PyTorch installation
        python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed')" || {
            echo "❌ PyTorch installation failed!"
            return 1
        }
        
        # Install xformers separately with flexible versioning to resolve conflicts
        echo "==> Installing xformers with automatic version resolution..."
        retry_command "pip install -U --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 xformers" || {
            echo "==> Trying xformers from PyPI..."
            retry_command "pip install -U --no-cache-dir xformers"
        }
        
        # CRITICAL: Install NumPy 1.x FIRST to prevent conflicts
        echo "==> Installing NumPy 1.x (preventing 2.x conflicts)..."
        pip uninstall -y numpy 2>/dev/null || true
        retry_command "pip install --no-cache-dir 'numpy<2.0' 'numpy==1.26.4'"
        
        # Install remaining packages from PyPI (default index)
        echo "==> Installing remaining ML packages from PyPI..."
        
        # Install in batches for better error handling with version constraints
        ML_PACKAGES=(
            "accelerate>=0.27.0"
            "transformers==4.36.2"  # Fixed version to prevent pytree error
            "safetensors>=0.4.0"
            "bitsandbytes>=0.41.0"
        )
        
        for package in "${ML_PACKAGES[@]}"; do
            retry_command "pip install --no-cache-dir $package" || echo "⚠️ Failed to install $package, continuing..."
        done
        
        # Essential packages with version constraints
        ESSENTIAL_PACKAGES=(
            "pillow>=10.0.0"
            "scipy<1.14"  # Prevent scipy issues
            "aiofiles>=22.1.0,<23"  # Fix ypy-websocket conflict
            "requests>=2.28.0"
            "tqdm>=4.64.0"
            "gpustat>=1.0.0"
            "pynvml>=11.4.0"
            "jupyterlab>=4.0.0"
        )
        
        for package in "${ESSENTIAL_PACKAGES[@]}"; do
            retry_command "pip install -U --no-cache-dir $package" || echo "⚠️ Failed to install $package, continuing..."
        done
        
        # Install missing ComfyUI dependencies
        echo "==> Installing ComfyUI-specific dependencies..."
        retry_command "pip install --no-cache-dir torchaudio --index-url https://download.pytorch.org/whl/cu124" || {
            echo "⚠️ torchaudio installation failed, trying without CUDA..."
            pip install --no-cache-dir torchaudio
        }
        
        retry_command "pip install --no-cache-dir torchsde" || echo "⚠️ torchsde installation failed"
        retry_command "pip install --no-cache-dir scipy einops" || echo "⚠️ scipy/einops installation failed"
        retry_command "pip install --no-cache-dir av" || echo "⚠️ av (PyAV) installation failed"
        
        # Install Flash Attention AFTER PyTorch if in CUDA environment
        if command -v nvcc &> /dev/null && [ -d "/usr/local/cuda" ]; then
            echo "==> CUDA environment detected - installing Flash Attention..."
            
            # Try pre-built wheels first (Python 3.12 with PyTorch 2.8)
            if [ "$PYTHON_VERSION" = "3.12" ]; then
                echo "==> Installing pre-built Flash Attention for Python 3.12..."
                pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.0/flash_attn-2.6.3+cu124torch2.8-cp312-cp312-linux_x86_64.whl || {
                    echo "==> Pre-built wheel failed, building from source..."
                    export CUDA_HOME=/usr/local/cuda
                    pip install flash-attn --no-build-isolation
                }
            else
                echo "==> Building Flash Attention from source for Python $PYTHON_VERSION..."
                export CUDA_HOME=/usr/local/cuda
                pip install flash-attn --no-build-isolation
            fi
        else
            echo "==> Non-CUDA environment detected - skipping Flash Attention"
            echo "==> Flash Attention wird in Paperspace automatisch installiert"
        fi
        echo "==> Python packages installed successfully"
    else
        echo "==> Warning: configs/python_requirements.txt not found, skipping Python package installation"
    fi
    
    # Install JupyterLab with root access support
    echo "==> Installing JupyterLab with root access support..."
    pip install jupyterlab jupyter-server-proxy
    echo "==> JupyterLab installed successfully"
    
    # Install custom nodes
    echo "==> Installing custom ComfyUI nodes..."
    if [ -f "configs/custom_nodes.txt" ]; then
        # Create custom_nodes directory if it doesn't exist
        mkdir -p ComfyUI/custom_nodes
        
        # Read each line from custom_nodes.txt
        while IFS= read -r line || [ -n "$line" ]; do
            # Skip empty lines and comments
            if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
                continue
            fi
            
            # Extract repository name from URL
            repo_name=$(basename "$line" .git)
            target_dir="ComfyUI/custom_nodes/$repo_name"
            
            echo "==> Processing custom node: $repo_name"
            
            # Clone repository if directory doesn't exist
            if [ ! -d "$target_dir" ]; then
                echo "==> Cloning $repo_name..."
                # Clone directly to target directory
                git clone "$line" "$target_dir" || {
                    echo "==> Clone failed for $repo_name, trying with timeout..."
                    timeout 60 git clone "$line" "$target_dir" || {
                        echo "==> Warning: Could not clone $repo_name, skipping..."
                        continue
                    }
                }
                echo "==> Successfully cloned $repo_name"
            else
                echo "==> $repo_name already exists, updating..."
                cd "$target_dir" && git pull || echo "⚠️ Could not update $repo_name"
                cd - > /dev/null
            fi
            
            # ALWAYS install/update requirements for each custom node
            if [ -f "$target_dir/requirements.txt" ]; then
                echo "==> Installing requirements for $repo_name..."
                pip install -r "$target_dir/requirements.txt" || {
                    echo "⚠️ Some requirements for $repo_name failed, trying with upgrade..."
                    pip install -U -r "$target_dir/requirements.txt" || echo "⚠️ Requirements installation had issues"
                }
                echo "==> Requirements processing completed for $repo_name"
            else
                echo "==> No requirements.txt found for $repo_name"
            fi
        done < configs/custom_nodes.txt
        
        echo "==> Custom nodes installation completed"
    else
        echo "==> Warning: configs/custom_nodes.txt not found, skipping custom nodes installation"
    fi
    
    # Final verification
    echo "==> Verifying critical dependencies..."
    local failed_deps=0
    
    CRITICAL_DEPS=(
        "torch"
        "torchvision"
        "numpy"
        "PIL"
        "tqdm"
        "requests"
    )
    
    for dep in "${CRITICAL_DEPS[@]}"; do
        if ! python -c "import $dep" 2>/dev/null; then
            echo "❌ Critical dependency missing: $dep"
            ((failed_deps++))
        else
            echo "✅ $dep verified"
        fi
    done
    
    if [ $failed_deps -gt 0 ]; then
        echo "⚠️ $failed_deps critical dependencies missing. ComfyUI may not work properly."
        echo "==> Run ./scripts/fix_dependencies.sh to repair"
    else
        echo "✅ All critical dependencies installed successfully"
    fi
    
    echo "==> Dependency installation completed"
    echo "==> Full log available at: $LOG_FILE"
}