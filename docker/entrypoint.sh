#!/bin/bash

# Docker Entrypoint Script for ComfyUI on Paperspace
# Handles initialization, storage management, and service startup

set -euo pipefail

# Configuration
COMFYUI_PATH="${COMFYUI_PATH:-/app/ComfyUI}"
MAX_STORAGE_GB="${MAX_STORAGE_GB:-45}"
AUTO_CLEANUP="${AUTO_CLEANUP:-true}"
PAPERSPACE_FQDN="${PAPERSPACE_FQDN:-}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INIT]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Signal handlers for graceful shutdown
shutdown_handler() {
    log_info "Received shutdown signal, cleaning up..."
    
    # Kill background processes
    if [ -n "${COMFYUI_PID:-}" ]; then
        kill "$COMFYUI_PID" 2>/dev/null || true
        wait "$COMFYUI_PID" 2>/dev/null || true
    fi
    
    if [ -n "${TENSORBOARD_PID:-}" ]; then
        kill "$TENSORBOARD_PID" 2>/dev/null || true
    fi
    
    log_info "Shutdown complete"
    exit 0
}

trap shutdown_handler SIGTERM SIGINT

# Check if running on Paperspace
check_paperspace() {
    if [[ -n "${PAPERSPACE_NOTEBOOK_REPO_ID:-}" ]] || [[ -n "$PAPERSPACE_FQDN" ]] || [[ "$HOSTNAME" == *"paperspace"* ]]; then
        log_info "Running on Paperspace environment"
        export PAPERSPACE_MODE=true
        return 0
    else
        log_info "Running in local/cloud environment"
        export PAPERSPACE_MODE=false
        return 1
    fi
}

# Detect available storage
detect_storage() {
    log_info "Detecting storage configuration..."
    
    # Get total and available disk space
    local total_gb
    local available_gb
    local used_gb
    
    total_gb=$(df -BG / | awk 'NR==2 {print $2}' | sed 's/G//')
    available_gb=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    used_gb=$(df -BG / | awk 'NR==2 {print $3}' | sed 's/G//')
    
    log_info "Storage: ${used_gb}GB used / ${total_gb}GB total (${available_gb}GB available)"
    
    # Set storage limits based on detection
    if [ "$total_gb" -le 60 ] && [ "$PAPERSPACE_MODE" = "true" ]; then
        log_warning "Detected Paperspace Free Tier (50GB limit)"
        export MAX_STORAGE_GB=45
        export STORAGE_CONSTRAINED=true
    else
        export STORAGE_CONSTRAINED=false
    fi
    
    # Check if approaching storage limits
    if [ "$used_gb" -gt "$MAX_STORAGE_GB" ]; then
        log_error "Storage usage (${used_gb}GB) exceeds limit (${MAX_STORAGE_GB}GB)"
        if [ "$AUTO_CLEANUP" = "true" ]; then
            cleanup_storage
        else
            log_error "Auto-cleanup disabled, manual intervention required"
        fi
    fi
}

# Cleanup storage to free space
cleanup_storage() {
    log_info "Running storage cleanup..."
    
    # Clean temporary files
    find /tmp -type f -mtime +1 -delete 2>/dev/null || true
    find /var/tmp -type f -mtime +1 -delete 2>/dev/null || true
    
    # Clean ComfyUI temp files
    if [ -d "$COMFYUI_PATH/temp" ]; then
        find "$COMFYUI_PATH/temp" -type f -mtime +1 -delete 2>/dev/null || true
    fi
    
    # Clean old outputs (keep last 7 days)
    if [ -d "$COMFYUI_PATH/output" ]; then
        find "$COMFYUI_PATH/output" -type f -mtime +7 -delete 2>/dev/null || true
    fi
    
    # Clean Python cache
    find /app -name "*.pyc" -delete 2>/dev/null || true
    find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    log_success "Storage cleanup completed"
}

# Create necessary directories
create_directories() {
    log_info "Creating required directories..."
    
    local dirs=(
        "$COMFYUI_PATH/models/checkpoints"
        "$COMFYUI_PATH/models/vae"
        "$COMFYUI_PATH/models/loras"
        "$COMFYUI_PATH/models/controlnet"
        "$COMFYUI_PATH/models/clip_vision"
        "$COMFYUI_PATH/models/style_models"
        "$COMFYUI_PATH/models/upscale_models"
        "$COMFYUI_PATH/models/embeddings"
        "$COMFYUI_PATH/output"
        "$COMFYUI_PATH/input"
        "$COMFYUI_PATH/temp"
        "$COMFYUI_PATH/logs"
        "$COMFYUI_PATH/user"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    # Set proper permissions
    chmod -R 755 "$COMFYUI_PATH"/{models,output,input,temp,logs,user} 2>/dev/null || true
    
    log_success "Directories created and permissions set"
}

# Create symlinks for mounted volumes
create_symlinks() {
    log_info "Creating symlinks for mounted volumes..."
    
    # Model symlinks (if external models are mounted)
    local model_dirs=(
        "checkpoints"
        "vae" 
        "loras"
        "controlnet"
        "clip_vision"
        "upscale_models"
        "embeddings"
    )
    
    for model_dir in "${model_dirs[@]}"; do
        local mount_path="/mnt/models/$model_dir"
        local target_path="$COMFYUI_PATH/models/$model_dir"
        
        if [ -d "$mount_path" ] && [ ! -L "$target_path" ]; then
            rm -rf "$target_path"
            ln -sf "$mount_path" "$target_path"
            log_info "Created symlink: $target_path -> $mount_path"
        fi
    done
    
    # Output symlink (if external output is mounted)
    if [ -d "/mnt/output" ] && [ ! -L "$COMFYUI_PATH/output" ]; then
        rm -rf "$COMFYUI_PATH/output"
        ln -sf "/mnt/output" "$COMFYUI_PATH/output"
        log_info "Created output symlink"
    fi
}

# Check GPU availability
check_gpu() {
    log_info "Checking GPU availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            local gpu_info
            gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
            log_success "GPU detected: $gpu_info"
            export GPU_AVAILABLE=true
            export CUDA_AVAILABLE=true
        else
            log_warning "NVIDIA drivers available but no GPU detected"
            export GPU_AVAILABLE=false
        fi
    else
        log_warning "NVIDIA drivers not available, using CPU mode"
        export GPU_AVAILABLE=false
        export CUDA_AVAILABLE=false
    fi
}

# Check Python environment
check_python_env() {
    log_info "Checking Python environment..."
    
    # Verify Python version
    local python_version
    python_version=$(python --version 2>&1 | cut -d' ' -f2)
    log_info "Python version: $python_version"
    
    # Check PyTorch installation
    if python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        log_success "PyTorch available"
        
        # Check CUDA in PyTorch
        if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            local cuda_version
            cuda_version=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
            log_success "CUDA available in PyTorch (version: $cuda_version)"
        else
            log_warning "CUDA not available in PyTorch"
        fi
    else
        log_error "PyTorch not available"
        exit 1
    fi
    
    # Check other critical packages
    local packages=(
        "transformers"
        "diffusers"
        "safetensors"
        "PIL"
        "cv2"
    )
    
    for package in "${packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            log_info "Package available: $package"
        else
            log_warning "Package missing: $package"
        fi
    done
}

# Initialize ComfyUI configuration
init_comfyui_config() {
    log_info "Initializing ComfyUI configuration..."
    
    # Create extra_model_paths.yaml if it doesn't exist
    local config_file="$COMFYUI_PATH/extra_model_paths.yaml"
    
    if [ ! -f "$config_file" ]; then
        cat > "$config_file" << EOF
# Extra model paths configuration for ComfyUI Docker
# This file maps model directories to custom locations

# Base model paths
base_path: /app/ComfyUI/

# Model type mappings
checkpoints: models/checkpoints/
vae: models/vae/
loras: models/loras/
controlnet: models/controlnet/
clip_vision: models/clip_vision/
style_models: models/style_models/
upscale_models: models/upscale_models/
embeddings: models/embeddings/

# Custom node paths
custom_nodes: custom_nodes/

# Output configuration
output_directory: output/
temp_directory: temp/
input_directory: input/

# Cache settings
model_cache_size: ${MODEL_CACHE_SIZE:-20}
enable_model_caching: true
auto_cleanup_temp: ${AUTO_CLEANUP:-true}

# Performance settings
async_model_loading: true
low_vram_mode: ${LOW_VRAM_MODE:-false}
cpu_offload: ${CPU_OFFLOAD:-false}
EOF
        log_success "Created ComfyUI configuration: $config_file"
    fi
    
    # Create web configuration
    local web_config="$COMFYUI_PATH/web/user.css"
    if [ ! -f "$web_config" ] && [ -d "$COMFYUI_PATH/web" ]; then
        mkdir -p "$(dirname "$web_config")"
        cat > "$web_config" << EOF
/* Custom CSS for ComfyUI in Docker */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.comfy-modal {
    backdrop-filter: blur(10px);
}
EOF
        log_info "Created web configuration"
    fi
}

# Start Tensorboard service
start_tensorboard() {
    if [ ! -d "$COMFYUI_PATH/logs" ]; then
        mkdir -p "$COMFYUI_PATH/logs"
    fi
    
    log_info "Starting Tensorboard service..."
    
    # Start Tensorboard in background
    tensorboard --logdir="$COMFYUI_PATH/logs" \
                --bind_all \
                --port=6006 \
                --reload_interval=10 \
                --window_title="ComfyUI Training Logs" \
                > /dev/null 2>&1 &
    
    TENSORBOARD_PID=$!
    
    # Wait a moment and check if it's running
    sleep 2
    if kill -0 "$TENSORBOARD_PID" 2>/dev/null; then
        log_success "Tensorboard started (PID: $TENSORBOARD_PID)"
        
        # Show Paperspace URL
        if [ "$PAPERSPACE_MODE" = "true" ] && [ -n "$PAPERSPACE_FQDN" ]; then
            log_info "Tensorboard URL: https://tensorboard-${PAPERSPACE_FQDN}/"
        else
            log_info "Tensorboard URL: http://localhost:6006"
        fi
    else
        log_warning "Tensorboard failed to start"
        unset TENSORBOARD_PID
    fi
}

# Health monitoring function
start_health_monitor() {
    log_info "Starting health monitoring..."
    
    (
        while true; do
            sleep 60
            
            # Check storage usage
            if [ "$STORAGE_CONSTRAINED" = "true" ]; then
                local used_gb
                used_gb=$(df -BG / | awk 'NR==2 {print $3}' | sed 's/G//')
                
                if [ "$used_gb" -gt "$MAX_STORAGE_GB" ]; then
                    log_warning "Storage limit exceeded: ${used_gb}GB > ${MAX_STORAGE_GB}GB"
                    if [ "$AUTO_CLEANUP" = "true" ]; then
                        cleanup_storage
                    fi
                fi
            fi
            
            # Check ComfyUI process
            if [ -n "${COMFYUI_PID:-}" ] && ! kill -0 "$COMFYUI_PID" 2>/dev/null; then
                log_error "ComfyUI process died unexpectedly"
                exit 1
            fi
            
        done
    ) &
    
    HEALTH_MONITOR_PID=$!
    log_success "Health monitoring started (PID: $HEALTH_MONITOR_PID)"
}

# Start ComfyUI application
start_comfyui() {
    log_info "Starting ComfyUI application..."
    
    cd "$COMFYUI_PATH"
    
    # Build command line arguments
    local comfyui_args=(
        "--listen" "0.0.0.0"
        "--port" "8188"
    )
    
    # Add GPU-specific arguments
    if [ "$GPU_AVAILABLE" = "true" ]; then
        comfyui_args+=("--cuda-device" "0")
        
        # Check available VRAM and adjust settings
        local vram_gb
        if command -v nvidia-smi &> /dev/null; then
            vram_gb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
            vram_gb=$((vram_gb / 1024))
            
            if [ "$vram_gb" -lt 8 ]; then
                comfyui_args+=("--lowvram")
                log_info "Low VRAM mode enabled (${vram_gb}GB detected)"
            elif [ "$vram_gb" -ge 24 ]; then
                log_info "High VRAM mode (${vram_gb}GB detected)"
            fi
        fi
    else
        comfyui_args+=("--cpu")
        log_info "CPU mode enabled"
    fi
    
    # Add other configuration
    comfyui_args+=("--output-directory" "$COMFYUI_PATH/output")
    
    if [ "$STORAGE_CONSTRAINED" = "true" ]; then
        comfyui_args+=("--temp-directory" "$COMFYUI_PATH/temp")
        log_info "Storage-constrained mode enabled"
    fi
    
    log_info "Starting ComfyUI with args: ${comfyui_args[*]}"
    
    # Start ComfyUI
    python main.py "${comfyui_args[@]}" &
    COMFYUI_PID=$!
    
    # Wait a moment and check if it started successfully
    sleep 5
    if kill -0 "$COMFYUI_PID" 2>/dev/null; then
        log_success "ComfyUI started successfully (PID: $COMFYUI_PID)"
        
        # Show access URL
        if [ "$PAPERSPACE_MODE" = "true" ] && [ -n "$PAPERSPACE_FQDN" ]; then
            log_info "ComfyUI URL: https://8188-${PAPERSPACE_FQDN}/"
        else
            log_info "ComfyUI URL: http://localhost:8188"
        fi
        
        return 0
    else
        log_error "ComfyUI failed to start"
        return 1
    fi
}

# Main initialization function
main() {
    log_info "Starting ComfyUI Docker container initialization..."
    
    # Initialization steps
    check_paperspace
    detect_storage
    create_directories
    create_symlinks
    check_gpu
    check_python_env
    init_comfyui_config
    
    # Start services
    start_tensorboard
    start_health_monitor
    
    # Start main application
    if start_comfyui; then
        log_success "All services started successfully"
        
        # Show final status
        log_info "=== Service Status ==="
        log_info "ComfyUI: Running (PID: $COMFYUI_PID)"
        log_info "Tensorboard: ${TENSORBOARD_PID:+Running (PID: $TENSORBOARD_PID)}"
        log_info "Health Monitor: Running (PID: $HEALTH_MONITOR_PID)"
        log_info "Storage: ${STORAGE_CONSTRAINED:+Constrained (${MAX_STORAGE_GB}GB limit)}"
        log_info "GPU: ${GPU_AVAILABLE:+Available}"
        
        # Wait for ComfyUI process
        log_info "Waiting for ComfyUI process..."
        wait "$COMFYUI_PID"
    else
        log_error "Failed to start ComfyUI"
        exit 1
    fi
}

# Handle different execution modes
if [ $# -eq 0 ]; then
    # Default mode: full initialization
    main
else
    # Execute provided command
    log_info "Executing command: $*"
    exec "$@"
fi