#!/bin/bash

# Docker Build Cache Script for ComfyUI on Paperspace
# Optimizes build process with layer caching and registry management
# Reduces rebuild time from 15-20 minutes to 1-2 minutes

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_CONTEXT="$SCRIPT_DIR"
IMAGE_NAME="comfyui"
IMAGE_TAG="paperspace-optimized"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
REGISTRY_URL="${DOCKER_REGISTRY:-}"
BUILD_CACHE_DIR="${HOME}/.docker-cache/comfyui"
MAX_CACHE_SIZE_GB=10

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
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

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check for NVIDIA Docker runtime
    if ! docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        log_warning "NVIDIA Docker runtime not available or no GPU detected"
    else
        log_success "NVIDIA Docker runtime detected"
    fi
    
    log_success "All dependencies satisfied"
}

# Cleanup old cache
cleanup_cache() {
    log_info "Cleaning up Docker cache..."
    
    # Remove dangling images
    docker image prune -f || true
    
    # Remove unused build cache (keep recent ones)
    docker builder prune --filter until=168h -f || true
    
    # Check cache size
    if [ -d "$BUILD_CACHE_DIR" ]; then
        cache_size=$(du -s "$BUILD_CACHE_DIR" 2>/dev/null | cut -f1 || echo "0")
        cache_size_gb=$((cache_size / 1024 / 1024))
        
        if [ "$cache_size_gb" -gt "$MAX_CACHE_SIZE_GB" ]; then
            log_warning "Cache size (${cache_size_gb}GB) exceeds limit (${MAX_CACHE_SIZE_GB}GB)"
            log_info "Cleaning cache directory..."
            find "$BUILD_CACHE_DIR" -type f -mtime +7 -delete || true
        fi
    fi
    
    log_success "Cache cleanup completed"
}

# Pre-download models to cache
predownload_models() {
    log_info "Pre-downloading essential models..."
    
    mkdir -p "$BUILD_CACHE_DIR/models"
    cd "$BUILD_CACHE_DIR/models"
    
    # Define model URLs and filenames
    declare -A MODELS=(
        ["vae/vae-ft-mse-840000-ema-pruned.safetensors"]="https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors"
        ["upscale_models/RealESRGAN_x4plus.pth"]="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        ["controlnet/canny.safetensors"]="https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/diffusion_pytorch_model.safetensors"
    )
    
    for file_path in "${!MODELS[@]}"; do
        url="${MODELS[$file_path]}"
        dir_path=$(dirname "$file_path")
        filename=$(basename "$file_path")
        
        mkdir -p "$dir_path"
        
        if [ ! -f "$file_path" ]; then
            log_info "Downloading $filename..."
            if wget -q --show-progress -O "$file_path" "$url"; then
                log_success "Downloaded $filename"
            else
                log_warning "Failed to download $filename"
                rm -f "$file_path"
            fi
        else
            log_info "$filename already cached"
        fi
    done
    
    log_success "Model pre-download completed"
}

# Build Docker image with optimized caching
build_image() {
    local push_to_registry="${1:-false}"
    
    log_info "Building Docker image with optimized caching..."
    
    # Create build cache directory
    mkdir -p "$BUILD_CACHE_DIR"
    
    # Set build arguments
    BUILD_ARGS=(
        --build-arg BUILDKIT_INLINE_CACHE=1
        --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        --build-arg VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
        --cache-from "${FULL_IMAGE_NAME}"
        --tag "${FULL_IMAGE_NAME}"
        --file "${DOCKER_CONTEXT}/Dockerfile.comfyui"
        --target production
        --progress plain
    )
    
    # Add registry cache if available
    if [ -n "$REGISTRY_URL" ]; then
        BUILD_ARGS+=(--cache-from "${REGISTRY_URL}/${FULL_IMAGE_NAME}")
    fi
    
    # Enable BuildKit for better caching
    export DOCKER_BUILDKIT=1
    
    # Build the image
    log_info "Starting Docker build..."
    if docker build "${BUILD_ARGS[@]}" "$DOCKER_CONTEXT"; then
        log_success "Docker image built successfully: $FULL_IMAGE_NAME"
    else
        log_error "Docker build failed"
        exit 1
    fi
    
    # Push to registry if requested
    if [ "$push_to_registry" = "true" ] && [ -n "$REGISTRY_URL" ]; then
        push_image
    fi
}

# Push image to registry
push_image() {
    if [ -z "$REGISTRY_URL" ]; then
        log_warning "No registry URL configured, skipping push"
        return 0
    fi
    
    log_info "Pushing image to registry..."
    
    # Tag for registry
    REGISTRY_IMAGE="${REGISTRY_URL}/${FULL_IMAGE_NAME}"
    docker tag "$FULL_IMAGE_NAME" "$REGISTRY_IMAGE"
    
    # Push image
    if docker push "$REGISTRY_IMAGE"; then
        log_success "Image pushed to registry: $REGISTRY_IMAGE"
    else
        log_error "Failed to push image to registry"
        exit 1
    fi
}

# Test built image
test_image() {
    log_info "Testing built image..."
    
    # Basic container test
    if docker run --rm --name comfyui-test "$FULL_IMAGE_NAME" python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"; then
        log_success "Image test passed"
    else
        log_error "Image test failed"
        exit 1
    fi
}

# Show build information
show_info() {
    log_info "Build Information:"
    echo "  Image: $FULL_IMAGE_NAME"
    echo "  Registry: ${REGISTRY_URL:-'Not configured'}"
    echo "  Cache Directory: $BUILD_CACHE_DIR"
    echo "  Docker Context: $DOCKER_CONTEXT"
    
    if docker image inspect "$FULL_IMAGE_NAME" &>/dev/null; then
        echo "  Image Size: $(docker image inspect "$FULL_IMAGE_NAME" --format '{{.Size}}' | numfmt --to=iec)"
        echo "  Created: $(docker image inspect "$FULL_IMAGE_NAME" --format '{{.Created}}')"
    fi
}

# Main execution
main() {
    log_info "Starting ComfyUI Docker build cache script..."
    
    case "${1:-build}" in
        "check")
            check_dependencies
            ;;
        "clean")
            cleanup_cache
            ;;
        "models")
            predownload_models
            ;;
        "build")
            check_dependencies
            cleanup_cache
            predownload_models
            build_image "${2:-false}"
            test_image
            show_info
            ;;
        "push")
            check_dependencies
            build_image "true"
            ;;
        "test")
            test_image
            ;;
        "info")
            show_info
            ;;
        "full")
            check_dependencies
            cleanup_cache
            predownload_models
            build_image "true"
            test_image
            show_info
            ;;
        *)
            echo "Usage: $0 {check|clean|models|build|push|test|info|full} [push]"
            echo ""
            echo "Commands:"
            echo "  check  - Check dependencies"
            echo "  clean  - Cleanup cache and unused images"
            echo "  models - Pre-download essential models"
            echo "  build  - Build Docker image (add 'push' to push to registry)"
            echo "  push   - Build and push to registry"
            echo "  test   - Test built image"
            echo "  info   - Show build information"
            echo "  full   - Run complete build pipeline with registry push"
            echo ""
            echo "Environment Variables:"
            echo "  DOCKER_REGISTRY - Registry URL for pushing images"
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"