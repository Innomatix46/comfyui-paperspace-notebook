#!/bin/bash

# Docker Quick Start Script for ComfyUI on Paperspace
# Fast deployment using pre-built Docker images
# Reduces startup time from 15-20 minutes to 1-2 minutes

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$PROJECT_ROOT/docker"
DATA_DIR="$PROJECT_ROOT/data"

# Docker configuration
IMAGE_NAME="comfyui:paperspace-optimized"
CONTAINER_NAME="comfyui-paperspace"
REGISTRY_URL="${DOCKER_REGISTRY:-}"
COMPOSE_FILE="$DOCKER_DIR/docker-compose.yml"

# Paperspace configuration
PAPERSPACE_FQDN="${PAPERSPACE_FQDN:-$(hostname -f 2>/dev/null || echo 'localhost')}"
COMFYUI_PORT="${COMFYUI_PORT:-8188}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"

# Storage configuration
MAX_STORAGE_GB="${MAX_STORAGE_GB:-45}"
STORAGE_CHECK_ENABLED="${STORAGE_CHECK_ENABLED:-true}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

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

log_header() {
    echo -e "${PURPLE}[DOCKER]${NC} $1"
}

# Check if running on Paperspace
check_paperspace() {
    if [[ -n "${PAPERSPACE_NOTEBOOK_REPO_ID:-}" ]] || [[ -n "${PAPERSPACE_FQDN:-}" ]] || [[ "$HOSTNAME" == *"paperspace"* ]]; then
        log_info "Detected Paperspace environment"
        return 0
    else
        log_warning "Not running on Paperspace, using local configuration"
        return 1
    fi
}

# Check storage usage
check_storage() {
    if [ "$STORAGE_CHECK_ENABLED" != "true" ]; then
        return 0
    fi
    
    log_info "Checking storage usage..."
    
    # Get disk usage
    local usage_gb
    usage_gb=$(df -BG / | awk 'NR==2 {print $3}' | sed 's/G//')
    
    if [ "$usage_gb" -gt "$MAX_STORAGE_GB" ]; then
        log_error "Storage usage (${usage_gb}GB) exceeds limit (${MAX_STORAGE_GB}GB)"
        log_info "Running cleanup..."
        docker system prune -f || true
        docker volume prune -f || true
    else
        log_success "Storage usage: ${usage_gb}GB / ${MAX_STORAGE_GB}GB"
    fi
}

# Setup data directories
setup_directories() {
    log_info "Setting up data directories..."
    
    # Create required directories
    local dirs=(
        "$DATA_DIR/models/checkpoints"
        "$DATA_DIR/models/vae"
        "$DATA_DIR/models/loras"
        "$DATA_DIR/models/controlnet"
        "$DATA_DIR/models/clip_vision"
        "$DATA_DIR/models/upscale_models"
        "$DATA_DIR/models/embeddings"
        "$DATA_DIR/output"
        "$DATA_DIR/input"
        "$DATA_DIR/temp"
        "$DATA_DIR/logs"
        "$DATA_DIR/cache"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        chmod 755 "$dir"
    done
    
    # Set ownership if running as root
    if [ "$(id -u)" -eq 0 ]; then
        chown -R 1000:1000 "$DATA_DIR"
    fi
    
    log_success "Data directories created"
}

# Check if image exists
check_image() {
    log_info "Checking for Docker image..."
    
    if docker image inspect "$IMAGE_NAME" &>/dev/null; then
        log_success "Docker image found: $IMAGE_NAME"
        return 0
    else
        log_warning "Docker image not found: $IMAGE_NAME"
        return 1
    fi
}

# Pull image from registry
pull_image() {
    if [ -z "$REGISTRY_URL" ]; then
        log_warning "No registry configured, cannot pull image"
        return 1
    fi
    
    log_info "Pulling image from registry..."
    
    local registry_image="${REGISTRY_URL}/${IMAGE_NAME}"
    
    if docker pull "$registry_image"; then
        docker tag "$registry_image" "$IMAGE_NAME"
        log_success "Image pulled and tagged: $IMAGE_NAME"
        return 0
    else
        log_error "Failed to pull image from registry"
        return 1
    fi
}

# Build image locally
build_image() {
    log_info "Building Docker image locally..."
    
    if [ ! -f "$DOCKER_DIR/build_cache.sh" ]; then
        log_error "Build script not found: $DOCKER_DIR/build_cache.sh"
        return 1
    fi
    
    cd "$DOCKER_DIR"
    if bash build_cache.sh build; then
        log_success "Image built successfully"
        return 0
    else
        log_error "Failed to build image"
        return 1
    fi
}

# Ensure image is available
ensure_image() {
    if ! check_image; then
        log_info "Attempting to acquire Docker image..."
        
        # Try pulling from registry first
        if ! pull_image; then
            log_info "Registry pull failed, building locally..."
            if ! build_image; then
                log_error "Failed to acquire Docker image"
                exit 1
            fi
        fi
    fi
}

# Stop existing container
stop_container() {
    log_info "Checking for existing container..."
    
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        log_info "Stopping existing container..."
        docker stop "$CONTAINER_NAME" || true
    fi
    
    if docker ps -aq -f name="$CONTAINER_NAME" | grep -q .; then
        log_info "Removing existing container..."
        docker rm "$CONTAINER_NAME" || true
    fi
}

# Start container with Docker Compose
start_with_compose() {
    log_info "Starting services with Docker Compose..."
    
    # Set environment variables
    export PAPERSPACE_FQDN
    export PWD="$PROJECT_ROOT"
    
    cd "$DOCKER_DIR"
    
    if docker-compose -f "$COMPOSE_FILE" up -d; then
        log_success "Services started with Docker Compose"
        return 0
    else
        log_error "Failed to start with Docker Compose"
        return 1
    fi
}

# Start container manually
start_container() {
    log_info "Starting ComfyUI container manually..."
    
    # Docker run parameters
    local docker_args=(
        --name "$CONTAINER_NAME"
        --detach
        --restart unless-stopped
        --gpus all
        --publish "${COMFYUI_PORT}:8188"
        --publish "${TENSORBOARD_PORT}:6006"
        --volume "$DATA_DIR/models:/app/ComfyUI/models:rw"
        --volume "$DATA_DIR/output:/app/ComfyUI/output:rw"
        --volume "$DATA_DIR/input:/app/ComfyUI/input:rw"
        --volume "$DATA_DIR/temp:/app/ComfyUI/temp:rw"
        --volume "$DATA_DIR/logs:/app/ComfyUI/logs:rw"
        --volume "$DATA_DIR/cache:/app/.cache:rw"
        --env "PAPERSPACE_FQDN=$PAPERSPACE_FQDN"
        --env "MAX_STORAGE_GB=$MAX_STORAGE_GB"
        --env "AUTO_CLEANUP=true"
        --env "NVIDIA_VISIBLE_DEVICES=all"
        --env "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
        --shm-size 2g
    )
    
    # Add Paperspace-specific environment variables
    if check_paperspace; then
        docker_args+=(--env "PAPERSPACE_NOTEBOOK_REPO_ID=${PAPERSPACE_NOTEBOOK_REPO_ID:-}")
    fi
    
    # Start container
    if docker run "${docker_args[@]}" "$IMAGE_NAME"; then
        log_success "Container started: $CONTAINER_NAME"
        return 0
    else
        log_error "Failed to start container"
        return 1
    fi
}

# Monitor container startup
monitor_startup() {
    log_info "Monitoring container startup..."
    
    local max_attempts=60
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if docker ps -f name="$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}" | grep -q "Up"; then
            break
        fi
        
        ((attempt++))
        log_info "Waiting for container to start... ($attempt/$max_attempts)"
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        log_error "Container failed to start within timeout"
        return 1
    fi
    
    # Wait for ComfyUI to be ready
    log_info "Waiting for ComfyUI to be ready..."
    attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -f "http://localhost:$COMFYUI_PORT/system_stats" &>/dev/null; then
            break
        fi
        
        ((attempt++))
        log_info "Waiting for ComfyUI to respond... ($attempt/$max_attempts)"
        sleep 3
    done
    
    if [ $attempt -eq $max_attempts ]; then
        log_warning "ComfyUI health check timeout, but container is running"
    else
        log_success "ComfyUI is ready!"
    fi
}

# Show access information
show_access_info() {
    log_header "=== ComfyUI Access Information ==="
    
    if check_paperspace; then
        echo -e "${GREEN}ComfyUI URL:${NC} https://${COMFYUI_PORT}-${PAPERSPACE_FQDN}/"
        echo -e "${GREEN}Tensorboard URL:${NC} https://tensorboard-${PAPERSPACE_FQDN}/"
    else
        echo -e "${GREEN}ComfyUI URL:${NC} http://localhost:${COMFYUI_PORT}"
        echo -e "${GREEN}Tensorboard URL:${NC} http://localhost:${TENSORBOARD_PORT}"
    fi
    
    echo -e "${BLUE}Container:${NC} $CONTAINER_NAME"
    echo -e "${BLUE}Image:${NC} $IMAGE_NAME"
    echo -e "${BLUE}Data Directory:${NC} $DATA_DIR"
    
    log_header "=== Quick Commands ==="
    echo "View logs:    docker logs -f $CONTAINER_NAME"
    echo "Stop:         docker stop $CONTAINER_NAME"
    echo "Restart:      docker restart $CONTAINER_NAME"
    echo "Shell:        docker exec -it $CONTAINER_NAME bash"
}

# Show container status
show_status() {
    log_info "Container Status:"
    
    if docker ps -f name="$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q "$CONTAINER_NAME"; then
        docker ps -f name="$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        log_success "Container is running"
    else
        log_warning "Container is not running"
    fi
}

# Main execution
main() {
    log_header "ComfyUI Docker Quick Start"
    
    case "${1:-start}" in
        "start")
            check_storage
            setup_directories
            ensure_image
            stop_container
            
            # Try Docker Compose first, fallback to manual
            if [ -f "$COMPOSE_FILE" ]; then
                if start_with_compose; then
                    monitor_startup
                    show_access_info
                else
                    log_warning "Docker Compose failed, trying manual start..."
                    start_container
                    monitor_startup
                    show_access_info
                fi
            else
                start_container
                monitor_startup
                show_access_info
            fi
            ;;
        "stop")
            log_info "Stopping ComfyUI services..."
            docker-compose -f "$COMPOSE_FILE" down || true
            stop_container
            log_success "Services stopped"
            ;;
        "restart")
            "$0" stop
            sleep 2
            "$0" start
            ;;
        "status")
            show_status
            ;;
        "logs")
            docker logs -f "$CONTAINER_NAME" 2>/dev/null || log_error "Container not found"
            ;;
        "shell")
            docker exec -it "$CONTAINER_NAME" bash || log_error "Container not running"
            ;;
        "build")
            build_image
            ;;
        "pull")
            pull_image
            ;;
        "clean")
            log_info "Cleaning up Docker resources..."
            docker-compose -f "$COMPOSE_FILE" down --volumes || true
            docker system prune -f
            log_success "Cleanup completed"
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|status|logs|shell|build|pull|clean}"
            echo ""
            echo "Commands:"
            echo "  start   - Start ComfyUI services (default)"
            echo "  stop    - Stop all services"
            echo "  restart - Restart services"
            echo "  status  - Show container status"
            echo "  logs    - Follow container logs"
            echo "  shell   - Open shell in container"
            echo "  build   - Build Docker image locally"
            echo "  pull    - Pull image from registry"
            echo "  clean   - Clean up Docker resources"
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"