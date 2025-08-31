#!/bin/bash
# install_docker.sh - Automatische Docker-Installation für ComfyUI

set -e

echo "🚀 ComfyUI Docker Quick Installer"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }

# Detect environment
detect_environment() {
    if [ ! -z "$PAPERSPACE_CLUSTER_ID" ] || [ -f "/notebooks/.paperspace" ]; then
        echo "📍 Paperspace environment detected"
        ENVIRONMENT="paperspace"
        WORK_DIR="/notebooks/comfyui-paperspace-notebook"
    else
        echo "📍 Local/Standard environment detected"
        ENVIRONMENT="local"
        WORK_DIR="$(pwd)"
    fi
}

# Check Docker installation
check_docker() {
    if command -v docker &> /dev/null; then
        print_status "Docker is installed: $(docker --version)"
        return 0
    else
        print_error "Docker is not installed"
        echo "Please install Docker first:"
        echo "  - Ubuntu/Debian: sudo apt-get install docker.io docker-compose"
        echo "  - Or visit: https://docs.docker.com/get-docker/"
        return 1
    fi
}

# Check GPU availability
check_gpu() {
    echo ""
    echo "🔍 Checking GPU availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
            print_status "GPU detected: $GPU_NAME"
            GPU_AVAILABLE=true
        else
            print_warning "nvidia-smi found but no GPU detected"
            GPU_AVAILABLE=false
        fi
    else
        print_warning "nvidia-smi not found - GPU support may not be available"
        GPU_AVAILABLE=false
    fi
}

# Clone or update repository
setup_repository() {
    echo ""
    echo "📦 Setting up repository..."
    
    if [ -d "$WORK_DIR" ]; then
        print_status "Repository exists at $WORK_DIR"
        cd "$WORK_DIR"
        
        # Update repository
        echo "Updating repository..."
        git pull origin master 2>/dev/null || print_warning "Could not update repository"
    else
        print_status "Cloning repository..."
        git clone https://github.com/Innomatix46/comfyui-paperspace-notebook.git "$WORK_DIR"
        cd "$WORK_DIR"
    fi
}

# Build or pull Docker image
setup_docker_image() {
    echo ""
    echo "🐳 Setting up Docker image..."
    echo ""
    echo "Choose an option:"
    echo "1) Pull pre-built image (faster, ~2 min)"
    echo "2) Build from source (slower, ~10 min)"
    echo "3) Skip (use existing image)"
    echo ""
    read -p "Select option [1-3]: " -n 1 -r
    echo ""
    
    case $REPLY in
        1)
            print_status "Pulling pre-built image..."
            docker pull innomatix46/comfyui-paperspace:optimized || {
                print_error "Failed to pull image"
                echo "Trying alternative registry..."
                docker pull ghcr.io/innomatix46/comfyui-paperspace:optimized || {
                    print_error "Could not pull image from any registry"
                    return 1
                }
            }
            docker tag innomatix46/comfyui-paperspace:optimized comfyui-paperspace:optimized
            print_status "Image pulled successfully"
            ;;
        2)
            print_status "Building image from source..."
            cd docker
            docker build -f Dockerfile.optimized -t comfyui-paperspace:optimized . || {
                print_error "Build failed"
                return 1
            }
            cd ..
            print_status "Image built successfully"
            ;;
        3)
            print_warning "Skipping image setup"
            ;;
        *)
            print_error "Invalid option"
            return 1
            ;;
    esac
}

# Create environment file
create_env_file() {
    echo ""
    echo "⚙️ Creating environment configuration..."
    
    ENV_FILE="docker/.env"
    
    if [ -f "$ENV_FILE" ]; then
        print_warning "Environment file already exists"
        read -p "Overwrite? (y/n): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    cat > "$ENV_FILE" << EOF
# ComfyUI Docker Environment Configuration
# Generated: $(date)

# GPU Settings
NVIDIA_VISIBLE_DEVICES=all
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
CUDA_LAUNCH_BLOCKING=0

# Storage Paths
STORAGE_PATH=/storage
OUTPUT_PATH=/storage/output
TEMP_STORAGE=/temp-storage

# Network Ports
COMFYUI_PORT=8188
TENSORBOARD_PORT=6006
JUPYTER_PORT=8888
NGINX_PORT=80
NGINX_SSL_PORT=443

# Resource Limits
MEMORY_LIMIT=48G
MEMORY_RESERVATION=8G
SHM_SIZE=8gb

# Paperspace Settings (auto-detected)
PAPERSPACE_CLUSTER_ID=${PAPERSPACE_CLUSTER_ID:-}
PAPERSPACE_FQDN=${PAPERSPACE_FQDN:-}

# Database (optional)
DB_USER=comfyui
DB_PASSWORD=comfyui_pass_$(openssl rand -hex 8)

# Custom settings
CUSTOM_COMMAND=
EOF
    
    print_status "Environment file created at $ENV_FILE"
}

# Start ComfyUI with Docker
start_comfyui() {
    echo ""
    echo "🚀 Starting ComfyUI..."
    echo ""
    echo "Choose startup mode:"
    echo "1) Basic (ComfyUI only)"
    echo "2) With Nginx proxy"
    echo "3) Full stack (Nginx + Redis + PostgreSQL)"
    echo ""
    read -p "Select mode [1-3]: " -n 1 -r
    echo ""
    
    cd docker
    
    case $REPLY in
        1)
            print_status "Starting ComfyUI (basic mode)..."
            docker-compose -f docker-compose.optimized.yml up -d comfyui
            ;;
        2)
            print_status "Starting ComfyUI with Nginx proxy..."
            docker-compose -f docker-compose.optimized.yml \
                --profile with-proxy \
                up -d
            ;;
        3)
            print_status "Starting full stack..."
            docker-compose -f docker-compose.optimized.yml \
                --profile with-proxy \
                --profile with-cache \
                --profile with-database \
                up -d
            ;;
        *)
            print_error "Invalid option"
            return 1
            ;;
    esac
    
    cd ..
    
    # Wait for container to start
    echo ""
    echo "⏳ Waiting for ComfyUI to start..."
    sleep 5
    
    # Check container status
    if docker ps | grep -q comfyui; then
        print_status "ComfyUI is running!"
        
        # Show logs
        echo ""
        echo "📋 Recent logs:"
        docker logs comfyui-optimized --tail 10
        
        # Show access URLs
        echo ""
        echo "🌐 Access URLs:"
        if [ "$ENVIRONMENT" = "paperspace" ]; then
            echo "  ComfyUI: https://8188-${PAPERSPACE_FQDN}/"
            echo "  Tensorboard: https://tensorboard-${PAPERSPACE_FQDN}/"
        else
            echo "  ComfyUI: http://localhost:8188"
            echo "  Tensorboard: http://localhost:6006"
        fi
    else
        print_error "ComfyUI container is not running"
        echo "Check logs with: docker logs comfyui-optimized"
        return 1
    fi
}

# Show management commands
show_commands() {
    echo ""
    echo "📚 Useful Commands:"
    echo "=================================="
    echo ""
    echo "# Container Management:"
    echo "docker ps                          # Show running containers"
    echo "docker logs comfyui-optimized      # View logs"
    echo "docker exec -it comfyui-optimized bash  # Enter container"
    echo "docker restart comfyui-optimized   # Restart container"
    echo "docker stop comfyui-optimized      # Stop container"
    echo ""
    echo "# GPU Monitoring:"
    echo "docker exec comfyui-optimized nvidia-smi  # Check GPU"
    echo ""
    echo "# Update Image:"
    echo "docker pull innomatix46/comfyui-paperspace:optimized"
    echo ""
    echo "# Clean Up:"
    echo "docker system prune -a             # Remove unused images"
    echo ""
}

# Main installation flow
main() {
    echo "Starting installation process..."
    echo ""
    
    # Step 1: Detect environment
    detect_environment
    
    # Step 2: Check Docker
    if ! check_docker; then
        exit 1
    fi
    
    # Step 3: Check GPU
    check_gpu
    
    # Step 4: Setup repository
    setup_repository
    
    # Step 5: Setup Docker image
    if ! setup_docker_image; then
        exit 1
    fi
    
    # Step 6: Create environment file
    create_env_file
    
    # Step 7: Start ComfyUI
    if ! start_comfyui; then
        exit 1
    fi
    
    # Step 8: Show commands
    show_commands
    
    echo ""
    echo "=================================="
    print_status "Installation complete! 🎉"
    echo "=================================="
    echo ""
    echo "ComfyUI is now running in Docker!"
    echo ""
    
    if [ "$GPU_AVAILABLE" = true ]; then
        print_status "GPU acceleration is enabled"
    else
        print_warning "Running in CPU mode (slow performance)"
    fi
}

# Run main function
main