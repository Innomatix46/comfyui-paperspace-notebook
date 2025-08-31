#!/bin/bash

# Test Docker Setup for ComfyUI Paperspace
# Validates all Docker components before deployment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Test counters
TESTS_RUN=0
TESTS_PASSED=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    ((TESTS_RUN++))
    log_info "Running: $test_name"
    
    if eval "$test_command" &>/dev/null; then
        log_success "$test_name"
        ((TESTS_PASSED++))
    else
        log_error "$test_name"
    fi
}

# Test Docker availability
test_docker() {
    log_info "Testing Docker environment..."
    
    run_test "Docker installed" "command -v docker"
    run_test "Docker daemon running" "docker info"
    run_test "Docker Compose available" "command -v docker-compose"
    
    if docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        log_success "NVIDIA Docker runtime available"
        ((TESTS_PASSED++))
    else
        log_warning "NVIDIA Docker runtime not available (GPU tests will be skipped)"
    fi
    ((TESTS_RUN++))
}

# Test file structure
test_files() {
    log_info "Testing file structure..."
    
    local required_files=(
        "docker/Dockerfile.comfyui"
        "docker/docker-compose.yml"
        "docker/build_cache.sh"
        "docker/entrypoint.sh"
        "docker/.dockerignore"
        "scripts/docker_quick_start.sh"
    )
    
    for file in "${required_files[@]}"; do
        run_test "File exists: $file" "[ -f '$PROJECT_ROOT/$file' ]"
    done
}

# Test script permissions
test_permissions() {
    log_info "Testing script permissions..."
    
    local executable_files=(
        "docker/build_cache.sh"
        "docker/entrypoint.sh"
        "scripts/docker_quick_start.sh"
    )
    
    for file in "${executable_files[@]}"; do
        run_test "Executable: $file" "[ -x '$PROJECT_ROOT/$file' ]"
    done
}

# Test Docker build context
test_build_context() {
    log_info "Testing Docker build context..."
    
    cd "$PROJECT_ROOT/docker"
    
    # Check .dockerignore syntax
    run_test "dockerignore syntax" "docker build --dry-run -f Dockerfile.comfyui . 2>&1 | grep -v 'ERROR'"
    
    # Estimate build context size
    local context_size
    context_size=$(find . -name "*" -not -path "./.git/*" | wc -l)
    
    if [ "$context_size" -lt 1000 ]; then
        log_success "Build context size reasonable ($context_size files)"
        ((TESTS_PASSED++))
    else
        log_warning "Build context may be large ($context_size files)"
    fi
    ((TESTS_RUN++))
}

# Test Dockerfile syntax
test_dockerfile() {
    log_info "Testing Dockerfile syntax..."
    
    cd "$PROJECT_ROOT/docker"
    
    # Basic syntax check
    run_test "Dockerfile syntax" "docker build --dry-run -f Dockerfile.comfyui --target base ."
    
    # Check for required elements
    run_test "CUDA base image" "grep -q 'nvidia/cuda:12.4' Dockerfile.comfyui"
    run_test "Multi-stage build" "grep -q 'FROM.*AS' Dockerfile.comfyui"
    run_test "Health check" "grep -q 'HEALTHCHECK' Dockerfile.comfyui"
    run_test "Non-root user" "grep -q 'USER comfyui' Dockerfile.comfyui"
}

# Test docker-compose configuration
test_compose() {
    log_info "Testing Docker Compose configuration..."
    
    cd "$PROJECT_ROOT/docker"
    
    # Syntax validation
    run_test "Compose file syntax" "docker-compose config"
    
    # Check for required services
    run_test "ComfyUI service defined" "docker-compose config | grep -q 'comfyui:'"
    run_test "GPU configuration" "docker-compose config | grep -q 'devices:'"
    run_test "Volume mounts" "docker-compose config | grep -q 'volumes:'"
}

# Test scripts functionality
test_scripts() {
    log_info "Testing script functionality..."
    
    # Test build cache script
    cd "$PROJECT_ROOT/docker"
    run_test "Build cache help" "./build_cache.sh 2>&1 | grep -q 'Usage'"
    
    # Test quick start script
    cd "$PROJECT_ROOT/scripts"
    run_test "Quick start help" "./docker_quick_start.sh 2>&1 | grep -q 'Usage'"
    
    # Test entrypoint script
    cd "$PROJECT_ROOT/docker"
    run_test "Entrypoint syntax" "bash -n entrypoint.sh"
}

# Test data directory structure
test_data_structure() {
    log_info "Testing data directory structure..."
    
    local required_dirs=(
        "data/models/checkpoints"
        "data/models/vae"
        "data/models/loras" 
        "data/models/controlnet"
        "data/output"
        "data/input"
        "data/logs"
    )
    
    for dir in "${required_dirs[@]}"; do
        run_test "Directory exists: $dir" "[ -d '$PROJECT_ROOT/$dir' ]"
    done
}

# Test environment configuration
test_environment() {
    log_info "Testing environment configuration..."
    
    # Test .env.example
    run_test ".env.example exists" "[ -f '$PROJECT_ROOT/docker/.env.example' ]"
    
    # Check for required variables
    if [ -f "$PROJECT_ROOT/docker/.env.example" ]; then
        run_test "Paperspace config" "grep -q 'PAPERSPACE_FQDN' '$PROJECT_ROOT/docker/.env.example'"
        run_test "Storage config" "grep -q 'MAX_STORAGE_GB' '$PROJECT_ROOT/docker/.env.example'"
        run_test "GPU config" "grep -q 'CUDA_VISIBLE_DEVICES' '$PROJECT_ROOT/docker/.env.example'"
    fi
}

# Test integration with existing setup
test_integration() {
    log_info "Testing integration with existing setup..."
    
    # Test run.sh Docker mode
    run_test "run.sh Docker support" "grep -q 'DOCKER_MODE' '$PROJECT_ROOT/run.sh'"
    run_test "run.sh fallback" "grep -q 'docker_quick_start.sh' '$PROJECT_ROOT/run.sh'"
    
    # Test script sourcing
    if [ -f "$PROJECT_ROOT/scripts/install_dependencies.sh" ]; then
        log_success "Existing scripts preserved"
        ((TESTS_PASSED++))
    else
        log_warning "Original scripts not found (expected for Docker-only setup)"
    fi
    ((TESTS_RUN++))
}

# Show final results
show_results() {
    echo ""
    echo "==========================================="
    echo "🧪 Docker Setup Test Results"
    echo "==========================================="
    echo "Tests Run: $TESTS_RUN"
    echo "Tests Passed: $TESTS_PASSED"
    echo "Success Rate: $((TESTS_PASSED * 100 / TESTS_RUN))%"
    
    if [ $TESTS_PASSED -eq $TESTS_RUN ]; then
        log_success "All tests passed! Docker setup is ready for deployment."
        echo ""
        echo "🚀 Next Steps:"
        echo "1. Build image: ./docker/build_cache.sh build"
        echo "2. Start services: ./scripts/docker_quick_start.sh start"
        echo "3. Or use Docker mode: DOCKER_MODE=true ./run.sh"
        return 0
    elif [ $TESTS_PASSED -gt $((TESTS_RUN * 3 / 4)) ]; then
        log_warning "Most tests passed. Docker setup should work with minor issues."
        return 1
    else
        log_error "Many tests failed. Please fix issues before deployment."
        return 2
    fi
}

# Main execution
main() {
    echo "==========================================="
    echo "🐳 ComfyUI Docker Setup Validation"
    echo "==========================================="
    echo "Project: $PROJECT_ROOT"
    echo "Timestamp: $(date)"
    echo ""
    
    test_docker
    test_files
    test_permissions
    test_build_context
    test_dockerfile
    test_compose
    test_scripts
    test_data_structure
    test_environment
    test_integration
    
    show_results
}

# Execute main function
main "$@"