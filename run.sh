#!/bin/bash
# run.sh - Main ComfyUI Orchestration Script for Paperspace
# This is the primary entrypoint that handles the complete ComfyUI setup and launch process
# Now supports Docker mode for 1-2 minute deployments

# Exit immediately on any error to ensure robust operation
set -e

# =============================================================================
# EXECUTION MODE CONFIGURATION
# =============================================================================

# Check for Docker mode
DOCKER_MODE="${DOCKER_MODE:-false}"
USE_DOCKER="${USE_DOCKER:-$DOCKER_MODE}"

# Docker mode execution
if [ "$USE_DOCKER" = "true" ] || [ "$DOCKER_MODE" = "true" ]; then
    echo "=========================================="
    echo "🐳 ComfyUI Docker Mode - Fast Deployment"
    echo "=========================================="
    echo "Startup Time: 1-2 minutes (vs 15-20 traditional)"
    echo "Docker Image: Pre-built with all dependencies"
    echo "A6000 Optimized: CUDA 12.4 + Flash Attention"
    echo "=========================================="
    
    if [ -f "scripts/docker_quick_start.sh" ]; then
        echo "==> Launching Docker quick start..."
        exec "./scripts/docker_quick_start.sh" start
    else
        echo "❌ Docker quick start script not found"
        echo "==> Falling back to traditional setup..."
        echo ""
    fi
fi

# =============================================================================
# TRADITIONAL MODE ENVIRONMENT SETUP
# =============================================================================

# Define core project paths for consistent operation across all modules
export PROJECT_ROOT="$(pwd)"
export COMFYUI_DIR="$PROJECT_ROOT/ComfyUI"
export VENV_DIR="$PROJECT_ROOT/venv"
export STORAGE_DIR="/storage/ComfyUI"

# Create essential storage directories for persistent data
mkdir -p "$STORAGE_DIR/output"
mkdir -p "$STORAGE_DIR/input"
mkdir -p "$STORAGE_DIR/temp"

echo "=========================================="
echo "🚀 ComfyUI Paperspace Setup & Launch"
echo "🐌 Traditional Mode - Full Installation"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "ComfyUI Directory: $COMFYUI_DIR"
echo "Storage Directory: $STORAGE_DIR"
echo "Setup Time: 15-20 minutes (use DOCKER_MODE=true for 1-2 min)"
echo "=========================================="

# =============================================================================
# SOURCE MODULAR SCRIPTS
# =============================================================================

# Load the dependency installation module
# This handles ComfyUI cloning, venv setup, pip packages, and custom nodes
echo "==> Loading dependency installation module..."
source "$PROJECT_ROOT/scripts/install_dependencies.sh"

# Load the model download and linking module  
# This handles model downloads to persistent storage and symlink creation
echo "==> Loading model download module..."
source "$PROJECT_ROOT/scripts/download_models.sh"

# Load the JupyterLab configuration module
# This handles JupyterLab setup with root access for development
echo "==> Loading JupyterLab configuration module..."
source "$PROJECT_ROOT/scripts/configure_jupyterlab.sh"

# Load the auto-restart module
# This handles 6-hour automatic restarts with graceful shutdown
echo "==> Loading auto-restart module..."
source "$PROJECT_ROOT/scripts/auto_restart.sh"

# Load the storage optimizer module
# This handles Free Tier 50GB storage constraints and A6000 optimization
echo "==> Loading storage optimizer module..."
source "$PROJECT_ROOT/scripts/storage_optimizer.sh"

# =============================================================================
# NOTEBOOK REPAIR (Fix corrupted notebooks before starting)
# =============================================================================

echo ""
echo "🔧 Pre-Phase: Checking and Repairing Notebooks"
echo "=========================================="
if [ -f "$PROJECT_ROOT/fix_notebooks_paperspace.sh" ]; then
    bash "$PROJECT_ROOT/fix_notebooks_paperspace.sh"
else
    echo "Notebook repair script not found, skipping..."
fi

# =============================================================================
# EXECUTION FLOW
# =============================================================================

# Phase 1: Install all dependencies (ComfyUI, Python packages, custom nodes)
echo ""
echo "🔧 Phase 1: Installing Dependencies"
echo "=========================================="
install_dependencies

# Phase 2: Download models and create storage symlinks
echo ""
echo "📦 Phase 2: Processing Models"
echo "=========================================="
download_models

# Phase 2.5: Configure and start JupyterLab
echo ""
echo "🔬 Phase 2.5: Setting up JupyterLab"
echo "=========================================="
configure_jupyterlab
start_jupyterlab

# Phase 2.6: Setup auto-restart scheduler
echo ""
echo "⏰ Phase 2.6: Setting up Auto-Restart (6h intervals)"
echo "=========================================="
setup_auto_restart "$@"

# Phase 2.7: A6000 + Free Tier optimization
echo ""
echo "🚀 Phase 2.7: A6000 + Free Tier Optimization (50GB)"
echo "=========================================="
show_storage_status
optimize_for_a6000

# Phase 3: Prepare application environment
echo ""
echo "🎯 Phase 3: Preparing Application Launch"
echo "=========================================="

# Activate the Python virtual environment for ComfyUI execution
echo "==> Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Navigate to ComfyUI directory for proper execution context
echo "==> Changing to ComfyUI directory..."
cd "$COMFYUI_DIR"

# =============================================================================
# APPLICATION LAUNCH CONFIGURATION
# =============================================================================

echo ""
echo "🌟 Phase 4: Launching ComfyUI"
echo "=========================================="

# Construct ComfyUI launch arguments optimized for A6000 + Free Tier + Paperspace
COMFYUI_ARGS=(
    --listen                                    # Accept connections from any IP
    --port 6006                                # Paperspace Tensorboard port (for external access)
    --preview-method auto                      # Automatic preview generation
    --output-directory "$STORAGE_DIR/output"   # Persistent output storage
    --input-directory "$STORAGE_DIR/input"     # Persistent input storage
    --temp-directory "$STORAGE_DIR/temp"       # Persistent temp storage
    --enable-cors-header                       # CORS support for external access
    --disable-metadata                         # Save storage space (skip metadata)
    --lowvram                                  # Optimize for variable VRAM usage
    --use-split-cross-attention                # A6000 memory optimization
)

# Display the constructed launch command for transparency
echo "==> Launch command: python main.py ${COMFYUI_ARGS[*]}"

# =============================================================================
# PAPERSPACE INTEGRATION
# =============================================================================

# Detect and display the Paperspace public URL for easy access
if [ -n "$PAPERSPACE_FQDN" ]; then
    echo ""
    echo "🌐 ComfyUI Access Information"
    echo "=========================================="
    echo "🔗 ComfyUI URL: https://tensorboard-$PAPERSPACE_FQDN/"
    echo "🔗 JupyterLab URL: https://$PAPERSPACE_FQDN/lab/"
    echo ""
    echo "📋 Click the URLs above to access your instances!"
    echo "🚀 A6000 GPU: 48GB VRAM optimized with Flash Attention"
    echo "💾 Free Tier: 50GB storage with intelligent management"
    echo "⏰ Auto-restart: Every 6 hours (logs at /storage/ComfyUI/restart.log)"
    echo "📊 Storage monitor: ./scripts/storage_optimizer.sh status"
    echo "🎯 Port 6006: Uses Paperspace Tensorboard URL mapping"
    echo "=========================================="
else
    echo "==> Note: PAPERSPACE_FQDN not detected"
    echo "==> ComfyUI: http://localhost:6006 (A6000 optimized, Tensorboard port)"
    echo "==> JupyterLab: http://localhost:8889 (root access)"
    echo "==> Storage: 50GB Free Tier with monitoring"
    echo "==> Auto-restart: Every 6 hours"
fi

# =============================================================================
# APPLICATION EXECUTION
# =============================================================================

# Launch ComfyUI with all configured arguments
echo "==> Starting ComfyUI application..."
echo "==> Press Ctrl+C to stop the application"
echo ""

# Execute ComfyUI main application with constructed arguments
exec python main.py "${COMFYUI_ARGS[@]}"