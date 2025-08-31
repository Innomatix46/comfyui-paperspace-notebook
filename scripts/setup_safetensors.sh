#!/bin/bash

# SafeTensors Model Loader Setup Script
# Sets up SafeTensors support for ComfyUI Paperspace environment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Setting up SafeTensors Model Loader..."
echo "Project root: $PROJECT_ROOT"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Python packages
install_python_packages() {
    echo "📦 Installing Python dependencies..."
    
    # Core SafeTensors dependencies
    pip install --upgrade safetensors
    pip install --upgrade numpy
    
    # Optional dependencies for enhanced functionality
    if ! python -c "import torch" 2>/dev/null; then
        echo "⚠️  PyTorch not found. Installing PyTorch for enhanced functionality..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo "✅ PyTorch already available"
    fi
    
    # Additional utilities
    pip install --upgrade huggingface_hub
    pip install --upgrade transformers
    
    echo "✅ Python dependencies installed"
}

# Function to setup directory structure
setup_directories() {
    echo "📁 Setting up directory structure..."
    
    # Create cache directory
    mkdir -p "$HOME/.cache/safetensors_loader"
    
    # Create ComfyUI model directories if they don't exist
    COMFYUI_MODELS_DIR="/workspace/ComfyUI/models"
    if [ ! -d "$COMFYUI_MODELS_DIR" ]; then
        echo "⚠️  ComfyUI models directory not found at $COMFYUI_MODELS_DIR"
        echo "Creating placeholder directory structure..."
        mkdir -p "$COMFYUI_MODELS_DIR"/{unet,vae,clip,loras,controlnet,embeddings}
    else
        echo "✅ ComfyUI models directory found"
        # Ensure subdirectories exist
        mkdir -p "$COMFYUI_MODELS_DIR"/{unet,vae,clip,loras,controlnet,embeddings}
    fi
    
    echo "✅ Directory structure ready"
}

# Function to test the installation
test_installation() {
    echo "🧪 Testing SafeTensors loader installation..."
    
    cd "$SCRIPT_DIR"
    
    # Test SafeTensors import
    if python -c "import safetensors; print('SafeTensors version:', safetensors.__version__)" 2>/dev/null; then
        echo "✅ SafeTensors library working"
    else
        echo "❌ SafeTensors library not working"
        return 1
    fi
    
    # Test our loader
    if python -c "from safetensors_loader import SafeTensorsLoader; print('SafeTensors loader imported successfully')" 2>/dev/null; then
        echo "✅ SafeTensors loader working"
    else
        echo "❌ SafeTensors loader not working"
        return 1
    fi
    
    # Test utilities
    if python -c "from safetensors_utils import SafeTensorsManager; print('SafeTensors utilities working')" 2>/dev/null; then
        echo "✅ SafeTensors utilities working"
    else
        echo "❌ SafeTensors utilities not working"
        return 1
    fi
    
    # Test unified loader
    if python -c "from model_loader_integration import UnifiedModelLoader; print('Unified loader working')" 2>/dev/null; then
        echo "✅ Unified model loader working"
    else
        echo "❌ Unified model loader not working"
        return 1
    fi
    
    echo "✅ All components working correctly"
}

# Function to create example usage scripts
create_examples() {
    echo "📝 Creating example usage scripts..."
    
    cat > "$SCRIPT_DIR/example_safetensors_usage.py" << 'EOF'
#!/usr/bin/env python3
"""
Example usage of SafeTensors loader
"""

from safetensors_loader import SafeTensorsLoader
from safetensors_utils import SafeTensorsManager
from model_loader_integration import UnifiedModelLoader
import os

def main():
    # Initialize loaders
    loader = SafeTensorsLoader()
    manager = SafeTensorsManager()
    unified = UnifiedModelLoader()
    
    print("SafeTensors Model Loader Examples")
    print("=" * 40)
    
    # Example 1: Scan for models
    print("\n1. Scanning for models...")
    try:
        models = manager.scan_models()
        total = sum(len(model_list) for model_list in models.values())
        print(f"Found {total} total models")
        
        for model_type, model_list in models.items():
            if model_list:
                print(f"  {model_type}: {len(model_list)} models")
    except Exception as e:
        print(f"Error scanning models: {e}")
    
    # Example 2: Analyze a specific model (if any exists)
    print("\n2. Model analysis example...")
    # This would work with an actual model file
    # model_info = loader.load_model("path/to/model.safetensors")
    # print(f"Model type: {model_info.model_type}")
    # print(f"Architecture: {model_info.architecture}")
    
    print("Examples completed!")

if __name__ == "__main__":
    main()
EOF
    
    chmod +x "$SCRIPT_DIR/example_safetensors_usage.py"
    
    # Create a model scanning script
    cat > "$SCRIPT_DIR/scan_models.sh" << 'EOF'
#!/bin/bash

# Quick model scanning script
echo "🔍 Scanning SafeTensors models..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Scan with unified loader
python model_loader_integration.py --command scan

echo "📊 Generating report..."
python model_loader_integration.py --command report --output "/tmp/models_report.json"

if [ -f "/tmp/models_report.json" ]; then
    echo "✅ Report saved to /tmp/models_report.json"
    echo "📋 Summary:"
    python -c "
import json
with open('/tmp/models_report.json') as f:
    data = json.load(f)
    summary = data.get('summary', {})
    print(f'Total models: {summary.get(\"total_models\", 0)}')
    print(f'Total size: {summary.get(\"total_size_gb\", 0):.1f} GB')
    print(f'Average compatibility: {summary.get(\"average_compatibility_score\", 0):.2f}')
"
else
    echo "⚠️  Report not generated"
fi
EOF
    
    chmod +x "$SCRIPT_DIR/scan_models.sh"
    
    echo "✅ Example scripts created"
}

# Function to show usage information
show_usage() {
    echo "🔧 SafeTensors Model Loader Usage"
    echo "=" * 40
    echo
    echo "CLI Tools:"
    echo "  ./safetensors_loader.py MODEL_PATH --command info"
    echo "  ./safetensors_utils.py --command scan --models-dir /path/to/models"
    echo "  ./model_loader_integration.py --command scan"
    echo
    echo "Quick Commands:"
    echo "  ./scan_models.sh                    # Quick model scan"
    echo "  ./example_safetensors_usage.py      # Example usage"
    echo
    echo "Python Usage:"
    echo "  from safetensors_loader import SafeTensorsLoader"
    echo "  loader = SafeTensorsLoader()"
    echo "  model_info = loader.load_model('model.safetensors')"
    echo
    echo "Features:"
    echo "  ✅ SafeTensors format support"
    echo "  ✅ Architecture detection (SDXL, SD3, FLUX, etc.)"
    echo "  ✅ Memory-mapped loading"
    echo "  ✅ Model optimization suggestions"
    echo "  ✅ Paperspace A6000 compatibility scoring"
    echo "  ✅ Unified GGUF + SafeTensors interface"
    echo
}

# Main execution
main() {
    echo "SafeTensors Model Loader Setup"
    echo "=============================="
    
    # Check Python availability
    if ! command_exists python; then
        echo "❌ Python not found. Please install Python 3.7+"
        exit 1
    fi
    
    # Check pip availability
    if ! command_exists pip; then
        echo "❌ pip not found. Please install pip"
        exit 1
    fi
    
    echo "✅ Python and pip available"
    
    # Install dependencies
    install_python_packages
    
    # Setup directories
    setup_directories
    
    # Test installation
    test_installation
    
    # Create examples
    create_examples
    
    echo
    echo "🎉 SafeTensors Model Loader setup complete!"
    echo
    
    # Show usage
    show_usage
    
    echo "💡 Tips for Paperspace:"
    echo "  - Models are cached in ~/.cache/safetensors_loader"
    echo "  - Watch the 50GB storage limit with large models"
    echo "  - Use FP16 models for better A6000 VRAM utilization"
    echo "  - Run ./scan_models.sh to analyze your model collection"
    echo
}

# Run main function
main "$@"