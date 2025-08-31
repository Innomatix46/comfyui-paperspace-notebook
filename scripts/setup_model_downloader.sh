#!/bin/bash
# setup_model_downloader.sh - Setup Universal Model Downloader dependencies

setup_model_downloader() {
    echo "🎯 Setting up Universal Model Downloader..."
    
    # Install Python dependencies
    echo "==> Installing Python dependencies..."
    
    # Core dependencies
    pip3 install --upgrade \
        aiohttp \
        aiofiles \
        tqdm \
        requests \
        safetensors \
        || echo "⚠️ Some core dependencies failed to install"
    
    # Optional dependencies for format conversion
    echo "==> Installing optional format conversion dependencies..."
    pip3 install --upgrade \
        torch \
        transformers \
        llama-cpp-python \
        || echo "⚠️ Some conversion dependencies failed to install (optional features may be limited)"
    
    # Install aria2c for accelerated downloads
    echo "==> Installing aria2c for accelerated downloads..."
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update && apt-get install -y aria2
    elif command -v brew >/dev/null 2>&1; then
        brew install aria2
    elif command -v yum >/dev/null 2>&1; then
        yum install -y aria2
    else
        echo "⚠️ Could not install aria2c automatically. Install manually for better download speeds."
    fi
    
    # Create necessary directories
    echo "==> Creating directory structure..."
    mkdir -p /storage/ComfyUI/models/{checkpoints,vae,text_encoders,loras,controlnet,upscalers,embeddings,ipadapter,clip,motion}
    mkdir -p ComfyUI/models
    mkdir -p configs
    
    # Set permissions
    chmod +x scripts/universal_model_downloader.py
    chmod +x scripts/model_converter.py
    
    # Test installation
    echo "==> Testing installation..."
    if python3 scripts/universal_model_downloader.py --help >/dev/null 2>&1; then
        echo "✅ Universal Model Downloader installed successfully!"
    else
        echo "❌ Installation test failed"
        return 1
    fi
    
    if python3 scripts/model_converter.py --help >/dev/null 2>&1; then
        echo "✅ Model Converter installed successfully!"
    else
        echo "❌ Model Converter installation test failed"
    fi
    
    echo ""
    echo "🚀 Setup complete! Usage examples:"
    echo ""
    echo "📥 Interactive Mode:"
    echo "   python3 scripts/universal_model_downloader.py --interactive"
    echo ""
    echo "🔍 Search Models:"
    echo "   python3 scripts/universal_model_downloader.py --search \"sdxl\""
    echo ""
    echo "📋 List Models:"
    echo "   python3 scripts/universal_model_downloader.py --list --recommended"
    echo ""
    echo "💾 Storage Info:"
    echo "   python3 scripts/universal_model_downloader.py --storage-info"
    echo ""
    echo "🔄 Format Conversion:"
    echo "   python3 scripts/model_converter.py model.safetensors -f gguf -q Q4_0"
    echo ""
    echo "📖 Full documentation available in the interactive mode"
}

# Run setup if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_model_downloader
fi