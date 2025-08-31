#!/bin/bash
# fix_dependencies.sh - Quick fix for missing ComfyUI dependencies

echo "==> Fixing missing ComfyUI dependencies..."

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    echo "==> Activating virtual environment..."
    source venv/bin/activate
fi

# Install missing audio and SDE packages
echo "==> Installing torchaudio with CUDA support..."
pip install torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "==> Installing torchsde for sampling..."
pip install torchsde

echo "==> Installing additional dependencies..."
pip install scipy einops

echo "==> Installing av (PyAV) for video support..."
pip install av

echo "==> Installing additional ComfyUI dependencies..."
# Common missing packages that cause issues
pip install opencv-python-headless matplotlib scikit-image
pip install audio-separator blend-modes LangSegment onnx scepter albucore==0.0.16
pip install sageattention blinker mmcv

# Verify installations
echo "==> Verifying installations..."
python -c "import torchaudio; print(f'✅ torchaudio {torchaudio.__version__} installed')" || echo "❌ torchaudio installation failed"
python -c "import torchsde; print('✅ torchsde installed')" || echo "❌ torchsde installation failed"
python -c "import scipy; print(f'✅ scipy {scipy.__version__} installed')" || echo "❌ scipy installation failed"
python -c "import einops; print(f'✅ einops installed')" || echo "❌ einops installation failed"

echo "==> Dependency fix completed!"
echo "==> You can now restart ComfyUI"