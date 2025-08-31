#!/bin/bash
# quick_video_fix.sh - Quick fix for av module (video support)

echo "==> Quick fix for missing av module..."

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    echo "==> Activating virtual environment..."
    source venv/bin/activate
fi

# Install av (PyAV) for video support
echo "==> Installing av (PyAV) for video support..."
pip install -U av || {
    echo "==> Trying with system packages..."
    # On some systems, need ffmpeg libraries first
    apt-get update && apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev || true
    pip install -U av
}

# Verify installation
echo "==> Verifying av installation..."
python -c "import av; print(f'✅ PyAV {av.__version__} installed')" || echo "❌ av installation failed"

echo "==> Quick fix completed! You can now restart ComfyUI"