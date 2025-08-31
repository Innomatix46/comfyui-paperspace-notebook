#!/bin/bash
# Fix ComfyUI dependency conflicts for Paperspace
# Solves NumPy, transformers, and aiofiles conflicts

echo "🔧 FIXING COMFYUI DEPENDENCY CONFLICTS"
echo "======================================"
echo "This will fix:"
echo "  • NumPy 2.x compatibility issues"
echo "  • Transformers pytree AttributeError"
echo "  • aiofiles version conflict"
echo "  • PyTorch version warning"
echo "======================================"
echo ""

# Function to check if we're in Paperspace
is_paperspace() {
    [ -n "$PAPERSPACE_FQDN" ] || [ -d "/notebooks" ]
}

# Set working directory
if is_paperspace; then
    echo "📍 Paperspace environment detected"
    COMFYUI_DIR="/notebooks/ComfyUI"
else
    echo "📍 Local environment"
    COMFYUI_DIR="$(pwd)/ComfyUI"
fi

cd "$COMFYUI_DIR" || exit 1

# Step 1: Fix NumPy version (downgrade to 1.x for compatibility)
echo "1️⃣ Fixing NumPy version conflict..."
echo "   Downgrading NumPy to 1.26.4 (last 1.x version)"
pip uninstall -y numpy 2>/dev/null
pip install --no-cache-dir "numpy<2.0" "numpy==1.26.4"
echo "✅ NumPy fixed"
echo ""

# Step 2: Fix transformers compatibility
echo "2️⃣ Fixing transformers compatibility..."
echo "   Downgrading transformers to compatible version"
pip uninstall -y transformers 2>/dev/null
pip install --no-cache-dir "transformers==4.36.2"
echo "✅ Transformers fixed"
echo ""

# Step 3: Fix aiofiles version
echo "3️⃣ Fixing aiofiles version conflict..."
pip uninstall -y aiofiles 2>/dev/null
pip install --no-cache-dir "aiofiles>=22.1.0,<23"
echo "✅ aiofiles fixed"
echo ""

# Step 4: Update PyTorch (optional but recommended)
echo "4️⃣ Checking PyTorch version..."
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
echo "   Current PyTorch: $PYTORCH_VERSION"

if [[ "$PYTORCH_VERSION" < "2.4" ]]; then
    echo "   ⚠️ PyTorch is outdated (recommended: 2.4+)"
    echo "   To update PyTorch, run:"
    echo "   pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121"
else
    echo "   ✅ PyTorch version is good"
fi
echo ""

# Step 5: Reinstall other critical packages with correct versions
echo "5️⃣ Reinstalling other critical packages..."
pip install --no-cache-dir --force-reinstall \
    "scipy<1.14" \
    "scikit-image" \
    "opencv-python" \
    "Pillow" \
    "safetensors" \
    "einops" \
    "torchsde" \
    "GitPython"
echo "✅ Critical packages reinstalled"
echo ""

# Step 6: Fix any remaining conflicts
echo "6️⃣ Checking for remaining conflicts..."
pip check 2>&1 | grep -E "(incompatible|conflict)" || echo "No conflicts found"
echo ""

# Step 7: Verify the fixes
echo "7️⃣ Verifying fixes..."
python << 'EOF'
import sys
errors = []

# Check NumPy
try:
    import numpy as np
    version = np.__version__
    if version.startswith('2.'):
        errors.append(f"NumPy still at 2.x: {version}")
    else:
        print(f"✅ NumPy: {version}")
except Exception as e:
    errors.append(f"NumPy import error: {e}")

# Check transformers
try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except Exception as e:
    errors.append(f"Transformers import error: {e}")

# Check torch
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    import torchvision
    print(f"✅ TorchVision: {torchvision.__version__}")
except Exception as e:
    errors.append(f"PyTorch import error: {e}")

# Check aiofiles
try:
    import aiofiles
    print(f"✅ aiofiles: {aiofiles.__version__}")
except Exception as e:
    errors.append(f"aiofiles import error: {e}")

if errors:
    print("\n❌ Errors found:")
    for error in errors:
        print(f"   {error}")
    sys.exit(1)
else:
    print("\n✅ All dependencies verified!")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "✅ DEPENDENCY FIXES COMPLETE!"
    echo "======================================"
    echo ""
    echo "ComfyUI should now start without errors."
    echo "Run: cd $COMFYUI_DIR && python main.py --listen --port 6006"
else
    echo ""
    echo "⚠️ Some issues remain. Please check the errors above."
fi